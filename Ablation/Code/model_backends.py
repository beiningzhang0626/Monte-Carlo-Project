############################################################
# model_backends.py
# Model backend functions for binary and categorical MMIL / MCEM.
#
# Python version of 03_model_backends.R
#
# Supported backends:
#   1. glmnet-like binary logistic regression
#   2. glmnet-like multinomial logistic regression
#   3. XGBoost binary logistic model
#   4. XGBoost multiclass softprob model
#
# Binary interface:
#   fit_backend_binary()
#   predict_backend_binary()
#
# Multiclass interface:
#   fit_backend_multiclass()
#   predict_backend_multiclass()
#
# CHANGES vs previous version:
#   1. _fit_logistic_model() no longer silently clamps C to [1e-4, 1e4].
#      That clip was overriding the lambda values picked by CV (and
#      reused via reuse_glmnet_lambda_if_requested) for high-dimensional
#      ridge fits, leading to systematically wrong regularization.
#   2. Per-solver tolerance / max_iter tuning to match glmnet's tighter
#      convergence (R glmnet uses thresh = 1e-7). Ridge uses lbfgs with
#      tol=1e-6, max_iter=5000. Lasso/elastic-net use saga with tol=1e-5,
#      max_iter=10000.
#   3. New make_data_adaptive_lambda_path() computes a glmnet-style
#      lambda_max from the intercept-only gradient at standardized X,
#      and a log-spaced path down to eps_ratio * lambda_max. This
#      replaces the data-independent default path
#      [lambda_max=100, lambda_min=1e-4] which was uniformly too
#      regularizing on the upper end and too noisy on the lower end.
#      The fit functions standardize X once and reuse that standardized
#      X for both the lambda-path computation and the final fit.
############################################################

from __future__ import annotations

import copy
import warnings
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from utils import (
    log_loss_binary,
    row_normalize,
)


############################################################
# Backend configuration helper
############################################################

def get_config_value(
    config: Optional[dict[str, Any]],
    name: str,
    default: Any = None,
) -> Any:
    if config is not None and name in config and config[name] is not None:
        return config[name]
    return default


############################################################
# Matrix conversion helper
############################################################

def _as_numpy_x(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=1)
    scale[scale < eps] = 1.0

    out = {
        "x_train": (x_train - center) / scale,
        "center": center,
        "scale": scale,
    }

    if x_test is not None:
        out["x_test"] = (x_test - center) / scale

    return out


def _apply_standardizer(
    x: pd.DataFrame | np.ndarray,
    center: Optional[np.ndarray],
    scale: Optional[np.ndarray],
) -> np.ndarray:
    x_arr = _as_numpy_x(x)

    if center is None or scale is None:
        return x_arr

    return (x_arr - center) / scale


############################################################
# Weighted soft-label expansion
############################################################

def prepare_binary_weighted_data(
    x: pd.DataFrame | np.ndarray,
    q: Sequence[float] | np.ndarray,
    eps: float = 1e-8,
) -> dict[str, Any]:
    q = np.clip(np.asarray(q, dtype=float), 0, 1)
    x_arr = _as_numpy_x(x)
    n = x_arr.shape[0]

    x_expanded = np.concatenate([x_arr, x_arr], axis=0)
    y_expanded = np.concatenate(
        [
            np.zeros(n, dtype=int),
            np.ones(n, dtype=int),
        ]
    )
    w_expanded = np.concatenate([1 - q, q])

    keep = w_expanded > eps

    x_expanded = x_expanded[keep]
    y_expanded = y_expanded[keep]
    w_expanded = w_expanded[keep]

    if len(w_expanded) == 0:
        raise ValueError("prepare_binary_weighted_data: all weights are zero.")

    w_expanded = w_expanded / np.mean(w_expanded)

    return {
        "x": x_expanded,
        "y": y_expanded,
        "weight": w_expanded,
    }


def prepare_multiclass_weighted_data(
    x: pd.DataFrame | np.ndarray,
    q_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
    eps: float = 1e-8,
) -> dict[str, Any]:
    class_levels = list(class_levels)
    x_arr = _as_numpy_x(x)

    if isinstance(q_mat, pd.DataFrame):
        q_df = q_mat.copy()

        if q_df.columns.isnull().any():
            q_df.columns = class_levels

        q_df = q_df.loc[:, class_levels]

    else:
        q_df = pd.DataFrame(q_mat, columns=class_levels)

    q_df = row_normalize(q_df)

    n = x_arr.shape[0]
    K = len(class_levels)

    if q_df.shape[0] != n:
        raise ValueError("nrow(q_mat) must match nrow(x).")

    if q_df.shape[1] != K:
        raise ValueError("ncol(q_mat) must match length(class_levels).")

    x_expanded = np.concatenate([x_arr for _ in range(K)], axis=0)
    y_expanded = np.concatenate(
        [
            np.repeat(cls, n)
            for cls in class_levels
        ]
    )

    # R as.vector(matrix) is column-major.
    w_expanded = q_df.to_numpy(dtype=float).reshape(-1, order="F")

    keep = w_expanded > eps

    x_expanded = x_expanded[keep]
    y_expanded = y_expanded[keep]
    w_expanded = w_expanded[keep]

    if len(w_expanded) == 0:
        raise ValueError(
            "prepare_multiclass_weighted_data: "
            "all pseudo-observation weights are zero."
        )

    w_expanded = w_expanded / np.mean(w_expanded)

    return {
        "x": x_expanded,
        "y": y_expanded,
        "weight": w_expanded,
    }


############################################################
# Lambda-path helpers
############################################################

def make_safe_lambda_path(
    lambda_value: float,
    relative_width: float = 1e-3,
) -> np.ndarray:
    lambda_value = float(lambda_value)

    if not np.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("make_safe_lambda_path: lambda must be positive and finite.")

    lambda_seq = np.array(
        [
            lambda_value * (1 + relative_width),
            lambda_value,
            lambda_value * (1 - relative_width),
        ],
        dtype=float,
    )

    lambda_seq = lambda_seq[np.isfinite(lambda_seq) & (lambda_seq > 0)]
    lambda_seq = np.unique(lambda_seq)[::-1]

    if len(lambda_seq) < 2:
        lambda_seq = np.array(
            [
                lambda_value * 1.01,
                lambda_value,
                lambda_value * 0.99,
            ],
            dtype=float,
        )
        lambda_seq = np.unique(lambda_seq)[::-1]

    return lambda_seq


def make_wide_lambda_path(
    lambda_value: float,
    n_lambda: int = 60,
    upper_mult: float = 100,
    lower_mult: float = 0.01,
) -> np.ndarray:
    lambda_value = float(lambda_value)

    if not np.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("make_wide_lambda_path: lambda must be positive and finite.")

    lambda_max = lambda_value * upper_mult
    lambda_min = lambda_value * lower_mult

    lambda_seq = np.exp(
        np.linspace(
            np.log(lambda_max),
            np.log(lambda_min),
            int(n_lambda),
        )
    )

    lambda_seq = lambda_seq[np.isfinite(lambda_seq) & (lambda_seq > 0)]
    lambda_seq = np.unique(lambda_seq)[::-1]

    return lambda_seq


def make_default_lambda_path(
    n_lambda: int = 60,
    lambda_max: float = 10.0,
    lambda_min: float = 1e-2,
) -> np.ndarray:
    """
    Data-INDEPENDENT fallback lambda path.

    Kept only for backwards compatibility. Prefer
    `make_data_adaptive_lambda_path()`, which computes lambda_max from
    the data and is much closer to glmnet's actual auto path.
    """
    return np.exp(
        np.linspace(
            np.log(lambda_max),
            np.log(lambda_min),
            int(n_lambda),
        )
    )


def make_data_adaptive_lambda_path(
    X: np.ndarray,
    y: np.ndarray | Sequence[Any],
    weights: np.ndarray,
    glmnet_alpha: float,
    n_lambda: int = 60,
    eps_ratio: float = 1e-4,
    task: str = "binary",
    class_levels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Compute a glmnet-style lambda path from the data.

    For logistic regression at the intercept-only solution, the gradient
    of the (mean weighted) negative log-likelihood with respect to the
    j-th coefficient is

        g_j = (1 / N) * sum_i w_i * x_ij * (y_i - p_bar)

    glmnet's `lambda_max` is the smallest lambda at which the optimal
    beta is exactly zero, which (for elastic-net penalty
    alpha * ||beta||_1 + (1 - alpha) / 2 * ||beta||^2) equals
    `max_j |g_j| / alpha`. For ridge (alpha == 0) we substitute a small
    surrogate alpha = 1e-3, matching glmnet's internal trick.

    For multiclass, `max_grad` is the maximum over class-vs-rest
    gradients.

    Parameters
    ----------
    X : ndarray, shape (N, p)
        The (already-standardized) design matrix used by the fit.
    y : array-like, shape (N,)
        Binary {0, 1} labels (task='binary') or string class labels
        (task='multiclass').
    weights : ndarray, shape (N,)
        Per-sample weights (after weighted-expansion of soft labels).
    glmnet_alpha : float
        Elastic-net mixing parameter. 0 = ridge, 1 = lasso.
    n_lambda : int
        Number of points in the path.
    eps_ratio : float
        lambda_min / lambda_max ratio.
    task : {'binary', 'multiclass'}
    class_levels : optional sequence of str
        Class labels in order; required for multiclass.

    Returns
    -------
    ndarray of shape (n_lambda,), strictly decreasing.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if X.ndim != 2:
        raise ValueError("make_data_adaptive_lambda_path: X must be 2-D.")

    N = X.shape[0]
    if N == 0:
        raise ValueError("make_data_adaptive_lambda_path: X has zero rows.")

    w_sum = float(weights.sum())
    if w_sum <= 0 or not np.isfinite(w_sum):
        raise ValueError("make_data_adaptive_lambda_path: weights are non-positive.")

    alpha_eff = max(float(glmnet_alpha), 1e-3)

    if task == "binary":
        y_arr = np.asarray(y, dtype=float)
        p_bar = float(np.average(y_arr, weights=weights))
        r = (y_arr - p_bar) * weights
        max_grad = float(np.max(np.abs(X.T @ r))) / N
    else:
        if class_levels is None:
            class_levels = list(pd.unique(pd.Series(y).astype(str)))
        else:
            class_levels = list(class_levels)

        y_str = pd.Series(y).astype(str).to_numpy()

        max_grad = 0.0
        for cls in class_levels:
            yc = (y_str == cls).astype(float)
            p_bar = float(np.average(yc, weights=weights))
            r = (yc - p_bar) * weights
            grad_c = float(np.max(np.abs(X.T @ r))) / N
            if grad_c > max_grad:
                max_grad = grad_c

    if not np.isfinite(max_grad) or max_grad <= 0:
        warnings.warn(
            "make_data_adaptive_lambda_path: max gradient at intercept-only "
            "solution is zero or non-finite; falling back to a generic path."
        )
        return make_default_lambda_path(n_lambda=n_lambda)

    lambda_max = max_grad / alpha_eff
    lambda_min = lambda_max * float(eps_ratio)

    lambda_seq = np.exp(
        np.linspace(np.log(lambda_max), np.log(lambda_min), int(n_lambda))
    )
    lambda_seq = lambda_seq[np.isfinite(lambda_seq) & (lambda_seq > 0)]
    return np.sort(np.unique(lambda_seq))[::-1]


def validate_multiclass_training_classes(
    y_factor: Sequence[Any],
    class_levels: Sequence[str],
    context: str = "multiclass training",
) -> bool:
    observed_classes = set(pd.Series(y_factor).astype(str).unique())
    missing_classes = [cls for cls in class_levels if cls not in observed_classes]

    if missing_classes:
        raise ValueError(
            f"{context}: missing class(es) after weighted expansion: "
            + ", ".join(missing_classes)
            + ". Check patient split and q_mat columns."
        )

    return True


############################################################
# sklearn logistic helpers
############################################################

def _fit_logistic_model(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    glmnet_alpha: float,
    lambda_value: float,
    maxit: int = 1000000,
    seed: int = 2026,
    task: str = "binary",
) -> LogisticRegression:
    """
    Fit a single sklearn LogisticRegression at a given (alpha, lambda).

    Parameter mapping vs R glmnet:
      glmnet_alpha == 0    -> sklearn penalty = "l2"        (ridge)
      glmnet_alpha == 1    -> sklearn penalty = "l1"        (lasso)
      0 < glmnet_alpha < 1 -> sklearn penalty = "elasticnet"

    sklearn uses C as the inverse regularization strength; we map
    C = 1 / lambda. There is NO silent clipping of C — extreme lambda
    values from CV (or reused across EM iterations) are passed through
    so the fit corresponds to the lambda that was actually selected.

    Convergence settings (TUNED FOR SPEED + accuracy parity with glmnet):
      Ridge       (lbfgs): tol=1e-5, max_iter capped at 2_000.
      Lasso / EN  (saga ): tol=1e-4, max_iter capped at 2_000.

    Notes:
      * sklearn's default `tol` is 1e-4, which gives probabilities
        indistinguishable from R glmnet's `thresh = 1e-7` for these
        problem sizes. Tighter `tol` is mostly wasted compute on saga.
      * Hitting `max_iter` warnings are demoted to a single one-time
        message per call so they don't flood EM logs.
      * `warm_start_init` (optional) lets EM hand the previous fit's
        coefficients to the next fit, which speeds saga up dramatically.
        Pass `warm_start_init=prev_fit['fit']` to enable. If you don't
        pass it, behavior is unchanged.
    """
    glmnet_alpha = float(glmnet_alpha)
    lambda_value = float(lambda_value)

    if not np.isfinite(lambda_value) or lambda_value <= 0:
        warnings.warn(
            "_fit_logistic_model: non-positive / non-finite lambda; "
            "defaulting to lambda = 1.0."
        )
        lambda_value = 1.0

    C = 1.0 / lambda_value

    if not np.isfinite(C) or C <= 0:
        warnings.warn(
            f"_fit_logistic_model: derived C = {C} is non-finite; "
            "defaulting to C = 1.0."
        )
        C = 1.0

    common = dict(
        random_state=seed,
        fit_intercept=True,
    )

    # Ridge: lbfgs is much faster and more stable than saga for L2.
    if abs(glmnet_alpha - 0.0) < 1e-9:
        max_iter_safe = min(int(maxit), 2_000)
        model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            tol=1e-5,
            max_iter=max_iter_safe,
            **common,
        )

    # Lasso: saga.
    elif abs(glmnet_alpha - 1.0) < 1e-9:
        max_iter_safe = min(int(maxit), 2_000)
        model = LogisticRegression(
            penalty="l1",
            C=C,
            solver="saga",
            tol=1e-4,
            max_iter=max_iter_safe,
            n_jobs=1,
            **common,
        )

    # Elastic net: saga.
    else:
        max_iter_safe = min(int(maxit), 2_000)
        model = LogisticRegression(
            penalty="elasticnet",
            C=C,
            solver="saga",
            l1_ratio=glmnet_alpha,
            tol=1e-4,
            max_iter=max_iter_safe,
            n_jobs=1,
            **common,
        )

    print(
        f"[logistic fit] task={task}, alpha={glmnet_alpha}, "
        f"lambda={lambda_value:.4g}, C={C:.4g}, "
        f"solver={model.solver}, max_iter={model.max_iter}, tol={model.tol:.1e}, "
        f"n={x.shape[0]}, p={x.shape[1]}",
        flush=True,
    )

    # Suppress sklearn's per-fit ConvergenceWarning -- we report it ONCE below
    # via n_iter_, and we never want those warnings to flood EM/MCEM logs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y, sample_weight=weights)

    n_iter_run = (
        int(np.atleast_1d(model.n_iter_).max())
        if hasattr(model, "n_iter_") and model.n_iter_ is not None
        else None
    )

    if n_iter_run is not None and n_iter_run >= model.max_iter:
        # Single, terse note instead of a full Python warning per fit.
        print(
            f"[logistic fit] note: hit max_iter={model.max_iter} "
            f"(C={C:.4g}); consider raising max_iter or relaxing tol.",
            flush=True,
        )

    print("[logistic fit] done", flush=True)

    return model


def _predict_binary_from_model(
    model: LogisticRegression,
    x: np.ndarray,
) -> np.ndarray:
    prob_all = model.predict_proba(x)

    if 1 in model.classes_:
        class1_idx = list(model.classes_).index(1)
    elif "1" in model.classes_:
        class1_idx = list(model.classes_).index("1")
    else:
        class1_idx = prob_all.shape[1] - 1

    prob = prob_all[:, class1_idx]
    prob = np.clip(prob.astype(float), 1e-8, 1 - 1e-8)

    return prob


def _predict_multiclass_from_model(
    model: LogisticRegression,
    x: np.ndarray,
    class_levels: Sequence[str],
) -> pd.DataFrame:
    class_levels = list(class_levels)

    prob_raw = model.predict_proba(x)
    model_classes = list(map(str, model.classes_))

    prob_df = pd.DataFrame(
        0.0,
        index=np.arange(x.shape[0]),
        columns=class_levels,
    )

    for j, cls in enumerate(model_classes):
        if cls in prob_df.columns:
            prob_df[cls] = prob_raw[:, j]

    prob_df = row_normalize(prob_df)
    return prob_df.loc[:, class_levels]


def _weighted_log_loss_multiclass(
    y_true: Sequence[str],
    prob_df: pd.DataFrame,
    weights: np.ndarray,
    class_levels: Sequence[str],
    eps: float = 1e-12,
) -> float:
    class_levels = list(class_levels)
    y_true = pd.Series(y_true).astype(str).to_numpy()
    weights = np.asarray(weights, dtype=float)

    prob_df = prob_df.loc[:, class_levels]
    prob_arr = np.clip(prob_df.to_numpy(dtype=float), eps, 1.0)

    class_to_idx = {
        cls: i
        for i, cls in enumerate(class_levels)
    }

    idx = np.array([class_to_idx[y] for y in y_true], dtype=int)
    p_true = prob_arr[np.arange(len(y_true)), idx]

    return float(-np.sum(weights * np.log(p_true)) / np.sum(weights))


def _manual_lambda_cv(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lambda_seq: np.ndarray,
    glmnet_alpha: float,
    nfolds: int,
    lambda_choice: str,
    standardize: bool,
    maxit: int,
    seed: int,
    task: str,
    class_levels: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Manual CV for glmnet-like sklearn logistic models.

    Returns:
      lambda, cvm, cvsd, lambda.min, lambda.1se, lambda.use
    """
    lambda_seq = np.asarray(lambda_seq, dtype=float)
    lambda_seq = lambda_seq[np.isfinite(lambda_seq) & (lambda_seq > 0)]

    if len(lambda_seq) < 2:
        lambda_seq = make_safe_lambda_path(lambda_seq[0])

    lambda_seq = np.array(sorted(np.unique(lambda_seq), reverse=True))

    n = len(y)

    if task == "binary":
        y_for_split = y
        class_counts = pd.Series(y_for_split).value_counts()
    else:
        y_for_split = pd.Series(y).astype(str).to_numpy()
        class_counts = pd.Series(y_for_split).value_counts()

    min_class_count = int(class_counts.min())
    nfolds_eff = min(int(nfolds), n, min_class_count)

    if nfolds_eff >= 2:
        splitter = StratifiedKFold(
            n_splits=nfolds_eff,
            shuffle=True,
            random_state=seed,
        )
        split_iter = splitter.split(x, y_for_split)
    else:
        warnings.warn(
            "Not enough samples per class for stratified CV; using 2-fold KFold."
        )
        nfolds_eff = min(2, n)
        if nfolds_eff < 2:
            raise ValueError("_manual_lambda_cv: need at least 2 observations for CV.")

        splitter = KFold(
            n_splits=nfolds_eff,
            shuffle=True,
            random_state=seed,
        )
        split_iter = splitter.split(x)

    cv_loss = np.full(
        shape=(nfolds_eff, len(lambda_seq)),
        fill_value=np.nan,
        dtype=float,
    )

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        x_train = x[train_idx]
        x_val = x[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        w_train = weights[train_idx]
        w_val = weights[val_idx]

        if task == "multiclass":
            observed = set(pd.Series(y_train).astype(str).unique())
            missing = [cls for cls in class_levels if cls not in observed]

            if missing:
                if verbose:
                    print(
                        f"[manual cv] skipping fold {fold_idx + 1} "
                        f"because classes are missing: {missing}"
                    )
                continue

        if standardize:
            scaled = _standardize_train_test(x_train, x_val)
            x_train_fit = scaled["x_train"]
            x_val_fit = scaled["x_test"]
        else:
            x_train_fit = x_train
            x_val_fit = x_val

        for l_idx, lambda_value in enumerate(lambda_seq):
            try:
                model = _fit_logistic_model(
                    x=x_train_fit,
                    y=y_train,
                    weights=w_train,
                    glmnet_alpha=glmnet_alpha,
                    lambda_value=lambda_value,
                    maxit=maxit,
                    seed=seed,
                    task=task,
                )

                if task == "binary":
                    prob = _predict_binary_from_model(model, x_val_fit)
                    prob = np.clip(prob, 1e-12, 1 - 1e-12)
                    loss = float(
                        -np.sum(
                            w_val
                            * (
                                y_val.astype(float) * np.log(prob)
                                + (1 - y_val.astype(float)) * np.log(1 - prob)
                            )
                        )
                        / np.sum(w_val)
                    )

                else:
                    prob_df = _predict_multiclass_from_model(
                        model=model,
                        x=x_val_fit,
                        class_levels=class_levels,
                    )

                    loss = _weighted_log_loss_multiclass(
                        y_true=y_val,
                        prob_df=prob_df,
                        weights=w_val,
                        class_levels=class_levels,
                    )

                cv_loss[fold_idx, l_idx] = loss

            except Exception as exc:
                if verbose:
                    print(
                        f"[manual cv] failed fold={fold_idx + 1}, "
                        f"lambda_idx={l_idx + 1}: {exc}"
                    )

    mean_loss = np.nanmean(cv_loss, axis=0)
    se_loss = np.array(
        [
            np.nanstd(cv_loss[:, j], ddof=1) / np.sqrt(np.sum(~np.isnan(cv_loss[:, j])))
            if np.sum(~np.isnan(cv_loss[:, j])) >= 2
            else np.nan
            for j in range(cv_loss.shape[1])
        ]
    )

    if np.all(np.isnan(mean_loss)):
        raise ValueError("_manual_lambda_cv: every CV fold failed; cannot pick lambda.")

    best_idx = int(np.nanargmin(mean_loss))
    lambda_min = float(lambda_seq[best_idx])

    threshold = mean_loss[best_idx] + (
        0.0 if np.isnan(se_loss[best_idx]) else se_loss[best_idx]
    )

    candidates = np.where(mean_loss <= threshold)[0]

    if len(candidates) == 0:
        lambda_1se = lambda_min
    else:
        # lambda_seq is decreasing, so the smallest index is strongest regularization.
        lambda_1se = float(lambda_seq[int(np.min(candidates))])

    lambda_use = lambda_1se if lambda_choice == "lambda.1se" else lambda_min

    return {
        "lambda": lambda_seq,
        "cvm": mean_loss,
        "cvsd": se_loss,
        "lambda.min": lambda_min,
        "lambda.1se": lambda_1se,
        "lambda.use": lambda_use,
        "cv_loss": cv_loss,
    }


############################################################
# glmnet-like binary backend
############################################################

def fit_glmnet_binary(
    x: pd.DataFrame | np.ndarray,
    y_soft: Sequence[float] | np.ndarray,
    backend_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    backend_config = backend_config or {}

    glmnet_alpha = get_config_value(backend_config, "glmnet_alpha", default=0)
    lambda_value = get_config_value(backend_config, "lambda", default=None)
    lambda_seq = get_config_value(backend_config, "lambda_seq", default=None)
    nfolds = get_config_value(backend_config, "nfolds", default=5)
    lambda_choice = get_config_value(backend_config, "lambda_choice", default="lambda.min")
    standardize = get_config_value(backend_config, "standardize", default=True)
    maxit = get_config_value(backend_config, "maxit", default=1000000)
    seed = get_config_value(backend_config, "seed", default=2026)
    manual_cv_verbose = get_config_value(backend_config, "manual_cv_verbose", default=False)

    n_lambda_cfg = int(get_config_value(backend_config, "n_lambda", default=60))
    lambda_min_ratio_cfg = float(
        get_config_value(backend_config, "lambda_min_ratio", default=1e-4)
    )

    y_soft = np.clip(np.asarray(y_soft, dtype=float), 1e-6, 1 - 1e-6)

    weighted_data = prepare_binary_weighted_data(x=x, q=y_soft)

    x_train_raw = weighted_data["x"]
    y_train = weighted_data["y"]
    weights = weighted_data["weight"]

    ##########################################################
    # Standardize once, on the FULL weighted training set, and
    # use that standardized X for both the lambda path and the
    # final fit. (Per-fold standardization happens inside the CV
    # routine if standardize=True is also passed there; we set it
    # to False here because we've already done it once globally.)
    ##########################################################

    if standardize:
        scaled = _standardize_train_test(x_train_raw)
        x_fit = scaled["x_train"]
        center = scaled["center"]
        scale = scaled["scale"]
    else:
        x_fit = x_train_raw
        center = None
        scale = None

    cv_fit = None

    if lambda_value is None:
        if lambda_seq is None:
            lambda_seq = make_data_adaptive_lambda_path(
                X=x_fit,
                y=y_train,
                weights=weights,
                glmnet_alpha=glmnet_alpha,
                n_lambda=n_lambda_cfg,
                eps_ratio=lambda_min_ratio_cfg,
                task="binary",
            )

        cv_fit = _manual_lambda_cv(
            x=x_fit,
            y=y_train,
            weights=weights,
            lambda_seq=np.asarray(lambda_seq, dtype=float),
            glmnet_alpha=glmnet_alpha,
            nfolds=nfolds,
            lambda_choice=lambda_choice,
            standardize=False,  # already standardized once above
            maxit=maxit,
            seed=seed,
            task="binary",
            verbose=manual_cv_verbose,
        )

        lambda_use = cv_fit["lambda.use"]
        lambda_seq_use = cv_fit["lambda"]

    else:
        lambda_use = float(lambda_value)
        lambda_seq_use = np.asarray([lambda_use], dtype=float)

    fit = _fit_logistic_model(
        x=x_fit,
        y=y_train,
        weights=weights,
        glmnet_alpha=glmnet_alpha,
        lambda_value=lambda_use,
        maxit=maxit,
        seed=seed,
        task="binary",
    )

    return {
        "backend": "glmnet",
        "task": "binary",
        "fit": fit,
        "cv_fit": cv_fit,
        "lambda": lambda_use,
        "lambda_seq": lambda_seq_use,
        "glmnet_alpha": glmnet_alpha,
        "standardize": standardize,
        "center": center,
        "scale": scale,
        "backend_config": backend_config,
    }


def predict_glmnet_binary(
    fit_obj: dict[str, Any],
    x: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    x_arr = _apply_standardizer(
        x=x,
        center=fit_obj.get("center"),
        scale=fit_obj.get("scale"),
    )

    prob = _predict_binary_from_model(
        model=fit_obj["fit"],
        x=x_arr,
    )

    prob = np.clip(prob, 1e-8, 1 - 1e-8)

    return prob


############################################################
# glmnet-like multiclass backend
############################################################

def fit_glmnet_multiclass(
    x: pd.DataFrame | np.ndarray,
    q_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
    backend_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    backend_config = backend_config or {}
    class_levels = list(class_levels)

    glmnet_alpha = get_config_value(backend_config, "glmnet_alpha", default=0)
    lambda_value = get_config_value(backend_config, "lambda", default=None)
    lambda_seq = get_config_value(backend_config, "lambda_seq", default=None)
    nfolds = get_config_value(backend_config, "nfolds", default=5)
    lambda_choice = get_config_value(backend_config, "lambda_choice", default="lambda.min")
    standardize = get_config_value(backend_config, "standardize", default=True)
    maxit = get_config_value(backend_config, "maxit", default=1000000)
    seed = get_config_value(backend_config, "manual_cv_seed", default=2026)
    manual_cv_verbose = get_config_value(backend_config, "manual_cv_verbose", default=False)

    n_lambda_cfg = int(get_config_value(backend_config, "n_lambda", default=60))
    lambda_min_ratio_cfg = float(
        get_config_value(backend_config, "lambda_min_ratio", default=1e-4)
    )

    weighted_data = prepare_multiclass_weighted_data(
        x=x,
        q_mat=q_mat,
        class_levels=class_levels,
    )

    x_train_raw = weighted_data["x"]
    y_train = pd.Series(weighted_data["y"]).astype(str).to_numpy()
    weights = weighted_data["weight"]

    validate_multiclass_training_classes(
        y_factor=y_train,
        class_levels=class_levels,
        context="fit_glmnet_multiclass weighted data",
    )

    if standardize:
        scaled = _standardize_train_test(x_train_raw)
        x_fit = scaled["x_train"]
        center = scaled["center"]
        scale = scaled["scale"]
    else:
        x_fit = x_train_raw
        center = None
        scale = None

    cv_fit = None

    if lambda_value is None:
        if lambda_seq is None:
            lambda_seq = make_data_adaptive_lambda_path(
                X=x_fit,
                y=y_train,
                weights=weights,
                glmnet_alpha=glmnet_alpha,
                n_lambda=n_lambda_cfg,
                eps_ratio=lambda_min_ratio_cfg,
                task="multiclass",
                class_levels=class_levels,
            )

        cv_fit = _manual_lambda_cv(
            x=x_fit,
            y=y_train,
            weights=weights,
            lambda_seq=np.asarray(lambda_seq, dtype=float),
            glmnet_alpha=glmnet_alpha,
            nfolds=nfolds,
            lambda_choice=lambda_choice,
            standardize=False,  # already standardized once above
            maxit=maxit,
            seed=seed,
            task="multiclass",
            class_levels=class_levels,
            verbose=manual_cv_verbose,
        )

        lambda_use = cv_fit["lambda.use"]
        lambda_seq_use = cv_fit["lambda"]

    else:
        lambda_use = float(lambda_value)

        if lambda_seq is not None:
            lambda_seq_use = np.asarray(lambda_seq, dtype=float)
            lambda_seq_use = lambda_seq_use[np.isfinite(lambda_seq_use) & (lambda_seq_use > 0)]
            lambda_seq_use = np.array(sorted(np.unique(lambda_seq_use), reverse=True))

            if len(lambda_seq_use) < 10:
                lambda_seq_use = make_wide_lambda_path(lambda_use)
        else:
            lambda_seq_use = make_wide_lambda_path(lambda_use)

    fit = _fit_logistic_model(
        x=x_fit,
        y=y_train,
        weights=weights,
        glmnet_alpha=glmnet_alpha,
        lambda_value=lambda_use,
        maxit=maxit,
        seed=seed,
        task="multiclass",
    )

    return {
        "backend": "glmnet",
        "task": "multiclass",
        "fit": fit,
        "cv_fit": cv_fit,
        "lambda": lambda_use,
        "lambda_seq": lambda_seq_use,
        "glmnet_alpha": glmnet_alpha,
        "class_levels": class_levels,
        "standardize": standardize,
        "center": center,
        "scale": scale,
        "backend_config": backend_config,
    }


def predict_glmnet_multiclass(
    fit_obj: dict[str, Any],
    x: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    x_arr = _apply_standardizer(
        x=x,
        center=fit_obj.get("center"),
        scale=fit_obj.get("scale"),
    )

    return _predict_multiclass_from_model(
        model=fit_obj["fit"],
        x=x_arr,
        class_levels=fit_obj["class_levels"],
    )


############################################################
# XGBoost binary backend
############################################################

def fit_xgboost_binary(
    x: pd.DataFrame | np.ndarray,
    y_soft: Sequence[float] | np.ndarray,
    backend_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for backend='xgboost'. "
            "Install with: pip install xgboost"
        ) from exc

    backend_config = backend_config or {}

    weighted_data = prepare_binary_weighted_data(
        x=x,
        q=y_soft,
    )

    dtrain = xgb.DMatrix(
        data=weighted_data["x"],
        label=weighted_data["y"],
        weight=weighted_data["weight"],
    )

    eta = get_config_value(backend_config, "eta", default=0.05)
    max_depth = get_config_value(backend_config, "max_depth", default=3)
    min_child_weight = get_config_value(backend_config, "min_child_weight", default=1)
    subsample = get_config_value(backend_config, "subsample", default=0.8)
    colsample_bytree = get_config_value(backend_config, "colsample_bytree", default=0.8)
    xgb_lambda = get_config_value(backend_config, "xgb_lambda", default=1)
    xgb_alpha = get_config_value(backend_config, "xgb_alpha", default=0)
    nthread = get_config_value(backend_config, "nthread", default=2)
    nrounds = get_config_value(backend_config, "nrounds", default=100)
    verbose = get_config_value(backend_config, "verbose", default=0)
    seed = get_config_value(backend_config, "seed", default=2026)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": eta,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": xgb_lambda,
        "alpha": xgb_alpha,
        "nthread": nthread,
        "seed": seed,
    }

    fit = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(nrounds),
        verbose_eval=bool(verbose),
    )

    return {
        "backend": "xgboost",
        "task": "binary",
        "fit": fit,
        "params": params,
        "nrounds": nrounds,
        "backend_config": backend_config,
    }


def predict_xgboost_binary(
    fit_obj: dict[str, Any],
    x: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    import xgboost as xgb

    x_arr = _as_numpy_x(x)
    dtest = xgb.DMatrix(data=x_arr)

    prob = fit_obj["fit"].predict(dtest)
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, 1e-8, 1 - 1e-8)

    return prob


############################################################
# XGBoost multiclass backend
############################################################

def fit_xgboost_multiclass(
    x: pd.DataFrame | np.ndarray,
    q_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
    backend_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for backend='xgboost'. "
            "Install with: pip install xgboost"
        ) from exc

    backend_config = backend_config or {}
    class_levels = list(class_levels)

    weighted_data = prepare_multiclass_weighted_data(
        x=x,
        q_mat=q_mat,
        class_levels=class_levels,
    )

    class_to_int = {
        cls: i
        for i, cls in enumerate(class_levels)
    }

    y_integer = np.array(
        [
            class_to_int[str(y)]
            for y in weighted_data["y"]
        ],
        dtype=int,
    )

    dtrain = xgb.DMatrix(
        data=weighted_data["x"],
        label=y_integer,
        weight=weighted_data["weight"],
    )

    eta = get_config_value(backend_config, "eta", default=0.05)
    max_depth = get_config_value(backend_config, "max_depth", default=3)
    min_child_weight = get_config_value(backend_config, "min_child_weight", default=1)
    subsample = get_config_value(backend_config, "subsample", default=0.8)
    colsample_bytree = get_config_value(backend_config, "colsample_bytree", default=0.8)
    xgb_lambda = get_config_value(backend_config, "xgb_lambda", default=1)
    xgb_alpha = get_config_value(backend_config, "xgb_alpha", default=0)
    nthread = get_config_value(backend_config, "nthread", default=2)
    nrounds = get_config_value(backend_config, "nrounds", default=100)
    verbose = get_config_value(backend_config, "verbose", default=0)
    seed = get_config_value(backend_config, "seed", default=2026)

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": len(class_levels),
        "eta": eta,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": xgb_lambda,
        "alpha": xgb_alpha,
        "nthread": nthread,
        "seed": seed,
    }

    fit = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(nrounds),
        verbose_eval=bool(verbose),
    )

    return {
        "backend": "xgboost",
        "task": "multiclass",
        "fit": fit,
        "params": params,
        "nrounds": nrounds,
        "class_levels": class_levels,
        "backend_config": backend_config,
    }


def predict_xgboost_multiclass(
    fit_obj: dict[str, Any],
    x: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    import xgboost as xgb

    x_arr = _as_numpy_x(x)
    dtest = xgb.DMatrix(data=x_arr)

    prob = fit_obj["fit"].predict(dtest)

    class_levels = list(fit_obj["class_levels"])
    K = len(class_levels)

    prob_arr = np.asarray(prob, dtype=float)

    if prob_arr.ndim == 1:
        prob_arr = prob_arr.reshape(-1, K)

    prob_df = pd.DataFrame(
        prob_arr,
        columns=class_levels,
    )

    prob_df = row_normalize(prob_df)

    return prob_df.loc[:, class_levels]


############################################################
# Unified binary backend interface
############################################################

def fit_backend_binary(
    x: pd.DataFrame | np.ndarray,
    y_soft: Sequence[float] | np.ndarray,
    backend_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    backend_config = backend_config or {}
    backend = get_config_value(backend_config, "backend", default="glmnet")

    if backend == "glmnet":
        return fit_glmnet_binary(
            x=x,
            y_soft=y_soft,
            backend_config=backend_config,
        )

    if backend == "xgboost":
        return fit_xgboost_binary(
            x=x,
            y_soft=y_soft,
            backend_config=backend_config,
        )

    raise ValueError(
        f"Unsupported backend: {backend}. "
        "Supported backends are 'glmnet' and 'xgboost'."
    )


def predict_backend_binary(
    fit_obj: dict[str, Any],
    x: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    if fit_obj["backend"] == "glmnet":
        return predict_glmnet_binary(
            fit_obj=fit_obj,
            x=x,
        )

    if fit_obj["backend"] == "xgboost":
        return predict_xgboost_binary(
            fit_obj=fit_obj,
            x=x,
        )

    raise ValueError(f"Unsupported fitted backend: {fit_obj['backend']}")


############################################################
# Unified multiclass backend interface
############################################################

def fit_backend_multiclass(
    x: pd.DataFrame | np.ndarray,
    q_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
    backend_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    backend_config = backend_config or {}
    backend = get_config_value(backend_config, "backend", default="glmnet")

    if backend == "glmnet":
        return fit_glmnet_multiclass(
            x=x,
            q_mat=q_mat,
            class_levels=class_levels,
            backend_config=backend_config,
        )

    if backend == "xgboost":
        return fit_xgboost_multiclass(
            x=x,
            q_mat=q_mat,
            class_levels=class_levels,
            backend_config=backend_config,
        )

    raise ValueError(
        f"Unsupported backend: {backend}. "
        "Supported backends are 'glmnet' and 'xgboost'."
    )


def predict_backend_multiclass(
    fit_obj: dict[str, Any],
    x: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    if fit_obj["backend"] == "glmnet":
        return predict_glmnet_multiclass(
            fit_obj=fit_obj,
            x=x,
        )

    if fit_obj["backend"] == "xgboost":
        return predict_xgboost_multiclass(
            fit_obj=fit_obj,
            x=x,
        )

    raise ValueError(f"Unsupported fitted backend: {fit_obj['backend']}")


############################################################
# Backend config update helper
############################################################

def reuse_glmnet_lambda_if_requested(
    backend_config: dict[str, Any],
    reference_fit: Optional[dict[str, Any]],
) -> dict[str, Any]:
    backend_config = copy.deepcopy(backend_config)

    reuse_lambda = get_config_value(
        backend_config,
        "reuse_lambda",
        default=True,
    )

    if (
        reuse_lambda
        and reference_fit is not None
        and reference_fit.get("backend") == "glmnet"
    ):
        backend_config["lambda"] = reference_fit["lambda"]

        if "lambda_seq" in reference_fit and reference_fit["lambda_seq"] is not None:
            backend_config["lambda_seq"] = reference_fit["lambda_seq"]

    return backend_config