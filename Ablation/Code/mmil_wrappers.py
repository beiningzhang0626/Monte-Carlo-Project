############################################################
# mmil_wrappers.py
# Binary and categorical Naive / EM-MMIL / MCEM-MMIL wrappers.
#
# Python version of 04_mmil_wrappers.R
#
# Required previous modules:
#   utils.py
#   model_backends.py
############################################################

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from utils import (
    check_required_cols,
    compute_auc_safe,
    ensure_dir,
    log_loss_binary,
    make_onehot,
    multiclass_accuracy,
    multiclass_log_loss,
    row_normalize,
)
from model_backends import (
    fit_backend_binary,
    fit_backend_multiclass,
    predict_backend_binary,
    predict_backend_multiclass,
    reuse_glmnet_lambda_if_requested,
)


############################################################
# Shared helpers
############################################################

def sanitize_class_name(x: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x))


def get_metadata_columns_for_output(df: pd.DataFrame) -> list[str]:
    metadata_cols = [
        "Cell",
        "Patient",
        "split",
        "Stage",
        "Stage_broad",
        "Stage_group",
        "z_obs",
        "Stage_cat",
        "y_cat",
        "Cell_type",
        "Cell_type.refined",
        "Cell_subtype",
        "Tissue",
        "Sample",
        "Sample_Origin",
    ]

    return [col for col in metadata_cols if col in df.columns]


def _union_preserve_order(a: Sequence[str], b: Sequence[str]) -> list[str]:
    out = []
    for col in list(a) + list(b):
        if col not in out:
            out.append(col)
    return out


def _ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

    return df.loc[:, list(cols)].copy()


def _prob_to_df(
    prob_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
) -> pd.DataFrame:
    class_levels = list(class_levels)

    if isinstance(prob_mat, pd.DataFrame):
        out = prob_mat.loc[:, class_levels].copy()
    else:
        out = pd.DataFrame(prob_mat, columns=class_levels)

    out = row_normalize(out)

    return out.loc[:, class_levels]


############################################################
# Binary helpers
############################################################

def update_binary_q(
    p_hat: Sequence[float] | np.ndarray,
    z_train: Sequence[int] | np.ndarray,
    rho: float = 0.70,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    z_train:
      0 = Early
      1 = Advanced

    rho:
      assumed baseline-like fraction among Advanced cells.

    target_mean:
      expected advanced-associated fraction among Advanced cells.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    z_train = np.asarray(z_train, dtype=int)

    early_idx = z_train == 0
    adv_idx = z_train == 1

    target_mean = 1 - rho

    q_new = np.zeros_like(p_hat, dtype=float)
    q_new[early_idx] = 0.0

    if adv_idx.sum() == 0:
        return q_new

    p_adv = p_hat[adv_idx]
    current_mean = np.nanmean(p_adv)

    if not np.isfinite(current_mean) or current_mean <= eps:
        q_adv = np.full(adv_idx.sum(), target_mean, dtype=float)
    else:
        q_adv = p_adv * target_mean / current_mean

    q_adv = np.clip(q_adv, eps, 1 - eps)
    q_new[adv_idx] = q_adv

    return q_new


def monte_carlo_average_binary(
    q: Sequence[float] | np.ndarray,
    z_train: Sequence[int] | np.ndarray,
    mcem_samples: int = 30,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    z_train = np.asarray(z_train, dtype=int)

    if rng is None:
        rng = np.random.default_rng()

    n = len(q)
    sampled_sum = np.zeros(n, dtype=float)

    early_idx = z_train == 0
    adv_idx = z_train == 1

    for _ in range(int(mcem_samples)):
        sampled_y = np.zeros(n, dtype=float)
        sampled_y[early_idx] = 0.0

        if adv_idx.sum() > 0:
            sampled_y[adv_idx] = rng.binomial(
                n=1,
                p=q[adv_idx],
                size=adv_idx.sum(),
            )

        sampled_sum += sampled_y

    q_mc_avg = sampled_sum / int(mcem_samples)

    return q_mc_avg


############################################################
# Binary naive baseline
############################################################

def fit_binary_naive(
    x_train: pd.DataFrame | np.ndarray,
    x_test: pd.DataFrame | np.ndarray,
    z_train: Sequence[int] | np.ndarray,
    backend_config: dict[str, Any],
) -> dict[str, Any]:
    print("\n==============================")
    print("Fitting binary naive inherited-label baseline")
    print("==============================")

    z_train = np.asarray(z_train, dtype=float)

    naive_fit = fit_backend_binary(
        x=x_train,
        y_soft=z_train,
        backend_config=backend_config,
    )

    naive_train_prob = predict_backend_binary(
        fit_obj=naive_fit,
        x=x_train,
    )

    naive_test_prob = predict_backend_binary(
        fit_obj=naive_fit,
        x=x_test,
    )

    print("\nBinary naive train log loss:")
    print(log_loss_binary(z_train, naive_train_prob))

    return {
        "fit": naive_fit,
        "train_prob": naive_train_prob,
        "test_prob": naive_test_prob,
    }


############################################################
# Binary deterministic EM-MMIL
############################################################

def fit_binary_em(
    x_train: pd.DataFrame | np.ndarray,
    x_test: pd.DataFrame | np.ndarray,
    z_train: Sequence[int] | np.ndarray,
    backend_config: dict[str, Any],
    naive_fit: dict[str, Any] | None = None,
    rho: float = 0.70,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> dict[str, Any]:
    print("\n==============================")
    print("Fitting binary deterministic EM-MMIL")
    print("==============================")

    z_train = np.asarray(z_train, dtype=int)

    q = np.where(z_train == 1, 1 - rho, 0.0).astype(float)

    em_log_rows = []

    em_backend_config = dict(backend_config)

    if naive_fit is not None:
        em_backend_config = reuse_glmnet_lambda_if_requested(
            backend_config=em_backend_config,
            reference_fit=naive_fit,
        )

    for iteration in range(1, int(max_iter) + 1):
        old_q = q.copy()

        em_fit_iter = fit_backend_binary(
            x=x_train,
            y_soft=q,
            backend_config=em_backend_config,
        )

        p_hat = predict_backend_binary(
            fit_obj=em_fit_iter,
            x=x_train,
        )

        q = update_binary_q(
            p_hat=p_hat,
            z_train=z_train,
            rho=rho,
        )

        delta = float(np.mean(np.abs(q - old_q)))

        train_logloss_inherited = log_loss_binary(
            y=z_train,
            p=p_hat,
        )

        mean_q_advanced = (
            float(np.mean(q[z_train == 1]))
            if np.sum(z_train == 1) > 0
            else np.nan
        )

        em_log_rows.append(
            {
                "iter": iteration,
                "delta": delta,
                "mean_q_advanced": mean_q_advanced,
                "train_logloss_inherited": train_logloss_inherited,
            }
        )

        print(
            "Binary EM iter =", iteration,
            "| delta =", f"{delta:.4g}",
            "| mean q advanced =", f"{mean_q_advanced:.4g}",
            "| train logloss inherited =", f"{train_logloss_inherited:.4g}",
        )

        if delta < tol:
            print("Binary EM converged.")
            break

    em_fit = fit_backend_binary(
        x=x_train,
        y_soft=q,
        backend_config=em_backend_config,
    )

    em_train_prob = predict_backend_binary(
        fit_obj=em_fit,
        x=x_train,
    )

    em_test_prob = predict_backend_binary(
        fit_obj=em_fit,
        x=x_test,
    )

    return {
        "fit": em_fit,
        "train_prob": em_train_prob,
        "test_prob": em_test_prob,
        "soft_label": q,
        "log": pd.DataFrame(em_log_rows),
    }


############################################################
# Binary MCEM-MMIL
############################################################

def fit_binary_mcem(
    x_train: pd.DataFrame | np.ndarray,
    x_test: pd.DataFrame | np.ndarray,
    z_train: Sequence[int] | np.ndarray,
    backend_config: dict[str, Any],
    naive_fit: dict[str, Any] | None = None,
    rho: float = 0.70,
    max_iter: int = 30,
    tol: float = 1e-4,
    mcem_samples: int = 30,
    seed: int = 2026,
) -> dict[str, Any]:
    print("\n==============================")
    print("Fitting binary MCEM-MMIL")
    print("==============================")

    rng = np.random.default_rng(seed)
    z_train = np.asarray(z_train, dtype=int)

    q = np.where(z_train == 1, 1 - rho, 0.0).astype(float)

    mcem_log_rows = []

    mcem_backend_config = dict(backend_config)

    if naive_fit is not None:
        mcem_backend_config = reuse_glmnet_lambda_if_requested(
            backend_config=mcem_backend_config,
            reference_fit=naive_fit,
        )

    for iteration in range(1, int(max_iter) + 1):
        old_q = q.copy()

        q_mc_avg = monte_carlo_average_binary(
            q=q,
            z_train=z_train,
            mcem_samples=mcem_samples,
            rng=rng,
        )

        mcem_fit_iter = fit_backend_binary(
            x=x_train,
            y_soft=q_mc_avg,
            backend_config=mcem_backend_config,
        )

        p_hat = predict_backend_binary(
            fit_obj=mcem_fit_iter,
            x=x_train,
        )

        q = update_binary_q(
            p_hat=p_hat,
            z_train=z_train,
            rho=rho,
        )

        delta = float(np.mean(np.abs(q - old_q)))

        train_logloss_inherited = log_loss_binary(
            y=z_train,
            p=p_hat,
        )

        mean_q_advanced = (
            float(np.mean(q[z_train == 1]))
            if np.sum(z_train == 1) > 0
            else np.nan
        )

        mean_q_mc_advanced = (
            float(np.mean(q_mc_avg[z_train == 1]))
            if np.sum(z_train == 1) > 0
            else np.nan
        )

        mcem_log_rows.append(
            {
                "iter": iteration,
                "delta": delta,
                "mean_q_advanced": mean_q_advanced,
                "mean_q_mc_advanced": mean_q_mc_advanced,
                "train_logloss_inherited": train_logloss_inherited,
            }
        )

        print(
            "Binary MCEM iter =", iteration,
            "| delta =", f"{delta:.4g}",
            "| mean q advanced =", f"{mean_q_advanced:.4g}",
            "| mean MC q advanced =", f"{mean_q_mc_advanced:.4g}",
            "| train logloss inherited =", f"{train_logloss_inherited:.4g}",
        )

        if delta < tol:
            print("Binary MCEM converged.")
            break

    q_mc_final = monte_carlo_average_binary(
        q=q,
        z_train=z_train,
        mcem_samples=mcem_samples,
        rng=rng,
    )

    mcem_fit = fit_backend_binary(
        x=x_train,
        y_soft=q_mc_final,
        backend_config=mcem_backend_config,
    )

    mcem_train_prob = predict_backend_binary(
        fit_obj=mcem_fit,
        x=x_train,
    )

    mcem_test_prob = predict_backend_binary(
        fit_obj=mcem_fit,
        x=x_test,
    )

    return {
        "fit": mcem_fit,
        "train_prob": mcem_train_prob,
        "test_prob": mcem_test_prob,
        "soft_label": q,
        "soft_label_mc_final": q_mc_final,
        "log": pd.DataFrame(mcem_log_rows),
    }


############################################################
# Binary prediction output
############################################################

def build_binary_prediction_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    naive_result: dict[str, Any],
    em_result: dict[str, Any],
    mcem_result: dict[str, Any],
) -> pd.DataFrame:
    keep_cols = _union_preserve_order(
        get_metadata_columns_for_output(train_df),
        get_metadata_columns_for_output(test_df),
    )

    train_out = _ensure_columns(train_df, keep_cols)
    test_out = _ensure_columns(test_df, keep_cols)

    train_out["naive_prob"] = naive_result["train_prob"]
    train_out["em_mmil_prob"] = em_result["train_prob"]
    train_out["mcem_mmil_prob"] = mcem_result["train_prob"]

    train_out["em_soft_label"] = em_result["soft_label"]
    train_out["mcem_soft_label"] = mcem_result["soft_label"]

    test_out["naive_prob"] = naive_result["test_prob"]
    test_out["em_mmil_prob"] = em_result["test_prob"]
    test_out["mcem_mmil_prob"] = mcem_result["test_prob"]

    test_out["em_soft_label"] = np.nan
    test_out["mcem_soft_label"] = np.nan

    pred_df = pd.concat(
        [train_out, test_out],
        axis=0,
        ignore_index=True,
    )

    if "Cell_type" not in pred_df.columns:
        pred_df["Cell_type"] = "Unknown"

    return pred_df


############################################################
# Main binary MMIL runner
############################################################

def run_binary_mmil(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    backend_config: dict[str, Any],
    rho: float = 0.70,
    max_iter: int = 30,
    tol: float = 1e-4,
    mcem_samples: int = 30,
    seed: int = 2026,
) -> dict[str, Any]:
    feature_cols = list(feature_cols)

    check_required_cols(
        train_df,
        ["Stage_group", "z_obs"] + feature_cols,
        object_name="train_df",
    )

    check_required_cols(
        test_df,
        ["Stage_group", "z_obs"] + feature_cols,
        object_name="test_df",
    )

    x_train = train_df.loc[:, feature_cols].to_numpy(dtype=float)
    x_test = test_df.loc[:, feature_cols].to_numpy(dtype=float)

    z_train = train_df["z_obs"].to_numpy(dtype=int)
    z_test = test_df["z_obs"].to_numpy(dtype=int)

    print("\nBinary MMIL run summary:")
    print("Training cells:", x_train.shape[0])
    print("Test cells:", x_test.shape[0])
    print("Features:", len(feature_cols))
    print("Backend:", backend_config.get("backend"))

    naive_result = fit_binary_naive(
        x_train=x_train,
        x_test=x_test,
        z_train=z_train,
        backend_config=backend_config,
    )

    print("\nBinary naive test log loss using inherited labels:")
    print(log_loss_binary(z_test, naive_result["test_prob"]))

    print("\nBinary naive test AUC using inherited labels:")
    print(compute_auc_safe(z_test, naive_result["test_prob"]))

    em_result = fit_binary_em(
        x_train=x_train,
        x_test=x_test,
        z_train=z_train,
        backend_config=backend_config,
        naive_fit=naive_result["fit"],
        rho=rho,
        max_iter=max_iter,
        tol=tol,
    )

    print("\nBinary EM test log loss using inherited labels:")
    print(log_loss_binary(z_test, em_result["test_prob"]))

    print("\nBinary EM test AUC using inherited labels:")
    print(compute_auc_safe(z_test, em_result["test_prob"]))

    mcem_result = fit_binary_mcem(
        x_train=x_train,
        x_test=x_test,
        z_train=z_train,
        backend_config=backend_config,
        naive_fit=naive_result["fit"],
        rho=rho,
        max_iter=max_iter,
        tol=tol,
        mcem_samples=mcem_samples,
        seed=seed,
    )

    print("\nBinary MCEM test log loss using inherited labels:")
    print(log_loss_binary(z_test, mcem_result["test_prob"]))

    print("\nBinary MCEM test AUC using inherited labels:")
    print(compute_auc_safe(z_test, mcem_result["test_prob"]))

    pred_df = build_binary_prediction_df(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        naive_result=naive_result,
        em_result=em_result,
        mcem_result=mcem_result,
    )

    return {
        "task_type": "binary",
        "predictions": pred_df,
        "naive": naive_result,
        "em": em_result,
        "mcem": mcem_result,
    }


############################################################
# Categorical helpers
############################################################

def build_label_prior_matrix(
    y_class: Sequence[Any],
    class_levels: Sequence[str],
    label_strength: float = 0.70,
) -> pd.DataFrame:
    class_levels = list(class_levels)
    y = pd.Series(y_class).astype("string")

    n = len(y)
    K = len(class_levels)

    if K < 2:
        raise ValueError("Categorical MMIL requires at least two classes.")

    off_diag = (1 - label_strength) / (K - 1)

    prior = pd.DataFrame(
        off_diag,
        index=np.arange(n),
        columns=class_levels,
        dtype=float,
    )

    for cls in class_levels:
        prior.loc[y == cls, cls] = label_strength

    prior = row_normalize(prior)

    return prior


def sample_categorical_onehot(
    q_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    class_levels = list(class_levels)

    if rng is None:
        rng = np.random.default_rng()

    q_df = _prob_to_df(q_mat, class_levels)

    n = q_df.shape[0]
    K = len(class_levels)

    sampled = np.zeros((n, K), dtype=float)

    probs = q_df.to_numpy(dtype=float)

    for i in range(n):
        sampled_class = rng.choice(K, size=1, p=probs[i, :])[0]
        sampled[i, sampled_class] = 1.0

    return pd.DataFrame(
        sampled,
        columns=class_levels,
        index=q_df.index,
    )


def monte_carlo_average_multiclass(
    q_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
    mcem_samples: int = 30,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    class_levels = list(class_levels)

    if rng is None:
        rng = np.random.default_rng()

    q_df = _prob_to_df(q_mat, class_levels)

    sampled_sum = pd.DataFrame(
        0.0,
        index=q_df.index,
        columns=class_levels,
    )

    for _ in range(int(mcem_samples)):
        sampled_sum = sampled_sum + sample_categorical_onehot(
            q_mat=q_df,
            class_levels=class_levels,
            rng=rng,
        )

    q_mc_avg = sampled_sum / int(mcem_samples)
    q_mc_avg = row_normalize(q_mc_avg)

    return q_mc_avg.loc[:, class_levels]


def update_categorical_q(
    p_hat: pd.DataFrame | np.ndarray,
    label_prior: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    if isinstance(label_prior, pd.DataFrame):
        class_levels = list(label_prior.columns)
        prior_df = label_prior.copy()
    else:
        raise ValueError("label_prior should be a DataFrame with class columns.")

    p_df = _prob_to_df(p_hat, class_levels)

    if p_df.shape != prior_df.shape:
        raise ValueError("p_hat and label_prior must have the same dimensions.")

    q_new = p_df.to_numpy(dtype=float) * prior_df.to_numpy(dtype=float)

    q_new = pd.DataFrame(
        q_new,
        index=prior_df.index,
        columns=class_levels,
    )

    q_new = row_normalize(q_new)

    return q_new.loc[:, class_levels]


############################################################
# Categorical naive baseline
############################################################

def fit_categorical_naive(
    x_train: pd.DataFrame | np.ndarray,
    x_test: pd.DataFrame | np.ndarray,
    y_train_class: Sequence[Any],
    class_levels: Sequence[str],
    backend_config: dict[str, Any],
) -> dict[str, Any]:
    class_levels = list(class_levels)

    print("\n==============================")
    print("Fitting categorical naive inherited-label baseline")
    print("==============================")

    q_naive = make_onehot(
        y=y_train_class,
        class_levels=class_levels,
    )

    naive_fit = fit_backend_multiclass(
        x=x_train,
        q_mat=q_naive,
        class_levels=class_levels,
        backend_config=backend_config,
    )

    naive_train_prob = predict_backend_multiclass(
        fit_obj=naive_fit,
        x=x_train,
    )

    naive_test_prob = predict_backend_multiclass(
        fit_obj=naive_fit,
        x=x_test,
    )

    print("\nCategorical naive train log loss:")
    print(
        multiclass_log_loss(
            y_true=y_train_class,
            prob_mat=naive_train_prob,
            class_levels=class_levels,
        )
    )

    return {
        "fit": naive_fit,
        "train_prob": _prob_to_df(naive_train_prob, class_levels),
        "test_prob": _prob_to_df(naive_test_prob, class_levels),
        "q_naive": q_naive,
    }


############################################################
# Categorical deterministic EM-MMIL
############################################################

def fit_categorical_em(
    x_train: pd.DataFrame | np.ndarray,
    x_test: pd.DataFrame | np.ndarray,
    y_train_class: Sequence[Any],
    class_levels: Sequence[str],
    backend_config: dict[str, Any],
    naive_fit: dict[str, Any] | None = None,
    label_strength: float = 0.70,
    max_iter: int = 40,
    tol: float = 1e-4,
) -> dict[str, Any]:
    class_levels = list(class_levels)

    print("\n==============================")
    print("Fitting categorical deterministic EM-MMIL")
    print("==============================")

    label_prior = build_label_prior_matrix(
        y_class=y_train_class,
        class_levels=class_levels,
        label_strength=label_strength,
    )

    q_mat = label_prior.copy()

    em_log_rows = []

    em_backend_config = dict(backend_config)

    if naive_fit is not None:
        em_backend_config = reuse_glmnet_lambda_if_requested(
            backend_config=em_backend_config,
            reference_fit=naive_fit,
        )

    for iteration in range(1, int(max_iter) + 1):
        old_q = q_mat.copy()

        em_fit_iter = fit_backend_multiclass(
            x=x_train,
            q_mat=q_mat,
            class_levels=class_levels,
            backend_config=em_backend_config,
        )

        p_hat = predict_backend_multiclass(
            fit_obj=em_fit_iter,
            x=x_train,
        )

        q_mat = update_categorical_q(
            p_hat=p_hat,
            label_prior=label_prior,
        )

        delta = float(
            np.mean(
                np.abs(
                    q_mat.to_numpy(dtype=float)
                    - old_q.to_numpy(dtype=float)
                )
            )
        )

        q_arr = q_mat.to_numpy(dtype=float)
        entropy = -np.sum(q_arr * np.log(np.maximum(q_arr, 1e-12)), axis=1)
        mean_entropy = float(np.mean(entropy))

        train_logloss_inherited = multiclass_log_loss(
            y_true=y_train_class,
            prob_mat=p_hat,
            class_levels=class_levels,
        )

        train_accuracy_inherited = multiclass_accuracy(
            y_true=y_train_class,
            prob_mat=p_hat,
            class_levels=class_levels,
        )

        em_log_rows.append(
            {
                "iter": iteration,
                "delta": delta,
                "mean_entropy": mean_entropy,
                "train_logloss_inherited": train_logloss_inherited,
                "train_accuracy_inherited": train_accuracy_inherited,
            }
        )

        print(
            "Categorical EM iter =", iteration,
            "| delta =", f"{delta:.4g}",
            "| mean entropy =", f"{mean_entropy:.4g}",
            "| train logloss inherited =", f"{train_logloss_inherited:.4g}",
            "| train accuracy inherited =", f"{train_accuracy_inherited:.4g}",
        )

        if delta < tol:
            print("Categorical EM converged.")
            break

    em_fit = fit_backend_multiclass(
        x=x_train,
        q_mat=q_mat,
        class_levels=class_levels,
        backend_config=em_backend_config,
    )

    em_train_prob = predict_backend_multiclass(
        fit_obj=em_fit,
        x=x_train,
    )

    em_test_prob = predict_backend_multiclass(
        fit_obj=em_fit,
        x=x_test,
    )

    return {
        "fit": em_fit,
        "train_prob": _prob_to_df(em_train_prob, class_levels),
        "test_prob": _prob_to_df(em_test_prob, class_levels),
        "soft_label": q_mat,
        "label_prior": label_prior,
        "log": pd.DataFrame(em_log_rows),
    }


############################################################
# Categorical MCEM-MMIL
############################################################

def fit_categorical_mcem(
    x_train: pd.DataFrame | np.ndarray,
    x_test: pd.DataFrame | np.ndarray,
    y_train_class: Sequence[Any],
    class_levels: Sequence[str],
    backend_config: dict[str, Any],
    naive_fit: dict[str, Any] | None = None,
    label_strength: float = 0.70,
    max_iter: int = 40,
    tol: float = 1e-4,
    mcem_samples: int = 30,
    seed: int = 2026,
) -> dict[str, Any]:
    class_levels = list(class_levels)

    print("\n==============================")
    print("Fitting categorical MCEM-MMIL")
    print("==============================")

    rng = np.random.default_rng(seed)

    label_prior = build_label_prior_matrix(
        y_class=y_train_class,
        class_levels=class_levels,
        label_strength=label_strength,
    )

    q_mat = label_prior.copy()

    mcem_log_rows = []

    mcem_backend_config = dict(backend_config)

    if naive_fit is not None:
        mcem_backend_config = reuse_glmnet_lambda_if_requested(
            backend_config=mcem_backend_config,
            reference_fit=naive_fit,
        )

    for iteration in range(1, int(max_iter) + 1):
        old_q = q_mat.copy()

        q_mc_avg = monte_carlo_average_multiclass(
            q_mat=q_mat,
            class_levels=class_levels,
            mcem_samples=mcem_samples,
            rng=rng,
        )

        mcem_fit_iter = fit_backend_multiclass(
            x=x_train,
            q_mat=q_mc_avg,
            class_levels=class_levels,
            backend_config=mcem_backend_config,
        )

        p_hat = predict_backend_multiclass(
            fit_obj=mcem_fit_iter,
            x=x_train,
        )

        q_mat = update_categorical_q(
            p_hat=p_hat,
            label_prior=label_prior,
        )

        delta = float(
            np.mean(
                np.abs(
                    q_mat.to_numpy(dtype=float)
                    - old_q.to_numpy(dtype=float)
                )
            )
        )

        q_arr = q_mat.to_numpy(dtype=float)
        q_mc_arr = q_mc_avg.to_numpy(dtype=float)

        entropy = -np.sum(q_arr * np.log(np.maximum(q_arr, 1e-12)), axis=1)
        mc_entropy = -np.sum(
            q_mc_arr * np.log(np.maximum(q_mc_arr, 1e-12)),
            axis=1,
        )

        mean_entropy = float(np.mean(entropy))
        mean_mc_entropy = float(np.mean(mc_entropy))

        train_logloss_inherited = multiclass_log_loss(
            y_true=y_train_class,
            prob_mat=p_hat,
            class_levels=class_levels,
        )

        train_accuracy_inherited = multiclass_accuracy(
            y_true=y_train_class,
            prob_mat=p_hat,
            class_levels=class_levels,
        )

        mcem_log_rows.append(
            {
                "iter": iteration,
                "delta": delta,
                "mean_entropy": mean_entropy,
                "mean_mc_entropy": mean_mc_entropy,
                "train_logloss_inherited": train_logloss_inherited,
                "train_accuracy_inherited": train_accuracy_inherited,
            }
        )

        print(
            "Categorical MCEM iter =", iteration,
            "| delta =", f"{delta:.4g}",
            "| mean entropy =", f"{mean_entropy:.4g}",
            "| mean MC entropy =", f"{mean_mc_entropy:.4g}",
            "| train logloss inherited =", f"{train_logloss_inherited:.4g}",
            "| train accuracy inherited =", f"{train_accuracy_inherited:.4g}",
        )

        if delta < tol:
            print("Categorical MCEM converged.")
            break

    q_mc_final = monte_carlo_average_multiclass(
        q_mat=q_mat,
        class_levels=class_levels,
        mcem_samples=mcem_samples,
        rng=rng,
    )

    mcem_fit = fit_backend_multiclass(
        x=x_train,
        q_mat=q_mc_final,
        class_levels=class_levels,
        backend_config=mcem_backend_config,
    )

    mcem_train_prob = predict_backend_multiclass(
        fit_obj=mcem_fit,
        x=x_train,
    )

    mcem_test_prob = predict_backend_multiclass(
        fit_obj=mcem_fit,
        x=x_test,
    )

    return {
        "fit": mcem_fit,
        "train_prob": _prob_to_df(mcem_train_prob, class_levels),
        "test_prob": _prob_to_df(mcem_test_prob, class_levels),
        "soft_label": q_mat,
        "soft_label_mc_final": q_mc_final,
        "label_prior": label_prior,
        "log": pd.DataFrame(mcem_log_rows),
    }


############################################################
# Categorical prediction output
############################################################

def add_prob_columns(
    df: pd.DataFrame,
    prob_mat: pd.DataFrame | np.ndarray,
    prefix: str,
    class_levels: Sequence[str],
) -> pd.DataFrame:
    df = df.copy()
    prob_df = _prob_to_df(prob_mat, class_levels)

    for cls in class_levels:
        col_name = f"{prefix}_{sanitize_class_name(cls)}"
        df[col_name] = prob_df[cls].to_numpy(dtype=float)

    return df


def add_soft_label_columns(
    df: pd.DataFrame,
    q_mat: pd.DataFrame | np.ndarray,
    prefix: str,
    class_levels: Sequence[str],
) -> pd.DataFrame:
    df = df.copy()
    q_df = _prob_to_df(q_mat, class_levels)

    for cls in class_levels:
        col_name = f"{prefix}_{sanitize_class_name(cls)}"
        df[col_name] = q_df[cls].to_numpy(dtype=float)

    return df


def build_categorical_prediction_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    class_levels: Sequence[str],
    naive_result: dict[str, Any],
    em_result: dict[str, Any],
    mcem_result: dict[str, Any],
) -> pd.DataFrame:
    class_levels = list(class_levels)

    keep_cols = _union_preserve_order(
        get_metadata_columns_for_output(train_df),
        get_metadata_columns_for_output(test_df),
    )

    train_out = _ensure_columns(train_df, keep_cols)
    test_out = _ensure_columns(test_df, keep_cols)

    train_out = add_prob_columns(
        df=train_out,
        prob_mat=naive_result["train_prob"],
        prefix="naive_prob",
        class_levels=class_levels,
    )

    train_out = add_prob_columns(
        df=train_out,
        prob_mat=em_result["train_prob"],
        prefix="cat_em_prob",
        class_levels=class_levels,
    )

    train_out = add_prob_columns(
        df=train_out,
        prob_mat=mcem_result["train_prob"],
        prefix="cat_mcem_prob",
        class_levels=class_levels,
    )

    train_out = add_soft_label_columns(
        df=train_out,
        q_mat=em_result["soft_label"],
        prefix="cat_em_soft_label",
        class_levels=class_levels,
    )

    train_out = add_soft_label_columns(
        df=train_out,
        q_mat=mcem_result["soft_label"],
        prefix="cat_mcem_soft_label",
        class_levels=class_levels,
    )

    test_out = add_prob_columns(
        df=test_out,
        prob_mat=naive_result["test_prob"],
        prefix="naive_prob",
        class_levels=class_levels,
    )

    test_out = add_prob_columns(
        df=test_out,
        prob_mat=em_result["test_prob"],
        prefix="cat_em_prob",
        class_levels=class_levels,
    )

    test_out = add_prob_columns(
        df=test_out,
        prob_mat=mcem_result["test_prob"],
        prefix="cat_mcem_prob",
        class_levels=class_levels,
    )

    for cls in class_levels:
        test_out[f"cat_em_soft_label_{sanitize_class_name(cls)}"] = np.nan
        test_out[f"cat_mcem_soft_label_{sanitize_class_name(cls)}"] = np.nan

    pred_df = pd.concat(
        [train_out, test_out],
        axis=0,
        ignore_index=True,
    )

    if "Cell_type" not in pred_df.columns:
        pred_df["Cell_type"] = "Unknown"

    return pred_df


############################################################
# Main categorical MMIL runner
############################################################

def run_categorical_mmil(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    backend_config: dict[str, Any],
    class_levels: Sequence[str],
    label_strength: float = 0.70,
    max_iter: int = 40,
    tol: float = 1e-4,
    mcem_samples: int = 30,
    seed: int = 2026,
) -> dict[str, Any]:
    feature_cols = list(feature_cols)
    class_levels = list(class_levels)

    check_required_cols(
        train_df,
        ["Stage_cat", "y_cat"] + feature_cols,
        object_name="train_df",
    )

    check_required_cols(
        test_df,
        ["Stage_cat", "y_cat"] + feature_cols,
        object_name="test_df",
    )

    x_train = train_df.loc[:, feature_cols].to_numpy(dtype=float)
    x_test = test_df.loc[:, feature_cols].to_numpy(dtype=float)

    y_train_class = train_df["Stage_cat"].astype(str).to_numpy()
    y_test_class = test_df["Stage_cat"].astype(str).to_numpy()

    print("\nCategorical MMIL run summary:")
    print("Training cells:", x_train.shape[0])
    print("Test cells:", x_test.shape[0])
    print("Features:", len(feature_cols))
    print("Backend:", backend_config.get("backend"))
    print("Class levels:", ", ".join(class_levels))
    print("Label strength:", label_strength)

    naive_result = fit_categorical_naive(
        x_train=x_train,
        x_test=x_test,
        y_train_class=y_train_class,
        class_levels=class_levels,
        backend_config=backend_config,
    )

    print("\nCategorical naive test log loss using inherited labels:")
    print(
        multiclass_log_loss(
            y_true=y_test_class,
            prob_mat=naive_result["test_prob"],
            class_levels=class_levels,
        )
    )

    print("\nCategorical naive test accuracy using inherited labels:")
    print(
        multiclass_accuracy(
            y_true=y_test_class,
            prob_mat=naive_result["test_prob"],
            class_levels=class_levels,
        )
    )

    em_result = fit_categorical_em(
        x_train=x_train,
        x_test=x_test,
        y_train_class=y_train_class,
        class_levels=class_levels,
        backend_config=backend_config,
        naive_fit=naive_result["fit"],
        label_strength=label_strength,
        max_iter=max_iter,
        tol=tol,
    )

    print("\nCategorical EM test log loss using inherited labels:")
    print(
        multiclass_log_loss(
            y_true=y_test_class,
            prob_mat=em_result["test_prob"],
            class_levels=class_levels,
        )
    )

    print("\nCategorical EM test accuracy using inherited labels:")
    print(
        multiclass_accuracy(
            y_true=y_test_class,
            prob_mat=em_result["test_prob"],
            class_levels=class_levels,
        )
    )

    mcem_result = fit_categorical_mcem(
        x_train=x_train,
        x_test=x_test,
        y_train_class=y_train_class,
        class_levels=class_levels,
        backend_config=backend_config,
        naive_fit=naive_result["fit"],
        label_strength=label_strength,
        max_iter=max_iter,
        tol=tol,
        mcem_samples=mcem_samples,
        seed=seed,
    )

    print("\nCategorical MCEM test log loss using inherited labels:")
    print(
        multiclass_log_loss(
            y_true=y_test_class,
            prob_mat=mcem_result["test_prob"],
            class_levels=class_levels,
        )
    )

    print("\nCategorical MCEM test accuracy using inherited labels:")
    print(
        multiclass_accuracy(
            y_true=y_test_class,
            prob_mat=mcem_result["test_prob"],
            class_levels=class_levels,
        )
    )

    pred_df = build_categorical_prediction_df(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        class_levels=class_levels,
        naive_result=naive_result,
        em_result=em_result,
        mcem_result=mcem_result,
    )

    return {
        "task_type": "categorical",
        "class_levels": class_levels,
        "predictions": pred_df,
        "naive": naive_result,
        "em": em_result,
        "mcem": mcem_result,
    }


############################################################
# Save outputs
############################################################

def save_binary_mmil_outputs(
    mmil_result: dict[str, Any],
    out_dir: str | Path,
    pred_filename: str = "cell_predictions.csv",
) -> dict[str, str]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    pred_file = out_dir / pred_filename
    em_log_file = out_dir / "em_mmil_log.csv"
    mcem_log_file = out_dir / "mcem_mmil_log.csv"

    mmil_result["predictions"].to_csv(pred_file, index=False)
    mmil_result["em"]["log"].to_csv(em_log_file, index=False)
    mmil_result["mcem"]["log"].to_csv(mcem_log_file, index=False)

    print("\nSaved binary MMIL outputs:")
    print(pred_file)
    print(em_log_file)
    print(mcem_log_file)

    return {
        "pred_file": str(pred_file),
        "em_log_file": str(em_log_file),
        "mcem_log_file": str(mcem_log_file),
    }


def save_categorical_mmil_outputs(
    mmil_result: dict[str, Any],
    out_dir: str | Path,
    pred_filename: str = "cell_predictions.csv",
) -> dict[str, str]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    pred_file = out_dir / pred_filename
    em_log_file = out_dir / "cat_em_mmil_log.csv"
    mcem_log_file = out_dir / "cat_mcem_mmil_log.csv"

    mmil_result["predictions"].to_csv(pred_file, index=False)
    mmil_result["em"]["log"].to_csv(em_log_file, index=False)
    mmil_result["mcem"]["log"].to_csv(mcem_log_file, index=False)

    print("\nSaved categorical MMIL outputs:")
    print(pred_file)
    print(em_log_file)
    print(mcem_log_file)

    return {
        "pred_file": str(pred_file),
        "em_log_file": str(em_log_file),
        "mcem_log_file": str(mcem_log_file),
    }


def save_mmil_outputs(
    mmil_result: dict[str, Any],
    out_dir: str | Path,
    pred_filename: str = "cell_predictions.csv",
) -> dict[str, str]:
    if mmil_result["task_type"] == "binary":
        return save_binary_mmil_outputs(
            mmil_result=mmil_result,
            out_dir=out_dir,
            pred_filename=pred_filename,
        )

    if mmil_result["task_type"] == "categorical":
        return save_categorical_mmil_outputs(
            mmil_result=mmil_result,
            out_dir=out_dir,
            pred_filename=pred_filename,
        )

    raise ValueError("Unsupported mmil_result['task_type'].")