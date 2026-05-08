############################################################
# feature_extractors.py
# Feature extraction module for PCA, ICA, and NMF.
#
# Python version of 02_feature_extractors.R
#
# Important:
#   Feature extraction is fit only on train cells.
#   Test cells are transformed using the train-fitted feature model.
#
# Supported feature methods:
#   1. PCA
#   2. ICA
#   3. NMF
#
# CHANGES vs previous version:
#   - fit_transform_NMF() now maps R's `nmf_method` argument to the
#     correct sklearn beta_loss / solver pair:
#         "brunet" (R default) -> KL divergence + multiplicative updates
#         "lee"   / "frobenius" -> Frobenius      + multiplicative updates
#     Previously every NMF run silently used Frobenius loss regardless
#     of the requested method, which is the WRONG factor model for
#     `brunet`. This single fix changes every NMF row in the experiment
#     output.
#   - Init switched from "random" to "nndsvda" (deterministic given a
#     seed; safe for KL-divergence + multiplicative updates because it
#     replaces the hard zeros that "nndsvd" would otherwise produce).
############################################################

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import FastICA, NMF, PCA

from utils import (
    col_normalize_counts,
    safe_log1p,
)


############################################################
# Basic preprocessing
############################################################

def filter_genes_train_only(
    train_expr: pd.DataFrame,
    test_expr: pd.DataFrame,
    min_cell_fraction: float = 0.01,
) -> dict[str, Any]:
    """
    Filter genes using train cells only.

    Input:
      train_expr, test_expr: genes x cells count matrices.

    Output:
      filtered train/test matrices and retained gene names.
    """
    min_cells = int(np.ceil(min_cell_fraction * train_expr.shape[1]))

    gene_detected_cells = (train_expr > 0).sum(axis=1)
    keep_genes = gene_detected_cells >= min_cells

    print("\nGene filtering:")
    print("Minimum detected train cells required:", min_cells)
    print("Genes before filtering:", train_expr.shape[0])
    print("Genes after filtering:", int(keep_genes.sum()))

    train_filt = train_expr.loc[keep_genes].copy()
    test_filt = test_expr.loc[keep_genes].copy()

    return {
        "train_expr": train_filt,
        "test_expr": test_filt,
        "keep_genes": train_expr.index[keep_genes].astype(str).tolist(),
    }


def normalize_log_train_test(
    train_expr: pd.DataFrame,
    test_expr: pd.DataFrame,
    scale_factor: float = 10000,
) -> dict[str, pd.DataFrame]:
    """
    Column-normalize counts and apply log1p.
    """
    train_norm = col_normalize_counts(
        train_expr,
        scale_factor=scale_factor,
    )

    test_norm = col_normalize_counts(
        test_expr,
        scale_factor=scale_factor,
    )

    train_log = safe_log1p(train_norm)
    test_log = safe_log1p(test_norm)

    return {
        "train_log": train_log,
        "test_log": test_log,
    }


def select_hvg_train_only(
    train_log: pd.DataFrame,
    test_log: pd.DataFrame,
    n_hvg: int = 2000,
) -> dict[str, Any]:
    """
    Select highly variable genes using train cells only.

    Dispersion:
      variance / (mean + 1e-8)
    """
    gene_means = train_log.mean(axis=1)
    gene_vars = train_log.var(axis=1, ddof=1)

    hvg_df = pd.DataFrame(
        {
            "gene": train_log.index.astype(str),
            "mean": gene_means.to_numpy(),
            "variance": gene_vars.to_numpy(),
        }
    )

    hvg_df["dispersion"] = hvg_df["variance"] / (hvg_df["mean"] + 1e-8)

    hvg_df = (
        hvg_df[np.isfinite(hvg_df["dispersion"])]
        .sort_values("dispersion", ascending=False)
        .reset_index(drop=True)
    )

    n_select = min(int(n_hvg), hvg_df.shape[0])
    hvg_genes = hvg_df.loc[: n_select - 1, "gene"].tolist()

    print("\nHVG selection:")
    print("Requested HVGs:", n_hvg)
    print("Selected HVGs:", len(hvg_genes))

    train_hvg = train_log.loc[hvg_genes].copy()
    test_hvg = test_log.loc[hvg_genes].copy()

    return {
        "train_hvg": train_hvg,
        "test_hvg": test_hvg,
        "hvg_genes": hvg_genes,
        "hvg_stats": hvg_df,
    }


def preprocess_for_feature_extraction(
    train_expr: pd.DataFrame,
    test_expr: pd.DataFrame,
    min_cell_fraction: float = 0.01,
    n_hvg: int = 2000,
    scale_factor: float = 10000,
) -> dict[str, Any]:
    filtered = filter_genes_train_only(
        train_expr=train_expr,
        test_expr=test_expr,
        min_cell_fraction=min_cell_fraction,
    )

    normalized = normalize_log_train_test(
        train_expr=filtered["train_expr"],
        test_expr=filtered["test_expr"],
        scale_factor=scale_factor,
    )

    hvg = select_hvg_train_only(
        train_log=normalized["train_log"],
        test_log=normalized["test_log"],
        n_hvg=n_hvg,
    )

    return {
        "train_hvg": hvg["train_hvg"],
        "test_hvg": hvg["test_hvg"],
        "keep_genes": filtered["keep_genes"],
        "hvg_genes": hvg["hvg_genes"],
        "hvg_stats": hvg["hvg_stats"],
    }


############################################################
# Scaling helper
############################################################

def scale_train_test_by_train(
    train_x: pd.DataFrame | np.ndarray,
    test_x: pd.DataFrame | np.ndarray,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """
    Scale train and test using train mean and train sample sd.

    This mirrors R's scale() behavior more closely than sklearn's
    StandardScaler because R's sd uses ddof = 1.
    """
    train_df = (
        train_x.copy()
        if isinstance(train_x, pd.DataFrame)
        else pd.DataFrame(train_x)
    )

    test_df = (
        test_x.copy()
        if isinstance(test_x, pd.DataFrame)
        else pd.DataFrame(test_x)
    )

    train_center = train_df.mean(axis=0)
    train_scale = train_df.std(axis=0, ddof=1)
    train_scale = train_scale.mask(train_scale < eps, 1.0)

    train_scaled = (train_df - train_center) / train_scale
    test_scaled = (test_df - train_center) / train_scale

    return {
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "center": train_center,
        "scale": train_scale,
    }


############################################################
# PCA feature extraction
############################################################

def fit_transform_PCA(
    train_hvg: pd.DataFrame,
    test_hvg: pd.DataFrame,
    n_components: int = 20,
) -> dict[str, Any]:
    print("\nFitting PCA feature extractor:")
    print("Requested number of components:", n_components)

    # observations = cells, features = HVG genes
    train_x = train_hvg.T.copy()
    test_x = test_hvg.T.copy()

    max_components = min(train_x.shape[0], train_x.shape[1])
    n_components_eff = min(int(n_components), max_components)

    if n_components_eff < n_components:
        warnings.warn(
            f"Requested PCA components = {n_components}, "
            f"but maximum possible components = {max_components}. "
            f"Using n_components = {n_components_eff}."
        )

    scaled = scale_train_test_by_train(train_x, test_x)

    pca_fit = PCA(
        n_components=n_components_eff,
        svd_solver="full",
    )

    train_features_arr = pca_fit.fit_transform(scaled["train_scaled"])
    test_features_arr = pca_fit.transform(scaled["test_scaled"])

    feature_names = [f"PC{i}" for i in range(1, n_components_eff + 1)]

    train_features = pd.DataFrame(
        train_features_arr,
        index=train_hvg.columns.astype(str),
        columns=feature_names,
    )

    test_features = pd.DataFrame(
        test_features_arr,
        index=test_hvg.columns.astype(str),
        columns=feature_names,
    )

    rotation = pd.DataFrame(
        pca_fit.components_.T,
        index=train_x.columns.astype(str),
        columns=feature_names,
    )

    return {
        "x_train": train_features,
        "x_test": test_features,
        "feature_names": feature_names,
        "feature_model": pca_fit,
        "preprocessing": {
            "center": scaled["center"],
            "scale": scaled["scale"],
            "rotation": rotation,
        },
    }


############################################################
# ICA feature extraction
############################################################

def fit_transform_ICA(
    train_hvg: pd.DataFrame,
    test_hvg: pd.DataFrame,
    n_components: int = 20,
    seed: int = 2026,
) -> dict[str, Any]:
    print("\nFitting ICA feature extractor:")
    print("Requested number of components:", n_components)

    # observations = cells, features = HVG genes
    train_x = train_hvg.T.copy()
    test_x = test_hvg.T.copy()

    max_components = min(train_x.shape[0], train_x.shape[1])
    n_components_eff = min(int(n_components), max_components)

    if n_components_eff < n_components:
        warnings.warn(
            f"Requested ICA components = {n_components}, "
            f"but maximum possible components = {max_components}. "
            f"Using n_components = {n_components_eff}."
        )

    scaled = scale_train_test_by_train(
        train_x=train_x,
        test_x=test_x,
    )

    ica_fit = FastICA(
        n_components=n_components_eff,
        algorithm="parallel",
        fun="logcosh",
        random_state=seed,
        max_iter=200,
        tol=1e-4,
        whiten="unit-variance",
    )

    train_features_arr = ica_fit.fit_transform(scaled["train_scaled"])
    test_features_arr = ica_fit.transform(scaled["test_scaled"])

    feature_names = [f"IC{i}" for i in range(1, n_components_eff + 1)]

    train_features = pd.DataFrame(
        train_features_arr,
        index=train_hvg.columns.astype(str),
        columns=feature_names,
    )

    test_features = pd.DataFrame(
        test_features_arr,
        index=test_hvg.columns.astype(str),
        columns=feature_names,
    )

    return {
        "x_train": train_features,
        "x_test": test_features,
        "feature_names": feature_names,
        "feature_model": ica_fit,
        "preprocessing": {
            "center": scaled["center"],
            "scale": scaled["scale"],
        },
    }


############################################################
# NMF feature extraction
############################################################

def project_nmf_test(
    basis_mat: pd.DataFrame | np.ndarray,
    test_hvg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project test cells into train-fitted NMF basis using NNLS.

    basis_mat:
      genes x rank

    test_hvg:
      genes x test cells

    Returns:
      rank x test cells coefficient matrix
    """
    if isinstance(basis_mat, pd.DataFrame):
        basis_arr = basis_mat.to_numpy(dtype=float)
        rank = basis_mat.shape[1]
    else:
        basis_arr = np.asarray(basis_mat, dtype=float)
        rank = basis_arr.shape[1]

    n_test = test_hvg.shape[1]

    coef_mat = np.full(
        shape=(rank, n_test),
        fill_value=np.nan,
        dtype=float,
    )

    for j in range(n_test):
        b = test_hvg.iloc[:, j].to_numpy(dtype=float)
        coef_j, _ = nnls(basis_arr, b)
        coef_mat[:, j] = coef_j

    coef_df = pd.DataFrame(
        coef_mat,
        index=[f"NMF{i}" for i in range(1, rank + 1)],
        columns=test_hvg.columns.astype(str),
    )

    return coef_df


def _resolve_nmf_method(nmf_method: str) -> tuple[str, str]:
    """
    Map an R NMF method name to a (beta_loss, solver) pair for
    sklearn.decomposition.NMF.

    R 'brunet' (the default) is *Kullback-Leibler divergence* with
    multiplicative updates. R 'lee' is *Frobenius* with multiplicative
    updates. Anything we don't recognize falls back to KL divergence
    so that a config typo doesn't silently switch loss functions.
    """
    method_lc = (nmf_method or "brunet").strip().lower()

    if method_lc == "brunet":
        return ("kullback-leibler", "mu")

    if method_lc in {"lee", "lee-seung", "frobenius"}:
        return ("frobenius", "mu")

    warnings.warn(
        f"Unknown nmf_method '{nmf_method}'; defaulting to "
        "beta_loss='kullback-leibler' (R 'brunet' equivalent)."
    )
    return ("kullback-leibler", "mu")


def fit_transform_NMF(
    train_hvg: pd.DataFrame,
    test_hvg: pd.DataFrame,
    n_components: int = 20,
    seed: int = 2026,
    nmf_method: str = "brunet",
    nrun: int = 1,
) -> dict[str, Any]:
    print("\nFitting NMF feature extractor:")
    print("Requested number of components:", n_components)
    print("Requested NMF method:", nmf_method)

    max_components = min(train_hvg.shape[0], train_hvg.shape[1])
    n_components_eff = min(int(n_components), max_components)

    if n_components_eff < n_components:
        warnings.warn(
            f"Requested NMF components = {n_components}, "
            f"but maximum possible components = {max_components}. "
            f"Using n_components = {n_components_eff}."
        )

    # NMF requires non-negative matrix.
    train_nonneg = train_hvg.clip(lower=0)
    test_nonneg = test_hvg.clip(lower=0)

    # sklearn NMF expects samples x features:
    # cells x genes.
    train_x = train_nonneg.T.copy()

    beta_loss, solver = _resolve_nmf_method(nmf_method)

    print(f"Using sklearn NMF with beta_loss={beta_loss!r}, solver={solver!r}")

    # 'nndsvda' is deterministic given the data, and replaces the hard
    # zeros that 'nndsvd' would produce with the data mean. The latter
    # matters because 'mu' (multiplicative updates) cannot escape exact
    # zeros, so 'nndsvd' init can collapse rows/columns to zero.
    nmf_fit = NMF(
        n_components=n_components_eff,
        init="nndsvda",
        solver=solver,
        beta_loss=beta_loss,
        random_state=seed,
        max_iter=1000,
        tol=1e-4,
    )

    train_features_arr = nmf_fit.fit_transform(train_x)

    # sklearn components_: rank x genes.
    # R basis matrix: genes x rank.
    basis_mat = pd.DataFrame(
        nmf_fit.components_.T,
        index=train_nonneg.index.astype(str),
        columns=[f"NMF{i}" for i in range(1, n_components_eff + 1)],
    )

    test_coef = project_nmf_test(
        basis_mat=basis_mat,
        test_hvg=test_nonneg,
    )

    feature_names = [f"NMF{i}" for i in range(1, n_components_eff + 1)]

    train_features = pd.DataFrame(
        train_features_arr,
        index=train_hvg.columns.astype(str),
        columns=feature_names,
    )

    test_features = pd.DataFrame(
        test_coef.T.to_numpy(dtype=float),
        index=test_hvg.columns.astype(str),
        columns=feature_names,
    )

    return {
        "x_train": train_features,
        "x_test": test_features,
        "feature_names": feature_names,
        "feature_model": nmf_fit,
        "preprocessing": {
            "basis": basis_mat,
            "nmf_method_requested": nmf_method,
            "nmf_beta_loss_used": beta_loss,
            "nmf_solver_used": solver,
            "nrun_requested": nrun,
        },
    }


############################################################
# Main feature extraction wrapper
############################################################

def extract_features(
    train_expr: pd.DataFrame,
    test_expr: pd.DataFrame,
    feature_method: str = "PCA",
    n_components: int = 20,
    min_cell_fraction: float = 0.01,
    n_hvg: int = 2000,
    scale_factor: float = 10000,
    seed: int = 2026,
    nmf_method: str = "brunet",
    nmf_nrun: int = 1,
) -> dict[str, Any]:
    feature_method = str(feature_method).upper()

    if feature_method not in {"PCA", "ICA", "NMF"}:
        raise ValueError("feature_method must be one of: PCA, ICA, NMF.")

    preprocessed = preprocess_for_feature_extraction(
        train_expr=train_expr,
        test_expr=test_expr,
        min_cell_fraction=min_cell_fraction,
        n_hvg=n_hvg,
        scale_factor=scale_factor,
    )

    if feature_method == "PCA":
        features = fit_transform_PCA(
            train_hvg=preprocessed["train_hvg"],
            test_hvg=preprocessed["test_hvg"],
            n_components=n_components,
        )

    elif feature_method == "ICA":
        features = fit_transform_ICA(
            train_hvg=preprocessed["train_hvg"],
            test_hvg=preprocessed["test_hvg"],
            n_components=n_components,
            seed=seed,
        )

    elif feature_method == "NMF":
        features = fit_transform_NMF(
            train_hvg=preprocessed["train_hvg"],
            test_hvg=preprocessed["test_hvg"],
            n_components=n_components,
            seed=seed,
            nmf_method=nmf_method,
            nrun=nmf_nrun,
        )

    features["hvg_genes"] = preprocessed["hvg_genes"]
    features["keep_genes"] = preprocessed["keep_genes"]
    features["hvg_stats"] = preprocessed["hvg_stats"]
    features["feature_method"] = feature_method
    features["n_components"] = len(features["feature_names"])
    features["n_components_requested"] = n_components

    return features


############################################################
# Build modeling data frames
############################################################

def build_feature_modeling_dfs(
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    feature_result: dict[str, Any],
) -> dict[str, Any]:
    x_train = feature_result["x_train"]
    x_test = feature_result["x_test"]

    train_cells = train_meta["Cell"].astype(str).to_numpy()
    test_cells = test_meta["Cell"].astype(str).to_numpy()

    if not np.all(x_train.index.astype(str).to_numpy() == train_cells):
        raise ValueError("Row names of x_train do not match train_meta['Cell'].")

    if not np.all(x_test.index.astype(str).to_numpy() == test_cells):
        raise ValueError("Row names of x_test do not match test_meta['Cell'].")

    train_feature_df = pd.concat(
        [
            train_meta.reset_index(drop=True),
            x_train.reset_index(drop=True),
        ],
        axis=1,
    )

    test_feature_df = pd.concat(
        [
            test_meta.reset_index(drop=True),
            x_test.reset_index(drop=True),
        ],
        axis=1,
    )

    model_df = pd.concat(
        [train_feature_df, test_feature_df],
        axis=0,
        ignore_index=True,
    )

    return {
        "train_df": train_feature_df,
        "test_df": test_feature_df,
        "model_df": model_df,
        "feature_cols": feature_result["feature_names"],
    }