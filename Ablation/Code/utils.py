############################################################
# utils.py
# General utility functions for MMIL / MCEM ablation pipeline
#
# Python version of 00_utils.R
#
# Supports:
#   1. Binary task:
#        Early vs Advanced
#   2. Categorical task:
#        three_stage: I / II_III / IV
#        four_stage:  I / II / III / IV
#
# CHANGES vs previous version:
#   - make_patient_split() now accepts an optional `fixed_test_patients`
#     argument. When provided, the RNG is bypassed and the supplied
#     patient IDs are used as the test set verbatim. This is the
#     recommended way to do an apples-to-apples comparison against the
#     R pipeline, which uses a different RNG and therefore picks a
#     different test set even with the same numeric seed.
############################################################

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


############################################################
# Basic helper
############################################################

def coalesce(x: Any, y: Any) -> Any:
    """
    Python equivalent of R `%||%`.
    Return y if x is None; otherwise return x.
    """
    return y if x is None else x


############################################################
# File and directory helpers
############################################################

def ensure_dir(path: str | os.PathLike) -> Path:
    """
    Create directory recursively if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_write_csv(
    x: pd.DataFrame,
    file: str | os.PathLike,
    index: bool = False,
) -> Path:
    """
    Write CSV after making sure the output directory exists.
    """
    file = Path(file)
    ensure_dir(file.parent)
    x.to_csv(file, index=index)
    return file


def log_msg(*args: Any) -> None:
    """
    Print a message with blank lines, similar to the R helper.
    """
    print()
    print("".join(map(str, args)))
    print()


def print_dim(x: Any, name: str = "Object") -> None:
    """
    Print dimensions of a pandas / numpy object.
    """
    print(f"\n{name} dimensions:")
    if hasattr(x, "shape"):
        print(x.shape)
    else:
        print(None)


############################################################
# Data checks
############################################################

def check_required_cols(
    df: pd.DataFrame,
    required_cols: Sequence[str],
    object_name: str = "data frame",
) -> bool:
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in {object_name}: "
            f"{', '.join(missing_cols)}"
        )

    return True


def stop_if_any_na(x: Any, message: str) -> bool:
    if pd.isna(x).any():
        raise ValueError(message)
    return True


############################################################
# Task helpers
############################################################

def normalize_task_type(task_type: str = "binary") -> str:
    task_type = str(task_type).lower()

    if task_type not in {"binary", "categorical"}:
        raise ValueError("task_type must be either 'binary' or 'categorical'.")

    return task_type


def normalize_categorical_scheme(
    categorical_scheme: str = "three_stage",
) -> str:
    categorical_scheme = str(categorical_scheme).lower()

    if categorical_scheme not in {"three_stage", "four_stage"}:
        raise ValueError(
            "categorical_scheme must be either 'three_stage' or 'four_stage'."
        )

    return categorical_scheme


def get_categorical_class_levels(
    categorical_scheme: str = "three_stage",
) -> list[str]:
    categorical_scheme = normalize_categorical_scheme(categorical_scheme)

    if categorical_scheme == "three_stage":
        return ["I", "II_III", "IV"]

    if categorical_scheme == "four_stage":
        return ["I", "II", "III", "IV"]

    raise ValueError(f"Unknown categorical_scheme: {categorical_scheme}")


def get_task_label_col(task_type: str = "binary") -> str:
    task_type = normalize_task_type(task_type)

    if task_type == "binary":
        return "Stage_group"

    if task_type == "categorical":
        return "Stage_cat"

    raise ValueError(f"Unknown task_type: {task_type}")


############################################################
# Stage label helpers
############################################################

def create_stage_labels(
    meta: pd.DataFrame,
    categorical_scheme: str = "three_stage",
) -> pd.DataFrame:
    """
    Create:
      - Stage_broad: I / II / III / IV
      - Stage_group: Early / Advanced
      - Stage_cat: categorical stage label
    """
    check_required_cols(meta, ["Stage"], object_name="metadata")

    categorical_scheme = normalize_categorical_scheme(categorical_scheme)
    meta = meta.copy()

    ############################################################
    # Broad stage: I / II / III / IV
    ############################################################

    stage = meta["Stage"].astype("string")

    stage_broad = pd.Series(pd.NA, index=meta.index, dtype="string")

    stage_broad[stage.isin(["IA", "IA2", "IA3", "IB"])] = "I"
    stage_broad[stage.isin(["IIA", "IIB"])] = "II"
    stage_broad[stage.isin(["IIIA"])] = "III"
    stage_broad[stage.isin(["IV"])] = "IV"

    meta["Stage_broad"] = pd.Categorical(
        stage_broad,
        categories=["I", "II", "III", "IV"],
        ordered=True,
    )

    ############################################################
    # Binary label: Early / Advanced
    ############################################################

    stage_broad_str = pd.Series(meta["Stage_broad"], index=meta.index).astype("string")

    stage_group = pd.Series(pd.NA, index=meta.index, dtype="string")
    stage_group[stage_broad_str.isin(["I", "II"])] = "Early"
    stage_group[stage_broad_str.isin(["III", "IV"])] = "Advanced"

    meta["Stage_group"] = pd.Categorical(
        stage_group,
        categories=["Early", "Advanced"],
        ordered=True,
    )

    ############################################################
    # Categorical label
    ############################################################

    if categorical_scheme == "three_stage":
        stage_cat = pd.Series(pd.NA, index=meta.index, dtype="string")
        stage_cat[stage_broad_str == "I"] = "I"
        stage_cat[stage_broad_str.isin(["II", "III"])] = "II_III"
        stage_cat[stage_broad_str == "IV"] = "IV"

        meta["Stage_cat"] = pd.Categorical(
            stage_cat,
            categories=["I", "II_III", "IV"],
            ordered=True,
        )

    elif categorical_scheme == "four_stage":
        meta["Stage_cat"] = pd.Categorical(
            stage_broad_str,
            categories=["I", "II", "III", "IV"],
            ordered=True,
        )

    return meta


def create_binary_label(stage_group: pd.Series | Sequence[Any]) -> np.ndarray:
    """
    Advanced -> 1
    Early    -> 0
    """
    s = pd.Series(stage_group).astype("string")
    return np.where(s == "Advanced", 1, 0)


def create_categorical_label(
    stage_cat: pd.Series | Sequence[Any],
    class_levels: Sequence[str],
) -> np.ndarray:
    """
    Convert categorical labels to 0, 1, 2, ...
    Missing labels become -1.
    """
    cat = pd.Categorical(stage_cat, categories=list(class_levels), ordered=True)
    return cat.codes.astype(int)


def make_onehot(
    y: pd.Series | Sequence[Any],
    class_levels: Sequence[str],
) -> pd.DataFrame:
    y_cat = pd.Categorical(y, categories=list(class_levels), ordered=True)

    out = pd.DataFrame(
        0,
        index=np.arange(len(y_cat)),
        columns=list(class_levels),
        dtype=float,
    )

    y_str = pd.Series(y_cat.astype("object"))

    for cls in class_levels:
        out[cls] = (y_str == cls).astype(float).to_numpy()

    return out


############################################################
# Cell type helper
############################################################

def detect_celltype_column(meta: pd.DataFrame) -> str:
    possible_celltype_cols = [
        "Cell_type",
        "CellType",
        "cell_type",
        "Major_cell_type",
        "MajorCellType",
        "Annotation",
        "annotation",
        "Cell_subtype",
        "Cell_subtype_1",
        "Cell_type.refined",
    ]

    for col in possible_celltype_cols:
        if col in meta.columns:
            return col

    raise ValueError(
        "Could not automatically detect a cell type column. "
        "Please check meta.columns and set celltype_col manually."
    )


############################################################
# Matrix helpers
############################################################

def row_normalize(
    mat: pd.DataFrame | np.ndarray,
    eps: float = 1e-12,
) -> pd.DataFrame | np.ndarray:
    """
    Normalize each row to sum to one.
    """
    if isinstance(mat, pd.DataFrame):
        arr = mat.to_numpy(dtype=float)
        arr = np.maximum(arr, eps)
        arr = arr / arr.sum(axis=1, keepdims=True)
        return pd.DataFrame(arr, index=mat.index, columns=mat.columns)

    arr = np.asarray(mat, dtype=float)
    arr = np.maximum(arr, eps)
    return arr / arr.sum(axis=1, keepdims=True)


def col_normalize_counts(
    mat: pd.DataFrame | np.ndarray,
    scale_factor: float = 10000,
) -> pd.DataFrame | np.ndarray:
    """
    Column-normalize a genes x cells count matrix.
    """
    if isinstance(mat, pd.DataFrame):
        arr = mat.to_numpy(dtype=float)
        lib_size = arr.sum(axis=0)

        if np.any(lib_size <= 0):
            raise ValueError("Some cells have zero total counts after gene filtering.")

        out = arr / lib_size[None, :] * scale_factor
        return pd.DataFrame(out, index=mat.index, columns=mat.columns)

    arr = np.asarray(mat, dtype=float)
    lib_size = arr.sum(axis=0)

    if np.any(lib_size <= 0):
        raise ValueError("Some cells have zero total counts after gene filtering.")

    return arr / lib_size[None, :] * scale_factor


def safe_log1p(mat: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    if isinstance(mat, pd.DataFrame):
        return pd.DataFrame(
            np.log1p(mat.to_numpy(dtype=float)),
            index=mat.index,
            columns=mat.columns,
        )

    return np.log1p(np.asarray(mat, dtype=float))


############################################################
# Train/test split helpers
############################################################

def make_patient_split(
    meta: pd.DataFrame,
    label_col: str = "Stage_group",
    patient_col: str = "Patient",
    test_fraction: float = 0.25,
    seed: int = 2026,
    fixed_test_patients: Sequence[Any] | None = None,
) -> np.ndarray:
    """
    Patient-level stratified split.

    Equivalent logic to the R version:
      - split by patient, not by cell
      - for each class, keep at least one patient in train
      - if a class has only one patient, keep it entirely in train

    Parameters
    ----------
    fixed_test_patients : sequence or None, default None
        If provided, RNG is bypassed and the patients in this list are
        used as the test set verbatim. Use this to reproduce R's split
        in Python (e.g. paste in the patient IDs from R's
        `split_info$Patient[split == "test"]`). The function still
        validates that every class label is present in the resulting
        train set.

        If None, perform a fresh stratified random split using `seed`.
    """
    check_required_cols(
        meta,
        [patient_col, label_col],
        object_name="metadata",
    )

    ##########################################################
    # Branch 1: caller has supplied an explicit test patient set.
    ##########################################################

    if fixed_test_patients is not None:
        fixed_test_patients = list(fixed_test_patients)

        all_patients = pd.Series(meta[patient_col]).dropna().unique().tolist()
        unknown = [p for p in fixed_test_patients if p not in all_patients]

        if unknown:
            warnings.warn(
                "fixed_test_patients contains IDs not present in metadata: "
                + ", ".join(map(str, unknown))
            )

        split = np.where(
            meta[patient_col].isin(fixed_test_patients),
            "test",
            "train",
        )

        train_labels = (
            pd.Series(meta.loc[split == "train", label_col])
            .dropna()
            .unique()
            .tolist()
        )
        all_labels = pd.Series(meta[label_col]).dropna().unique().tolist()
        missing_train = [lab for lab in all_labels if lab not in train_labels]

        if missing_train:
            raise ValueError(
                "Train split (with fixed_test_patients) is missing class(es): "
                + ", ".join(map(str, missing_train))
                + ". Multiclass models cannot be run reliably."
            )

        n_test_patients = int(
            pd.Series(meta.loc[split == "test", patient_col]).dropna().nunique()
        )
        n_train_patients = int(
            pd.Series(meta.loc[split == "train", patient_col]).dropna().nunique()
        )

        print(
            "[make_patient_split] Using fixed_test_patients: "
            f"{n_train_patients} train patients, {n_test_patients} test patients."
        )

        return split

    ##########################################################
    # Branch 2: stratified random split by patient.
    ##########################################################

    rng = np.random.default_rng(seed)

    patient_labels = (
        meta.loc[~meta[label_col].isna(), [patient_col, label_col]]
        .drop_duplicates()
        .copy()
    )
    patient_labels.columns = ["Patient", "Label"]

    test_patients: list[Any] = []

    for lab in patient_labels["Label"].dropna().unique():
        pts = patient_labels.loc[patient_labels["Label"] == lab, "Patient"].to_numpy()
        n_pts = len(pts)

        if n_pts < 2:
            warnings.warn(
                f"Only {n_pts} patient(s) for class '{lab}'; "
                "keeping all in train so the model can see this class."
            )
            continue

        n_test = int(np.ceil(test_fraction * n_pts))
        n_test = max(1, n_test)
        n_test = min(n_test, n_pts - 1)

        sampled = rng.choice(pts, size=n_test, replace=False)
        test_patients.extend(sampled.tolist())

    split = np.where(
        meta[patient_col].isin(test_patients),
        "test",
        "train",
    )

    train_labels = pd.Series(meta.loc[split == "train", label_col]).dropna().unique()
    all_labels = patient_labels["Label"].dropna().unique()

    missing_train_labels = [lab for lab in all_labels if lab not in train_labels]

    if missing_train_labels:
        raise ValueError(
            "Train split is missing class(es): "
            + ", ".join(map(str, missing_train_labels))
            + ". Multiclass models cannot be run reliably."
        )

    return split


############################################################
# Binary metrics
############################################################

def log_loss_binary(
    y: Sequence[float] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    eps: float = 1e-8,
) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1 - eps)

    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def compute_auc_safe(
    y: Sequence[float] | np.ndarray,
    score: Sequence[float] | np.ndarray,
) -> float:
    y = np.asarray(y)
    score = np.asarray(score)

    if len(np.unique(y[~pd.isna(y)])) < 2:
        return np.nan

    if len(np.unique(score[~pd.isna(score)])) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan


############################################################
# Multiclass metrics
############################################################

def multiclass_log_loss(
    y_true: Sequence[Any],
    prob_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
    eps: float = 1e-8,
) -> float:
    class_levels = list(class_levels)
    y_true_cat = pd.Categorical(y_true, categories=class_levels, ordered=True)
    y_codes = y_true_cat.codes

    if isinstance(prob_mat, pd.DataFrame):
        prob_df = prob_mat.loc[:, class_levels].copy()
    else:
        prob_df = pd.DataFrame(prob_mat, columns=class_levels)

    prob_df = row_normalize(prob_df)
    prob_arr = np.clip(prob_df.to_numpy(dtype=float), eps, 1 - eps)

    valid = y_codes >= 0
    if not np.any(valid):
        return np.nan

    losses = -np.log(prob_arr[np.arange(len(y_codes))[valid], y_codes[valid]])
    return float(np.mean(losses))


def multiclass_accuracy(
    y_true: Sequence[Any],
    prob_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
) -> float:
    class_levels = list(class_levels)
    y_true_cat = pd.Categorical(y_true, categories=class_levels, ordered=True)
    y_true_str = pd.Series(y_true_cat.astype("object"))

    if isinstance(prob_mat, pd.DataFrame):
        prob_arr = prob_mat.loc[:, class_levels].to_numpy(dtype=float)
    else:
        prob_arr = np.asarray(prob_mat, dtype=float)

    pred_idx = np.argmax(prob_arr, axis=1)
    y_pred = np.asarray(class_levels, dtype=object)[pred_idx]

    valid = ~pd.isna(y_true_str)
    if valid.sum() == 0:
        return np.nan

    return float(np.mean(y_pred[valid.to_numpy()] == y_true_str[valid].to_numpy()))


def macro_f1_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    class_levels: Sequence[str],
) -> float:
    y_true_cat = pd.Categorical(y_true, categories=list(class_levels), ordered=True)
    y_pred_cat = pd.Categorical(y_pred, categories=list(class_levels), ordered=True)

    y_true_s = pd.Series(y_true_cat.astype("object"))
    y_pred_s = pd.Series(y_pred_cat.astype("object"))

    f1_values = []

    for cls in class_levels:
        tp = int(((y_true_s == cls) & (y_pred_s == cls)).sum())
        fp = int(((y_true_s != cls) & (y_pred_s == cls)).sum())
        fn = int(((y_true_s == cls) & (y_pred_s != cls)).sum())

        precision = np.nan if tp + fp == 0 else tp / (tp + fp)
        recall = np.nan if tp + fn == 0 else tp / (tp + fn)

        if (
            np.isnan(precision)
            or np.isnan(recall)
            or precision + recall == 0
        ):
            f1_values.append(np.nan)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))

    return float(np.nanmean(f1_values))


def balanced_accuracy_multiclass(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    class_levels: Sequence[str],
) -> float:
    y_true_cat = pd.Categorical(y_true, categories=list(class_levels), ordered=True)
    y_pred_cat = pd.Categorical(y_pred, categories=list(class_levels), ordered=True)

    y_true_s = pd.Series(y_true_cat.astype("object"))
    y_pred_s = pd.Series(y_pred_cat.astype("object"))

    recall_values = []

    for cls in class_levels:
        denom = int((y_true_s == cls).sum())

        if denom == 0:
            recall_values.append(np.nan)
        else:
            recall_values.append(
                int(((y_true_s == cls) & (y_pred_s == cls)).sum()) / denom
            )

    return float(np.nanmean(recall_values))


############################################################
# Soft-label expansion helpers
############################################################

def expand_binary_soft_labels(
    x: pd.DataFrame | np.ndarray,
    q: Sequence[float] | np.ndarray,
    eps: float = 1e-8,
) -> dict[str, Any]:
    q = np.clip(np.asarray(q, dtype=float), 0, 1)
    n = len(q)

    if isinstance(x, pd.DataFrame):
        x_expanded = pd.concat([x, x], axis=0, ignore_index=True)
    else:
        x_arr = np.asarray(x)
        x_expanded = np.concatenate([x_arr, x_arr], axis=0)

    y_expanded = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    w_expanded = np.concatenate([1 - q, q])

    keep = w_expanded > eps

    if isinstance(x_expanded, pd.DataFrame):
        x_expanded = x_expanded.loc[keep].reset_index(drop=True)
    else:
        x_expanded = x_expanded[keep]

    y_expanded = y_expanded[keep]
    w_expanded = w_expanded[keep]
    w_expanded = w_expanded / np.mean(w_expanded)

    return {
        "x": x_expanded,
        "y": y_expanded,
        "weight": w_expanded,
    }


def expand_multiclass_soft_labels(
    x: pd.DataFrame | np.ndarray,
    q_mat: pd.DataFrame | np.ndarray,
    class_levels: Sequence[str],
    eps: float = 1e-8,
) -> dict[str, Any]:
    class_levels = list(class_levels)

    if isinstance(q_mat, pd.DataFrame):
        q_df = q_mat.copy()
        if q_df.columns.isnull().any():
            q_df.columns = class_levels
        q_df = q_df.loc[:, class_levels]
    else:
        q_df = pd.DataFrame(q_mat, columns=class_levels)

    q_df = row_normalize(q_df)

    n = q_df.shape[0]
    K = q_df.shape[1]

    if K != len(class_levels):
        raise ValueError("Number of columns in q_mat must match length(class_levels).")

    if isinstance(x, pd.DataFrame):
        x_expanded = pd.concat([x] * K, axis=0, ignore_index=True)
    else:
        x_arr = np.asarray(x)
        x_expanded = np.concatenate([x_arr] * K, axis=0)

    y_expanded = np.concatenate([
        np.repeat(cls, n) for cls in class_levels
    ])

    w_expanded = q_df.to_numpy(dtype=float).reshape(-1, order="F")

    keep = w_expanded > eps

    if isinstance(x_expanded, pd.DataFrame):
        x_expanded = x_expanded.loc[keep].reset_index(drop=True)
    else:
        x_expanded = x_expanded[keep]

    y_expanded = y_expanded[keep]
    w_expanded = w_expanded[keep]
    w_expanded = w_expanded / np.mean(w_expanded)

    return {
        "x": x_expanded,
        "y": y_expanded,
        "weight": w_expanded,
    }


############################################################
# Experiment naming helper
############################################################

def make_experiment_name(config: dict[str, Any]) -> str:
    task_type = coalesce(config.get("task_type"), "binary")
    categorical_scheme = config.get("categorical_scheme")

    parts = [
        task_type,
    ]

    if categorical_scheme is not None and task_type == "categorical":
        parts.append(str(categorical_scheme))

    parts.extend([
        str(config.get("feature_method")),
        f"k{config.get('n_components')}",
        str(config.get("model_backend")),
    ])

    if config.get("glmnet_alpha") is not None:
        parts.append(f"alpha{config.get('glmnet_alpha')}")

    parts.append(f"seed{config.get('seed')}")

    return "_".join(parts)