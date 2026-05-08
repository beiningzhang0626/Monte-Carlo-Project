############################################################
# load_and_split.py
# Load expression matrix and metadata, align cells, create labels,
# and split train/test at patient level.
#
# Python version of 01_load_and_split.R
#
# Supports:
#   1. Binary task:
#        Stage_group = Early / Advanced
#        z_obs = 0 / 1
#
#   2. Categorical task:
#        Stage_cat = I / II_III / IV      if three_stage
#        Stage_cat = I / II / III / IV    if four_stage
#        y_cat = 0, 1, 2, ... for model backends such as XGBoost
#
# CHANGES vs previous version:
#   - load_and_split_data() now reads an optional `fixed_test_patients`
#     field from the config and forwards it to make_patient_split().
#     Use this to reproduce R's exact split in Python by pasting in
#     the patient IDs from R's split_info$Patient[split == "test"].
############################################################

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from utils import (
    check_required_cols,
    coalesce,
    create_binary_label,
    create_categorical_label,
    create_stage_labels,
    ensure_dir,
    get_categorical_class_levels,
    get_task_label_col,
    log_msg,
    make_patient_split,
    normalize_categorical_scheme,
    normalize_task_type,
    print_dim,
)


############################################################
# Load expression matrix
############################################################

def load_expression_matrix(expr_file: str | Path) -> pd.DataFrame:
    """
    Load expression matrix from an RDS file.

    Expected R format:
      - first column: gene names
      - remaining columns: cell IDs
      - values: expression counts

    Returned Python format:
      - pandas DataFrame
      - rows = genes
      - columns = cells
    """
    expr_file = Path(expr_file)

    log_msg("Loading expression data from: ", expr_file)

    if expr_file.suffix.lower() == ".rds":
        try:
            import pyreadr
        except ImportError as exc:
            raise ImportError(
                "Reading .rds files requires pyreadr. "
                "Install it with: pip install pyreadr"
            ) from exc

        result = pyreadr.read_r(str(expr_file))

        if len(result) == 0:
            raise ValueError(f"No object found inside RDS file: {expr_file}")

        expr_dt = next(iter(result.values()))

    elif expr_file.suffix.lower() in {".csv", ".txt"}:
        expr_dt = pd.read_csv(expr_file)

    elif expr_file.suffix.lower() in {".parquet"}:
        expr_dt = pd.read_parquet(expr_file)

    else:
        raise ValueError(
            f"Unsupported expression file format: {expr_file.suffix}. "
            "Expected .rds, .csv, .txt, or .parquet."
        )

    print_dim(expr_dt, "Expression data table")

    gene_names = expr_dt.iloc[:, 0].astype(str).to_numpy()
    cell_ids = expr_dt.columns[1:].astype(str).tolist()

    expr_mat = expr_dt.iloc[:, 1:].copy()
    expr_mat.index = gene_names
    expr_mat.columns = cell_ids

    # Make sure expression values are numeric.
    expr_mat = expr_mat.apply(pd.to_numeric, errors="coerce")

    if expr_mat.isna().any().any():
        raise ValueError(
            "Expression matrix contains NA after numeric conversion. "
            "Please check input expression data."
        )

    print_dim(expr_mat, "Expression matrix, genes x cells")

    return expr_mat


############################################################
# Load metadata
############################################################

def load_metadata(meta_file: str | Path) -> pd.DataFrame:
    """
    Load metadata CSV.
    """
    meta_file = Path(meta_file)

    log_msg("Loading metadata from: ", meta_file)

    meta = pd.read_csv(meta_file)

    print_dim(meta, "Metadata")

    print("\nMetadata columns:")
    print(list(meta.columns))

    return meta


############################################################
# Match metadata rows to expression matrix columns
############################################################

def match_metadata_to_expression(
    meta: pd.DataFrame,
    expr_mat: pd.DataFrame,
    cell_id_col: str = "Index",
) -> pd.DataFrame:
    """
    Reorder metadata rows to match expression matrix columns.
    """
    check_required_cols(meta, [cell_id_col], object_name="metadata")

    cell_ids = list(expr_mat.columns)

    meta_indexed = meta.set_index(cell_id_col, drop=False)

    missing_cells = [cell for cell in cell_ids if cell not in meta_indexed.index]
    n_unmatched = len(missing_cells)

    print("\nNumber of unmatched cells after metadata matching:")
    print(n_unmatched)

    if n_unmatched > 0:
        raise ValueError(
            f"Some expression matrix cell IDs were not found in metadata${cell_id_col}. "
            "Please check cell IDs."
        )

    meta_matched = meta_indexed.loc[cell_ids].copy()

    print("\nFirst few matched cell IDs:")
    print(
        pd.DataFrame(
            {
                "expr_cell": cell_ids[:10],
                "meta_cell": meta_matched[cell_id_col].head(10).to_numpy(),
            }
        )
    )

    meta_matched["Cell"] = meta_matched[cell_id_col].astype(str).to_numpy()
    meta_matched = meta_matched.reset_index(drop=True)

    return meta_matched


############################################################
# Filter cells with missing task label
############################################################

def filter_missing_task_label_cells(
    expr_mat: pd.DataFrame,
    meta: pd.DataFrame,
    task_type: str = "binary",
) -> dict[str, Any]:
    task_type = normalize_task_type(task_type)
    label_col = get_task_label_col(task_type)

    check_required_cols(meta, [label_col], object_name="metadata")

    keep_cells = ~meta[label_col].isna()

    expr_mat_filtered = expr_mat.loc[:, keep_cells.to_numpy()].copy()
    meta_filtered = meta.loc[keep_cells].copy().reset_index(drop=True)

    print("\nAfter filtering cells with missing task label:")
    print("Task type:", task_type)
    print("Label column:", label_col)

    print_dim(expr_mat_filtered, "Filtered expression matrix")
    print_dim(meta_filtered, "Filtered metadata")

    return {
        "expr_mat": expr_mat_filtered,
        "meta": meta_filtered,
    }


############################################################
# Add observed labels for binary and categorical tasks
############################################################

def add_observed_labels(
    meta: pd.DataFrame,
    categorical_scheme: str = "three_stage",
) -> pd.DataFrame:
    categorical_scheme = normalize_categorical_scheme(categorical_scheme)
    class_levels = get_categorical_class_levels(categorical_scheme)

    meta = meta.copy()

    ############################################################
    # Binary observed label
    ############################################################

    meta["z_obs"] = create_binary_label(meta["Stage_group"])

    ############################################################
    # Categorical observed label
    ############################################################

    meta["y_cat"] = create_categorical_label(
        stage_cat=meta["Stage_cat"],
        class_levels=class_levels,
    )

    meta["Stage_cat"] = pd.Categorical(
        meta["Stage_cat"],
        categories=class_levels,
        ordered=True,
    )

    meta.attrs["class_levels"] = class_levels

    print("\nBinary label check: Stage_group by z_obs")
    print(
        pd.crosstab(
            meta["Stage_group"],
            meta["z_obs"],
            dropna=False,
        )
    )

    print("\nCategorical label check: Stage_cat by y_cat")
    print(
        pd.crosstab(
            meta["Stage_cat"],
            meta["y_cat"],
            dropna=False,
        )
    )

    print("\nPatient counts by Stage_group:")
    print(
        meta[["Patient", "Stage_group", "z_obs"]]
        .drop_duplicates()
        .groupby(["Stage_group", "z_obs"], observed=True)
        .size()
        .reset_index(name="n_patients")
    )

    print("\nPatient counts by Stage_cat:")
    print(
        meta[["Patient", "Stage_cat", "y_cat"]]
        .drop_duplicates()
        .groupby(["Stage_cat", "y_cat"], observed=True)
        .size()
        .reset_index(name="n_patients")
    )

    return meta


############################################################
# Add train/test split by patient
############################################################

def add_train_test_split(
    meta: pd.DataFrame,
    task_type: str = "binary",
    patient_col: str = "Patient",
    test_fraction: float = 0.25,
    seed: int = 2026,
    fixed_test_patients: Sequence[Any] | None = None,
) -> pd.DataFrame:
    task_type = normalize_task_type(task_type)
    label_col = get_task_label_col(task_type)

    meta = meta.copy()

    meta["split"] = make_patient_split(
        meta=meta,
        label_col=label_col,
        patient_col=patient_col,
        test_fraction=test_fraction,
        seed=seed,
        fixed_test_patients=fixed_test_patients,
    )

    print("\nPatient split counts:")
    print(
        meta[[patient_col, label_col, "split"]]
        .drop_duplicates()
        .groupby(["split", label_col], observed=True)
        .size()
        .reset_index(name="n_patients")
    )

    print("\nCell split counts:")
    print(
        pd.crosstab(
            meta["split"],
            meta[label_col],
            dropna=False,
        )
    )

    return meta


############################################################
# Main data loading and split function
############################################################

def load_and_split_data(config: dict[str, Any]) -> dict[str, Any]:
    required_config = [
        "proj_dir",
        "in_dir",
        "out_dir",
        "expr_file",
        "meta_file",
        "seed",
        "test_fraction",
    ]

    missing_config = [key for key in required_config if key not in config]

    if missing_config:
        raise ValueError(
            "Missing config fields: "
            + ", ".join(missing_config)
        )

    ############################################################
    # Resolve task settings
    ############################################################

    task_type = normalize_task_type(
        coalesce(config.get("task_type"), "binary")
    )

    categorical_scheme = normalize_categorical_scheme(
        coalesce(config.get("categorical_scheme"), "three_stage")
    )

    class_levels = get_categorical_class_levels(categorical_scheme)

    ############################################################
    # Optional fixed test patient override
    ############################################################

    fixed_test_patients = config.get("fixed_test_patients")

    if fixed_test_patients is not None:
        # Coerce to a flat list of stripped strings to be forgiving with
        # the way YAML parses lists of IDs.
        fixed_test_patients = [
            str(p).strip() for p in list(fixed_test_patients) if p is not None
        ]
        if len(fixed_test_patients) == 0:
            fixed_test_patients = None

    print("\nData loading task settings:")
    print("task_type:", task_type)
    print("categorical_scheme:", categorical_scheme)
    print("categorical class levels:", ", ".join(class_levels))
    if fixed_test_patients is not None:
        print(
            "fixed_test_patients (overrides RNG-based split):",
            ", ".join(fixed_test_patients),
        )

    ensure_dir(config["out_dir"])

    ############################################################
    # 1. Load expression and metadata
    ############################################################

    expr_mat = load_expression_matrix(config["expr_file"])
    meta = load_metadata(config["meta_file"])

    check_required_cols(
        meta,
        ["Index", "Patient", "Stage"],
        object_name="metadata",
    )

    ############################################################
    # 2. Align metadata to expression columns
    ############################################################

    meta = match_metadata_to_expression(
        meta=meta,
        expr_mat=expr_mat,
        cell_id_col="Index",
    )

    ############################################################
    # 3. Create binary and categorical labels
    ############################################################

    meta = create_stage_labels(
        meta=meta,
        categorical_scheme=categorical_scheme,
    )

    print("\nCell counts by detailed Stage:")
    print(meta["Stage"].value_counts(dropna=False))

    print("\nCell counts by broad Stage:")
    print(meta["Stage_broad"].value_counts(dropna=False))

    print("\nCell counts by binary Stage_group:")
    print(meta["Stage_group"].value_counts(dropna=False))

    print("\nCell counts by categorical Stage_cat:")
    print(meta["Stage_cat"].value_counts(dropna=False))

    ############################################################
    # 4. Filter cells with missing label for selected task
    ############################################################

    filtered = filter_missing_task_label_cells(
        expr_mat=expr_mat,
        meta=meta,
        task_type=task_type,
    )

    expr_mat = filtered["expr_mat"]
    meta = filtered["meta"]

    ############################################################
    # 5. Add observed labels for both binary and categorical tasks
    ############################################################

    meta = add_observed_labels(
        meta=meta,
        categorical_scheme=categorical_scheme,
    )

    ############################################################
    # 6. Patient-level train/test split by selected task label
    ############################################################

    meta = add_train_test_split(
        meta=meta,
        task_type=task_type,
        patient_col="Patient",
        test_fraction=config["test_fraction"],
        seed=config["seed"],
        fixed_test_patients=fixed_test_patients,
    )

    ############################################################
    # 7. Build train/test expression and metadata objects
    ############################################################

    train_idx = meta["split"] == "train"
    test_idx = meta["split"] == "test"

    train_expr = expr_mat.loc[:, train_idx.to_numpy()].copy()
    test_expr = expr_mat.loc[:, test_idx.to_numpy()].copy()

    train_meta = meta.loc[train_idx].copy().reset_index(drop=True)
    test_meta = meta.loc[test_idx].copy().reset_index(drop=True)

    ############################################################
    # 8. Build split info
    ############################################################

    split_info_cols = [
        "Cell",
        "Patient",
        "Stage",
        "Stage_broad",
        "Stage_group",
        "z_obs",
        "Stage_cat",
        "y_cat",
        "split",
    ]

    split_info_cols = [
        col for col in split_info_cols if col in meta.columns
    ]

    split_info = meta.loc[:, split_info_cols].copy()

    ############################################################
    # 9. Return data object
    ############################################################

    return {
        "task_type": task_type,
        "categorical_scheme": categorical_scheme,
        "class_levels": class_levels,

        "expr_mat": expr_mat,
        "meta": meta,

        "train_expr": train_expr,
        "test_expr": test_expr,

        "train_meta": train_meta,
        "test_meta": test_meta,

        "split_info": split_info,
    }