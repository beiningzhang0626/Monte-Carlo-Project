############################################################
# patient_level_evaluation.py
# Patient-level aggregation evaluation for binary and categorical
# MMIL / MCEM experiments.
#
# Python version of 05_patient_level_evaluation.R
#
# Supported task types:
#   1. binary
#   2. categorical
############################################################

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from utils import (
    balanced_accuracy_multiclass,
    check_required_cols,
    compute_auc_safe,
    ensure_dir,
    log_loss_binary,
    macro_f1_score,
    multiclass_log_loss,
    normalize_task_type,
    row_normalize,
)


############################################################
# Shared helper
############################################################

def sanitize_class_name_eval(x: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x))


def get_aggregation_cols_binary() -> list[str]:
    return [
        "mean_prob",
        "median_prob",
        "q90_prob",
        "q99_prob",
        "prop_gt_0.5",
        "prop_gt_0.9",
    ]


def get_aggregation_cols_categorical() -> list[str]:
    return [
        "mean_prob",
        "median_prob",
        "q90_prob",
        "q99_prob",
    ]


def _safe_quantile(x: pd.Series, q: float) -> float:
    return float(x.dropna().quantile(q)) if x.dropna().shape[0] > 0 else np.nan


def _ensure_cell_and_celltype(cell_pred: pd.DataFrame) -> pd.DataFrame:
    cell_pred = cell_pred.copy()

    if "Cell_type" not in cell_pred.columns:
        cell_pred["Cell_type"] = "Unknown"

    if "Cell" not in cell_pred.columns:
        cell_pred["Cell"] = [
            f"Cell_{i}" for i in range(1, cell_pred.shape[0] + 1)
        ]

    return cell_pred


def _save_grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_file: str | Path,
    width: float = 10,
    height: float = 6,
) -> str:
    import matplotlib.pyplot as plt

    output_file = Path(output_file)
    ensure_dir(output_file.parent)

    pivot = df.pivot_table(
        index=x_col,
        columns=hue_col,
        values=y_col,
        aggfunc="mean",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(width, height),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=hue_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    return str(output_file)


############################################################
# Binary evaluation: load and check prediction file
############################################################

def load_binary_prediction_file(pred_file: str | Path) -> pd.DataFrame:
    pred_file = Path(pred_file)

    print("\nLoading binary cell-level prediction file:")
    print(pred_file)

    cell_pred = pd.read_csv(pred_file)

    print("\nLoaded cell-level prediction file dimensions:")
    print(cell_pred.shape)

    print("\nColumn names:")
    print(list(cell_pred.columns))

    required_cols = [
        "Patient",
        "split",
        "Stage_group",
        "naive_prob",
        "em_mmil_prob",
        "mcem_mmil_prob",
    ]

    check_required_cols(
        cell_pred,
        required_cols,
        object_name="binary cell-level prediction file",
    )

    cell_pred = _ensure_cell_and_celltype(cell_pred)

    print("\nCell counts by split and Stage_group:")
    print(pd.crosstab(cell_pred["split"], cell_pred["Stage_group"]))

    print("\nPatient counts by split and Stage_group:")
    print(
        cell_pred[["Patient", "split", "Stage_group"]]
        .drop_duplicates()
        .groupby(["split", "Stage_group"])
        .size()
        .reset_index(name="n_patients")
    )

    return cell_pred


############################################################
# Binary evaluation: prepare long probability table
############################################################

def prepare_binary_probability_long(
    cell_pred: pd.DataFrame,
) -> pd.DataFrame:
    cell_pred = cell_pred.copy()

    cell_pred["y_patient"] = np.where(
        cell_pred["Stage_group"].astype(str) == "Advanced",
        1,
        0,
    )

    print("\nCheck binary patient-level label:")
    print(pd.crosstab(cell_pred["Stage_group"], cell_pred["y_patient"]))

    prob_long = cell_pred[
        [
            "Cell",
            "Patient",
            "split",
            "Stage_group",
            "y_patient",
            "Cell_type",
            "naive_prob",
            "em_mmil_prob",
            "mcem_mmil_prob",
        ]
    ].melt(
        id_vars=[
            "Cell",
            "Patient",
            "split",
            "Stage_group",
            "y_patient",
            "Cell_type",
        ],
        value_vars=[
            "naive_prob",
            "em_mmil_prob",
            "mcem_mmil_prob",
        ],
        var_name="model",
        value_name="cell_prob",
    )

    model_map = {
        "naive_prob": "Naive inherited-label classifier",
        "em_mmil_prob": "Deterministic EM-MMIL",
        "mcem_mmil_prob": "MCEM-MMIL",
    }

    prob_long["model"] = prob_long["model"].map(model_map).fillna(prob_long["model"])

    print("\nBinary probability long format dimensions:")
    print(prob_long.shape)

    return prob_long


############################################################
# Binary evaluation: patient-level aggregation
############################################################

def aggregate_binary_patient_scores(
    prob_long: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "split",
        "Patient",
        "Stage_group",
        "y_patient",
        "model",
    ]

    for keys, g in prob_long.groupby(group_cols, dropna=False):
        cell_prob = g["cell_prob"]

        rows.append(
            dict(
                zip(group_cols, keys),
                n_cells=int(g.shape[0]),
                mean_prob=float(cell_prob.mean(skipna=True)),
                median_prob=float(cell_prob.median(skipna=True)),
                q90_prob=_safe_quantile(cell_prob, 0.90),
                q99_prob=_safe_quantile(cell_prob, 0.99),
                **{
                    "prop_gt_0.5": float((cell_prob > 0.5).mean()),
                    "prop_gt_0.9": float((cell_prob > 0.9).mean()),
                },
            )
        )

    patient_scores = pd.DataFrame(rows)

    print("\nBinary patient-level scores dimensions:")
    print(patient_scores.shape)

    print("\nFirst few binary patient-level scores:")
    print(patient_scores.head(10))

    return patient_scores


############################################################
# Binary evaluation: AUROC
############################################################

def compute_binary_patient_auc(
    patient_scores: pd.DataFrame,
) -> pd.DataFrame:
    aggregation_cols = get_aggregation_cols_binary()

    long_df = patient_scores.melt(
        id_vars=[
            "split",
            "Patient",
            "Stage_group",
            "y_patient",
            "model",
            "n_cells",
        ],
        value_vars=aggregation_cols,
        var_name="aggregation",
        value_name="patient_score",
    )

    rows = []

    for keys, g in long_df.groupby(["split", "model", "aggregation"], dropna=False):
        split, model, aggregation = keys

        rows.append(
            {
                "split": split,
                "model": model,
                "aggregation": aggregation,
                "n_patients": int(g.shape[0]),
                "n_early": int((g["y_patient"] == 0).sum()),
                "n_advanced": int((g["y_patient"] == 1).sum()),
                "auroc": compute_auc_safe(
                    g["y_patient"].to_numpy(),
                    g["patient_score"].to_numpy(),
                ),
            }
        )

    auc_results = pd.DataFrame(rows).sort_values(
        ["split", "model", "auroc"],
        ascending=[True, True, False],
        na_position="last",
    )

    print("\nBinary patient-level AUROC results:")
    print(auc_results.to_string(index=False))

    return auc_results


############################################################
# Binary evaluation: log loss
############################################################

def compute_binary_patient_logloss(
    patient_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, g in patient_scores.groupby(["split", "model"], dropna=False):
        split, model = keys

        rows.append(
            {
                "split": split,
                "model": model,
                "n_patients": int(g.shape[0]),
                "logloss_mean_prob": log_loss_binary(
                    g["y_patient"].to_numpy(),
                    g["mean_prob"].to_numpy(),
                ),
                "logloss_median_prob": log_loss_binary(
                    g["y_patient"].to_numpy(),
                    g["median_prob"].to_numpy(),
                ),
                "logloss_q90_prob": log_loss_binary(
                    g["y_patient"].to_numpy(),
                    g["q90_prob"].to_numpy(),
                ),
                "logloss_q99_prob": log_loss_binary(
                    g["y_patient"].to_numpy(),
                    g["q99_prob"].to_numpy(),
                ),
            }
        )

    patient_logloss = pd.DataFrame(rows)

    print("\nBinary patient-level log loss results:")
    print(patient_logloss)

    return patient_logloss


############################################################
# Binary evaluation: best test AUROC
############################################################

def get_binary_best_test_auc(
    auc_results: pd.DataFrame,
) -> pd.DataFrame:
    test_df = auc_results[auc_results["split"] == "test"].copy()

    best_test_auc = (
        test_df.sort_values(
            ["model", "auroc"],
            ascending=[True, False],
            na_position="last",
        )
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    print("\nBest binary test AUROC aggregation per model:")
    print(best_test_auc)

    return best_test_auc


############################################################
# Binary evaluation: cell-type enrichment
############################################################

def compute_binary_celltype_enrichment(
    prob_long: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "split",
        "model",
        "Stage_group",
        "Cell_type",
    ]

    for keys, g in prob_long.groupby(group_cols, dropna=False):
        cell_prob = g["cell_prob"]

        rows.append(
            dict(
                zip(group_cols, keys),
                n_cells=int(g.shape[0]),
                mean_prob=float(cell_prob.mean(skipna=True)),
                **{
                    "prop_gt_0.5": float((cell_prob > 0.5).mean()),
                    "prop_gt_0.9": float((cell_prob > 0.9).mean()),
                },
            )
        )

    celltype_enrichment = pd.DataFrame(rows).sort_values(
        ["split", "model", "Stage_group", "mean_prob"],
        ascending=[True, True, True, False],
    )

    print("\nBinary cell-type summary for biological interpretation:")
    print(celltype_enrichment.head(80).to_string(index=False))

    return celltype_enrichment


############################################################
# Binary plots
############################################################

def plot_binary_patient_scores(
    patient_scores: pd.DataFrame,
    out_dir: str | Path,
) -> str:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    aggregation_cols = get_aggregation_cols_binary()

    patient_scores_long = patient_scores.melt(
        id_vars=[
            "split",
            "Patient",
            "Stage_group",
            "y_patient",
            "model",
            "n_cells",
        ],
        value_vars=aggregation_cols,
        var_name="aggregation",
        value_name="patient_score",
    )

    plot_data = patient_scores_long[patient_scores_long["split"] == "test"].copy()

    plot_file = out_dir / "plot_patient_level_binary_MMIL_scores.png"

    combos = (
        plot_data[["model", "aggregation"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    n_panels = combos.shape[0]
    n_cols = len(aggregation_cols)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, max(4, 3 * n_rows)),
        squeeze=False,
    )

    for idx, row in combos.iterrows():
        ax = axes[idx // n_cols, idx % n_cols]

        sub = plot_data[
            (plot_data["model"] == row["model"])
            & (plot_data["aggregation"] == row["aggregation"])
        ]

        groups = [
            sub.loc[sub["Stage_group"] == stage, "patient_score"].dropna().to_numpy()
            for stage in ["Early", "Advanced"]
        ]

        ax.boxplot(groups, labels=["Early", "Advanced"])

        for pos, vals in enumerate(groups, start=1):
            if len(vals) > 0:
                jitter = np.random.default_rng(2026).normal(
                    loc=pos,
                    scale=0.04,
                    size=len(vals),
                )
                ax.scatter(jitter, vals, s=12, alpha=0.7)

        ax.set_title(f"{row['model']}\n{row['aggregation']}")
        ax.set_xlabel("Patient-level stage group")
        ax.set_ylabel("Aggregated predicted probability")
        ax.tick_params(axis="x", rotation=45)

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle("Binary patient-level aggregated scores on test patients")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()

    return str(plot_file)


def plot_binary_patient_auc(
    auc_results: pd.DataFrame,
    out_dir: str | Path,
) -> str:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    plot_file = out_dir / "plot_patient_level_binary_MMIL_AUROC.png"

    return _save_grouped_bar(
        df=auc_results[auc_results["split"] == "test"],
        x_col="aggregation",
        y_col="auroc",
        hue_col="model",
        title="Binary patient-level test AUROC by aggregation method",
        xlabel="Aggregation method",
        ylabel="Patient-level AUROC",
        output_file=plot_file,
        width=10,
        height=6,
    )


def plot_binary_celltype_enrichment(
    celltype_enrichment: pd.DataFrame,
    out_dir: str | Path,
) -> str:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    plot_file = out_dir / "plot_celltype_binary_MMIL_prop_gt_0.5.png"

    plot_data = celltype_enrichment[
        celltype_enrichment["split"] == "test"
    ].copy()

    if plot_data.empty:
        return str(plot_file)

    models = plot_data["model"].dropna().unique().tolist()

    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(12, max(4, 4 * len(models))),
        squeeze=False,
    )

    for i, model in enumerate(models):
        ax = axes[i, 0]

        sub = plot_data[plot_data["model"] == model].copy()

        pivot = sub.pivot_table(
            index="Cell_type",
            columns="Stage_group",
            values="prop_gt_0.5",
            aggfunc="mean",
        ).fillna(0)

        pivot = pivot.sort_values(
            by=pivot.columns.tolist()[0],
            ascending=True,
        )

        pivot.plot(
            kind="barh",
            ax=ax,
        )

        ax.set_title(model)
        ax.set_xlabel("Proportion of cells")
        ax.set_ylabel("Cell type")
        ax.legend(title="Stage group")

    fig.suptitle("Binary proportion of cells with predicted probability > 0.5")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()

    return str(plot_file)


############################################################
# Main binary evaluation function
############################################################

def evaluate_binary_predictions(
    pred_file: str | Path,
    out_dir: str | Path,
    save_plots: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    cell_pred = load_binary_prediction_file(pred_file)
    prob_long = prepare_binary_probability_long(cell_pred)
    patient_scores = aggregate_binary_patient_scores(prob_long)
    auc_results = compute_binary_patient_auc(patient_scores)
    patient_logloss = compute_binary_patient_logloss(patient_scores)
    best_test_auc = get_binary_best_test_auc(auc_results)
    celltype_enrichment = compute_binary_celltype_enrichment(prob_long)

    patient_scores_file = out_dir / "patient_level_binary_MMIL_aggregation_scores.csv"
    auc_results_file = out_dir / "patient_level_binary_MMIL_AUROC_results.csv"
    patient_logloss_file = out_dir / "patient_level_binary_MMIL_logloss_results.csv"
    best_test_auc_file = out_dir / "patient_level_binary_MMIL_best_test_AUROC.csv"
    celltype_enrichment_file = out_dir / "celltype_enrichment_binary_MMIL_probabilities.csv"

    patient_scores.to_csv(patient_scores_file, index=False)
    auc_results.to_csv(auc_results_file, index=False)
    patient_logloss.to_csv(patient_logloss_file, index=False)
    best_test_auc.to_csv(best_test_auc_file, index=False)
    celltype_enrichment.to_csv(celltype_enrichment_file, index=False)

    plot_files = {}

    if save_plots:
        plot_files["patient_scores"] = plot_binary_patient_scores(
            patient_scores=patient_scores,
            out_dir=out_dir,
        )

        plot_files["auc"] = plot_binary_patient_auc(
            auc_results=auc_results,
            out_dir=out_dir,
        )

        plot_files["celltype"] = plot_binary_celltype_enrichment(
            celltype_enrichment=celltype_enrichment,
            out_dir=out_dir,
        )

    print("\nBinary patient-level evaluation done.")

    return {
        "task_type": "binary",
        "cell_pred": cell_pred,
        "prob_long": prob_long,
        "patient_scores": patient_scores,
        "auc_results": auc_results,
        "patient_logloss": patient_logloss,
        "best_test_auc": best_test_auc,
        "celltype_enrichment": celltype_enrichment,
        "files": {
            "patient_scores_file": str(patient_scores_file),
            "auc_results_file": str(auc_results_file),
            "patient_logloss_file": str(patient_logloss_file),
            "best_test_auc_file": str(best_test_auc_file),
            "celltype_enrichment_file": str(celltype_enrichment_file),
            "plot_files": plot_files,
        },
    }


############################################################
# Categorical evaluation: load and check prediction file
############################################################

def get_categorical_prob_columns(
    class_levels: Sequence[str],
    prefix: str,
) -> list[str]:
    return [
        f"{prefix}_{sanitize_class_name_eval(cls)}"
        for cls in class_levels
    ]


def load_categorical_prediction_file(
    pred_file: str | Path,
    class_levels: Sequence[str],
) -> pd.DataFrame:
    pred_file = Path(pred_file)
    class_levels = list(class_levels)

    print("\nLoading categorical cell-level prediction file:")
    print(pred_file)

    cell_pred = pd.read_csv(pred_file)

    print("\nLoaded categorical prediction file dimensions:")
    print(cell_pred.shape)

    print("\nColumn names:")
    print(list(cell_pred.columns))

    naive_cols = get_categorical_prob_columns(class_levels, "naive_prob")
    em_cols = get_categorical_prob_columns(class_levels, "cat_em_prob")
    mcem_cols = get_categorical_prob_columns(class_levels, "cat_mcem_prob")

    required_cols = [
        "Patient",
        "split",
        "Stage_cat",
    ] + naive_cols + em_cols + mcem_cols

    check_required_cols(
        cell_pred,
        required_cols,
        object_name="categorical cell-level prediction file",
    )

    cell_pred = _ensure_cell_and_celltype(cell_pred)

    cell_pred["Stage_cat"] = pd.Categorical(
        cell_pred["Stage_cat"],
        categories=class_levels,
        ordered=True,
    )

    print("\nCell counts by split and Stage_cat:")
    print(pd.crosstab(cell_pred["split"], cell_pred["Stage_cat"]))

    print("\nPatient counts by split and Stage_cat:")
    print(
        cell_pred[["Patient", "split", "Stage_cat"]]
        .drop_duplicates()
        .groupby(["split", "Stage_cat"], observed=False)
        .size()
        .reset_index(name="n_patients")
    )

    return cell_pred


############################################################
# Categorical evaluation: prepare long probability table
############################################################

def prepare_categorical_probability_long(
    cell_pred: pd.DataFrame,
    class_levels: Sequence[str],
) -> pd.DataFrame:
    class_levels = list(class_levels)

    model_specs = [
        {
            "model": "Naive inherited-label classifier",
            "prefix": "naive_prob",
        },
        {
            "model": "Categorical EM-MMIL",
            "prefix": "cat_em_prob",
        },
        {
            "model": "Categorical MCEM-MMIL",
            "prefix": "cat_mcem_prob",
        },
    ]

    long_list = []

    for spec in model_specs:
        prob_cols = get_categorical_prob_columns(
            class_levels=class_levels,
            prefix=spec["prefix"],
        )

        tmp = cell_pred[
            [
                "Cell",
                "Patient",
                "split",
                "Stage_cat",
                "Cell_type",
            ] + prob_cols
        ].copy()

        rename_map = {
            old: new
            for old, new in zip(prob_cols, class_levels)
        }

        tmp = tmp.rename(columns=rename_map)

        tmp_long = tmp.melt(
            id_vars=[
                "Cell",
                "Patient",
                "split",
                "Stage_cat",
                "Cell_type",
            ],
            value_vars=class_levels,
            var_name="class",
            value_name="cell_prob",
        )

        tmp_long["model"] = spec["model"]
        tmp_long["y_true"] = tmp_long["Stage_cat"].astype(str)

        long_list.append(tmp_long)

    prob_long = pd.concat(long_list, axis=0, ignore_index=True)

    prob_long["class"] = pd.Categorical(
        prob_long["class"],
        categories=class_levels,
        ordered=True,
    )

    prob_long["Stage_cat"] = pd.Categorical(
        prob_long["Stage_cat"],
        categories=class_levels,
        ordered=True,
    )

    print("\nCategorical probability long format dimensions:")
    print(prob_long.shape)

    return prob_long


############################################################
# Categorical evaluation: patient-level aggregation
############################################################

def aggregate_categorical_patient_scores(
    prob_long: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "split",
        "Patient",
        "Stage_cat",
        "y_true",
        "model",
        "class",
    ]

    for keys, g in prob_long.groupby(group_cols, dropna=False, observed=False):
        cell_prob = g["cell_prob"]

        rows.append(
            dict(
                zip(group_cols, keys),
                n_cells=int(g.shape[0]),
                mean_prob=float(cell_prob.mean(skipna=True)),
                median_prob=float(cell_prob.median(skipna=True)),
                q90_prob=_safe_quantile(cell_prob, 0.90),
                q99_prob=_safe_quantile(cell_prob, 0.99),
            )
        )

    patient_scores = pd.DataFrame(rows)

    print("\nCategorical patient-level scores dimensions:")
    print(patient_scores.shape)

    print("\nFirst few categorical patient-level scores:")
    print(patient_scores.head(10))

    return patient_scores


############################################################
# Categorical evaluation: convert patient scores to wide format
############################################################

def categorical_patient_scores_wide(
    patient_scores: pd.DataFrame,
    class_levels: Sequence[str],
    aggregation: str,
) -> pd.DataFrame:
    class_levels = list(class_levels)

    wide = patient_scores[
        [
            "split",
            "Patient",
            "Stage_cat",
            "y_true",
            "model",
            "class",
            aggregation,
        ]
    ].rename(columns={aggregation: "patient_score"})

    wide = wide.pivot_table(
        index=[
            "split",
            "Patient",
            "Stage_cat",
            "y_true",
            "model",
        ],
        columns="class",
        values="patient_score",
        aggfunc="first",
        observed=False,
    ).reset_index()

    wide.columns.name = None

    for cls in class_levels:
        if cls not in wide.columns:
            wide[cls] = 0.0

    wide = wide[
        [
            "split",
            "Patient",
            "Stage_cat",
            "y_true",
            "model",
        ] + class_levels
    ].copy()

    prob_df = row_normalize(wide.loc[:, class_levels])
    wide.loc[:, class_levels] = prob_df.to_numpy(dtype=float)

    return wide


############################################################
# Categorical evaluation: metrics
############################################################

def compute_categorical_patient_metrics(
    patient_scores: pd.DataFrame,
    class_levels: Sequence[str],
) -> pd.DataFrame:
    class_levels = list(class_levels)
    aggregation_cols = get_aggregation_cols_categorical()

    metric_rows = []

    for aggregation in aggregation_cols:
        wide = categorical_patient_scores_wide(
            patient_scores=patient_scores,
            class_levels=class_levels,
            aggregation=aggregation,
        )

        for keys, g in wide.groupby(["split", "model"], dropna=False):
            split, model = keys

            prob_mat = row_normalize(g.loc[:, class_levels])
            y_true = g["Stage_cat"].astype(str).to_numpy()

            prob_arr = prob_mat.loc[:, class_levels].to_numpy(dtype=float)
            y_pred = np.array(class_levels, dtype=object)[
                np.argmax(prob_arr, axis=1)
            ]

            metric_rows.append(
                {
                    "split": split,
                    "model": model,
                    "n_patients": int(g.shape[0]),
                    "multiclass_logloss": multiclass_log_loss(
                        y_true=y_true,
                        prob_mat=prob_mat,
                        class_levels=class_levels,
                    ),
                    "accuracy": float(np.mean(y_pred == y_true)),
                    "balanced_accuracy": balanced_accuracy_multiclass(
                        y_true=y_true,
                        y_pred=y_pred,
                        class_levels=class_levels,
                    ),
                    "macro_f1": macro_f1_score(
                        y_true=y_true,
                        y_pred=y_pred,
                        class_levels=class_levels,
                    ),
                    "aggregation": aggregation,
                }
            )

    metrics_results = pd.DataFrame(metric_rows).sort_values(
        ["split", "model", "multiclass_logloss"],
        ascending=[True, True, True],
        na_position="last",
    )

    print("\nCategorical patient-level metrics:")
    print(metrics_results.to_string(index=False))

    return metrics_results


############################################################
# Categorical evaluation: best test metrics
############################################################

def get_categorical_best_test_metrics(
    metrics_results: pd.DataFrame,
) -> pd.DataFrame:
    test_df = metrics_results[metrics_results["split"] == "test"].copy()

    best_test_metrics = (
        test_df.sort_values(
            ["model", "accuracy", "multiclass_logloss"],
            ascending=[True, False, True],
            na_position="last",
        )
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    print("\nBest categorical test metrics per model:")
    print(best_test_metrics)

    return best_test_metrics


############################################################
# Categorical evaluation: confusion matrix
############################################################

def compute_categorical_confusion(
    patient_scores: pd.DataFrame,
    class_levels: Sequence[str],
) -> pd.DataFrame:
    class_levels = list(class_levels)
    aggregation_cols = get_aggregation_cols_categorical()

    confusion_list = []

    for aggregation in aggregation_cols:
        wide = categorical_patient_scores_wide(
            patient_scores=patient_scores,
            class_levels=class_levels,
            aggregation=aggregation,
        )

        prob_mat = row_normalize(wide.loc[:, class_levels])
        prob_arr = prob_mat.loc[:, class_levels].to_numpy(dtype=float)

        wide = wide.copy()
        wide["predicted_class"] = np.array(class_levels, dtype=object)[
            np.argmax(prob_arr, axis=1)
        ]
        wide["true_class"] = wide["Stage_cat"].astype(str)
        wide["aggregation"] = aggregation

        tmp = (
            wide.groupby(
                [
                    "split",
                    "model",
                    "aggregation",
                    "true_class",
                    "predicted_class",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="n_patients")
        )

        confusion_list.append(tmp)

    confusion_results = pd.concat(
        confusion_list,
        axis=0,
        ignore_index=True,
    )

    print("\nCategorical confusion matrix rows:")
    print(confusion_results.to_string(index=False))

    return confusion_results


############################################################
# Categorical evaluation: cell-type enrichment
############################################################

def compute_categorical_celltype_enrichment(
    prob_long: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "split",
        "model",
        "Stage_cat",
        "Cell_type",
        "class",
    ]

    for keys, g in prob_long.groupby(group_cols, dropna=False, observed=False):
        cell_prob = g["cell_prob"]

        rows.append(
            dict(
                zip(group_cols, keys),
                n_cells=int(g.shape[0]),
                mean_prob=float(cell_prob.mean(skipna=True)),
                q90_prob=_safe_quantile(cell_prob, 0.90),
            )
        )

    celltype_enrichment = pd.DataFrame(rows).sort_values(
        ["split", "model", "Stage_cat", "class", "mean_prob"],
        ascending=[True, True, True, True, False],
        na_position="last",
    )

    print("\nCategorical cell-type summary for biological interpretation:")
    print(celltype_enrichment.head(100).to_string(index=False))

    return celltype_enrichment


############################################################
# Categorical plots
############################################################

def plot_categorical_metrics(
    metrics_results: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    test_df = metrics_results[metrics_results["split"] == "test"].copy()

    acc_file = out_dir / "plot_patient_level_categorical_accuracy.png"
    logloss_file = out_dir / "plot_patient_level_categorical_logloss.png"

    _save_grouped_bar(
        df=test_df,
        x_col="aggregation",
        y_col="accuracy",
        hue_col="model",
        title="Categorical patient-level test accuracy",
        xlabel="Aggregation method",
        ylabel="Accuracy",
        output_file=acc_file,
        width=10,
        height=6,
    )

    _save_grouped_bar(
        df=test_df,
        x_col="aggregation",
        y_col="multiclass_logloss",
        hue_col="model",
        title="Categorical patient-level test multiclass log loss",
        xlabel="Aggregation method",
        ylabel="Multiclass log loss",
        output_file=logloss_file,
        width=10,
        height=6,
    )

    return {
        "accuracy": str(acc_file),
        "logloss": str(logloss_file),
    }


############################################################
# Main categorical evaluation function
############################################################

def evaluate_categorical_predictions(
    pred_file: str | Path,
    out_dir: str | Path,
    class_levels: Sequence[str],
    save_plots: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    class_levels = list(class_levels)

    cell_pred = load_categorical_prediction_file(
        pred_file=pred_file,
        class_levels=class_levels,
    )

    prob_long = prepare_categorical_probability_long(
        cell_pred=cell_pred,
        class_levels=class_levels,
    )

    patient_scores = aggregate_categorical_patient_scores(prob_long)

    metrics_results = compute_categorical_patient_metrics(
        patient_scores=patient_scores,
        class_levels=class_levels,
    )

    best_test_metrics = get_categorical_best_test_metrics(metrics_results)

    confusion_results = compute_categorical_confusion(
        patient_scores=patient_scores,
        class_levels=class_levels,
    )

    celltype_enrichment = compute_categorical_celltype_enrichment(prob_long)

    patient_scores_file = out_dir / "patient_level_categorical_MMIL_aggregation_scores.csv"
    metrics_results_file = out_dir / "patient_level_categorical_MMIL_metrics.csv"
    best_test_metrics_file = out_dir / "patient_level_categorical_MMIL_best_test_metrics.csv"
    confusion_results_file = out_dir / "patient_level_categorical_MMIL_confusion_matrix.csv"
    celltype_enrichment_file = out_dir / "celltype_enrichment_categorical_MMIL_probabilities.csv"

    patient_scores.to_csv(patient_scores_file, index=False)
    metrics_results.to_csv(metrics_results_file, index=False)
    best_test_metrics.to_csv(best_test_metrics_file, index=False)
    confusion_results.to_csv(confusion_results_file, index=False)
    celltype_enrichment.to_csv(celltype_enrichment_file, index=False)

    plot_files = {}

    if save_plots:
        plot_files["metrics"] = plot_categorical_metrics(
            metrics_results=metrics_results,
            out_dir=out_dir,
        )

    print("\nCategorical patient-level evaluation done.")

    return {
        "task_type": "categorical",
        "class_levels": class_levels,
        "cell_pred": cell_pred,
        "prob_long": prob_long,
        "patient_scores": patient_scores,
        "metrics_results": metrics_results,
        "best_test_metrics": best_test_metrics,
        "confusion_results": confusion_results,
        "celltype_enrichment": celltype_enrichment,
        "files": {
            "patient_scores_file": str(patient_scores_file),
            "metrics_results_file": str(metrics_results_file),
            "best_test_metrics_file": str(best_test_metrics_file),
            "confusion_results_file": str(confusion_results_file),
            "celltype_enrichment_file": str(celltype_enrichment_file),
            "plot_files": plot_files,
        },
    }


############################################################
# Unified evaluation dispatcher
############################################################

def evaluate_predictions(
    pred_file: str | Path,
    out_dir: str | Path,
    task_type: str = "binary",
    class_levels: Sequence[str] | None = None,
    save_plots: bool = True,
) -> dict[str, Any]:
    task_type = normalize_task_type(task_type)

    if task_type == "binary":
        return evaluate_binary_predictions(
            pred_file=pred_file,
            out_dir=out_dir,
            save_plots=save_plots,
        )

    if task_type == "categorical":
        if class_levels is None:
            raise ValueError("class_levels must be provided for categorical evaluation.")

        return evaluate_categorical_predictions(
            pred_file=pred_file,
            out_dir=out_dir,
            class_levels=class_levels,
            save_plots=save_plots,
        )

    raise ValueError("Unsupported task_type.")