############################################################
# visualize_ablation.py
# Standalone visualization of MMIL / MCEM-MMIL ablation results.
#
# Python version of visualize_ablation.R
#
# Usage:
#   python visualize_ablation.py \
#     "<.../Output/summary/all_experiment_metrics.csv>" \
#     [output_dir]
#
# First arg : path to all_experiment_metrics.csv
# Second arg: output directory.
#             If omitted, defaults to "<csv_dir>/../viz".
############################################################

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


############################################################
# Constants
############################################################

MODEL_ORDER = [
    "Naive inherited-label classifier",
    "Deterministic EM-MMIL",
    "MCEM-MMIL",
]

MODEL_SHORT = {
    "Naive inherited-label classifier": "Naive",
    "Deterministic EM-MMIL": "EM-MMIL",
    "MCEM-MMIL": "MCEM-MMIL",
    "Categorical EM-MMIL": "EM-MMIL",
    "Categorical MCEM-MMIL": "MCEM-MMIL",
}

MODEL_COLORS = {
    "Naive": "#888888",
    "EM-MMIL": "#4C72B0",
    "MCEM-MMIL": "#C44E52",
}

FEATURE_COLORS = {
    "PCA": "#1F77B4",
    "ICA": "#2CA02C",
    "NMF": "#FF7F0E",
}

RANK_COLORS = {
    "1st": "#2CA02C",
    "2nd": "#FFBB33",
    "3rd": "#D62728",
}

MODEL_SHORT_ORDER = ["Naive", "EM-MMIL", "MCEM-MMIL"]


############################################################
# Helpers
############################################################

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def shorten_model(model_name: Any) -> str:
    return MODEL_SHORT.get(str(model_name), str(model_name))


def make_model_label(model_backend: Any, glmnet_alpha: Any) -> str:
    model_backend = str(model_backend)

    if model_backend != "glmnet":
        return model_backend

    try:
        alpha = float(glmnet_alpha)
    except Exception:
        return "glmnet"

    if not np.isfinite(alpha):
        return "glmnet"

    if abs(alpha - 0) < 1e-9:
        return "glmnet (ridge)"

    if abs(alpha - 0.5) < 1e-9:
        return "glmnet (elasticnet)"

    if abs(alpha - 1) < 1e-9:
        return "glmnet (lasso)"

    return f"glmnet (alpha={alpha:.2f})"


def validate_input_df(df: pd.DataFrame) -> None:
    required_cols = [
        "experiment_id",
        "task_type",
        "feature_method",
        "n_components",
        "model_backend",
        "glmnet_alpha",
        "model",
        "aggregation",
        "auroc",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "multiclass_logloss",
        "metric_value",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Input CSV missing required columns: "
            + ", ".join(missing_cols)
        )


def prepare_task_metric(
    df_task: pd.DataFrame,
    task_type: str,
) -> pd.DataFrame:
    df_task = df_task.copy()

    if task_type == "binary":
        df_task["primary_metric"] = df_task["auroc"]
        df_task["primary_label"] = "Patient-level test AUROC"
    else:
        df_task["primary_metric"] = df_task["balanced_accuracy"]
        df_task["primary_label"] = "Patient-level test balanced accuracy"

    df_task["model_short"] = df_task["model"].map(shorten_model)
    df_task["model_short"] = pd.Categorical(
        df_task["model_short"],
        categories=MODEL_SHORT_ORDER,
        ordered=True,
    )

    df_task["model_label"] = [
        make_model_label(backend, alpha)
        for backend, alpha in zip(
            df_task["model_backend"],
            df_task["glmnet_alpha"],
        )
    ]

    df_task["n_components"] = df_task["n_components"].astype(int)

    return df_task


def collapse_best_aggregation(df_task: pd.DataFrame) -> pd.DataFrame:
    df = df_task.copy()
    df = df.sort_values(
        ["experiment_id", "model_short", "primary_metric"],
        ascending=[True, True, False],
        na_position="last",
    )

    df_best = (
        df.groupby(["experiment_id", "model_short"], observed=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    return df_best


def build_config_wide(df_task: pd.DataFrame) -> pd.DataFrame:
    df_best = collapse_best_aggregation(df_task)

    config_meta = (
        df_best[
            [
                "experiment_id",
                "feature_method",
                "n_components",
                "model_backend",
                "glmnet_alpha",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    config_meta["model_label"] = [
        make_model_label(backend, alpha)
        for backend, alpha in zip(
            config_meta["model_backend"],
            config_meta["glmnet_alpha"],
        )
    ]

    scores_wide = df_best.pivot_table(
        index="experiment_id",
        columns="model_short",
        values="primary_metric",
        aggfunc="first",
        observed=False,
    ).reset_index()

    scores_wide.columns.name = None

    wide = config_meta.merge(
        scores_wide,
        on="experiment_id",
        how="left",
    )

    return wide


def _metric_label(df_task: pd.DataFrame) -> str:
    labels = df_task["primary_label"].dropna().unique()
    return str(labels[0]) if len(labels) else "Primary metric"


def _savefig(path: str | Path, width: float, height: float, dpi: int = 200) -> None:
    plt.gcf().set_size_inches(width, height)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


############################################################
# Plot 1 — Main comparison
############################################################

def plot_main_comparison(
    df_task: pd.DataFrame,
    task_type: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    df_best = collapse_best_aggregation(df_task)
    metric_label = _metric_label(df_task)

    summary_stats = (
        df_best.groupby("model_short", observed=False)["primary_metric"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )

    fig, ax = plt.subplots()

    data = []
    labels = []

    for model in MODEL_SHORT_ORDER:
        vals = (
            df_best.loc[df_best["model_short"] == model, "primary_metric"]
            .dropna()
            .to_numpy()
        )
        data.append(vals)
        labels.append(model)

    box = ax.boxplot(
        data,
        labels=labels,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
    )

    for patch, model in zip(box["boxes"], MODEL_SHORT_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.45)

    rng = np.random.default_rng(2026)

    for i, model in enumerate(MODEL_SHORT_ORDER, start=1):
        vals = (
            df_best.loc[df_best["model_short"] == model, "primary_metric"]
            .dropna()
            .to_numpy()
        )

        if len(vals) == 0:
            continue

        jitter = rng.normal(i, 0.06, size=len(vals))

        ax.scatter(
            jitter,
            vals,
            s=18,
            alpha=0.75,
            color=MODEL_COLORS[model],
        )

        med_row = summary_stats[summary_stats["model_short"].astype(str) == model]

        if not med_row.empty and np.isfinite(med_row["median"].iloc[0]):
            median_val = med_row["median"].iloc[0]
            ax.text(
                i,
                median_val,
                f"median = {median_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(
        f"Main comparison ({task_type}): Naive vs MMIL methods",
        fontweight="bold",
    )

    ax.set_xlabel("")
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", alpha=0.25)

    out_file = output_dir / f"01_main_comparison_{task_type}.png"
    _savefig(out_file, width=7, height=5.5)

    print(f"  Saved: {out_file}")


############################################################
# Plot 2 — Feature method x dimensionality
############################################################

def plot_feature_method(
    df_task: pd.DataFrame,
    task_type: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)

    df_best = collapse_best_aggregation(df_task)
    df_best = df_best[df_best["model_short"].astype(str) == "MCEM-MMIL"].copy()

    if df_best.empty:
        print(f"  Skipping plot 2 for {task_type} (no MCEM-MMIL rows).")
        return

    metric_label = _metric_label(df_task)

    df_agg = (
        df_best.groupby(["feature_method", "n_components"], as_index=False)
        ["primary_metric"]
        .max()
        .rename(columns={"primary_metric": "best_metric"})
    )

    n_components_values = sorted(df_agg["n_components"].unique())
    feature_methods = [
        fm
        for fm in ["PCA", "ICA", "NMF"]
        if fm in set(df_agg["feature_method"])
    ]

    x = np.arange(len(n_components_values))
    width = 0.8 / max(len(feature_methods), 1)

    fig, ax = plt.subplots()

    for idx, feature_method in enumerate(feature_methods):
        vals = []

        for n_comp in n_components_values:
            row = df_agg[
                (df_agg["feature_method"] == feature_method)
                & (df_agg["n_components"] == n_comp)
            ]

            vals.append(
                row["best_metric"].iloc[0]
                if not row.empty
                else np.nan
            )

        positions = x - 0.4 + width / 2 + idx * width

        ax.bar(
            positions,
            vals,
            width=width,
            label=feature_method,
            color=FEATURE_COLORS.get(feature_method),
        )

        for px, val in zip(positions, vals):
            if np.isfinite(val):
                ax.text(
                    px,
                    val,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in n_components_values])
    ax.set_xlabel("Number of components")
    ax.set_ylabel(metric_label)
    ax.set_title(
        f"Feature method comparison ({task_type}, MCEM-MMIL)",
        fontweight="bold",
    )
    ax.legend(title="Feature method", loc="best")
    ax.grid(axis="y", alpha=0.25)

    out_file = output_dir / f"02_feature_method_{task_type}.png"
    _savefig(out_file, width=8, height=5.5)

    print(f"  Saved: {out_file}")


############################################################
# Plot 3 — Model backend comparison
############################################################

def plot_model_backend(
    df_task: pd.DataFrame,
    task_type: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)

    df_best = collapse_best_aggregation(df_task)
    df_best = df_best[df_best["model_short"].astype(str) == "MCEM-MMIL"].copy()

    if df_best.empty:
        print(f"  Skipping plot 3 for {task_type} (no MCEM-MMIL rows).")
        return

    metric_label = _metric_label(df_task)

    df_agg = (
        df_best.groupby(["feature_method", "model_label"], as_index=False)
        ["primary_metric"]
        .max()
        .rename(columns={"primary_metric": "best_metric"})
    )

    backend_order = (
        df_agg.groupby("model_label")["best_metric"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    feature_methods = [
        fm
        for fm in ["PCA", "ICA", "NMF"]
        if fm in set(df_agg["feature_method"])
    ]

    fig, axes = plt.subplots(
        1,
        len(feature_methods),
        squeeze=False,
        figsize=(9, 5.5),
    )

    for ax, feature_method in zip(axes[0], feature_methods):
        sub = df_agg[df_agg["feature_method"] == feature_method].copy()
        sub["model_label"] = pd.Categorical(
            sub["model_label"],
            categories=backend_order,
            ordered=True,
        )
        sub = sub.sort_values("model_label")

        labels = sub["model_label"].astype(str).tolist()
        vals = sub["best_metric"].to_numpy(dtype=float)

        ax.bar(
            np.arange(len(labels)),
            vals,
            width=0.65,
        )

        for i, val in enumerate(vals):
            if np.isfinite(val):
                ax.text(
                    i,
                    val,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(feature_method, fontweight="bold")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Model backend comparison ({task_type}, MCEM-MMIL)",
        fontweight="bold",
    )

    out_file = output_dir / f"03_model_backend_{task_type}.png"
    _savefig(out_file, width=9, height=5.5)

    print(f"  Saved: {out_file}")


############################################################
# Plot 4 — Dimensionality trend
############################################################

def plot_dimensionality(
    df_task: pd.DataFrame,
    task_type: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)

    df_best = collapse_best_aggregation(df_task)
    df_best = df_best[df_best["model_short"].astype(str) == "MCEM-MMIL"].copy()

    if df_best.empty:
        print(f"  Skipping plot 4 for {task_type} (no MCEM-MMIL rows).")
        return

    metric_label = _metric_label(df_task)

    df_agg = (
        df_best.groupby(["feature_method", "n_components"], as_index=False)
        .agg(
            mean_metric=("primary_metric", "mean"),
            min_metric=("primary_metric", "min"),
            max_metric=("primary_metric", "max"),
        )
    )

    fig, ax = plt.subplots()

    for feature_method in ["PCA", "ICA", "NMF"]:
        sub = df_agg[df_agg["feature_method"] == feature_method].copy()

        if sub.empty:
            continue

        sub = sub.sort_values("n_components")

        x = sub["n_components"].to_numpy(dtype=float)
        y = sub["mean_metric"].to_numpy(dtype=float)
        y_min = sub["min_metric"].to_numpy(dtype=float)
        y_max = sub["max_metric"].to_numpy(dtype=float)

        color = FEATURE_COLORS.get(feature_method)

        ax.fill_between(
            x,
            y_min,
            y_max,
            color=color,
            alpha=0.15,
        )

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            color=color,
            label=feature_method,
        )

    ax.set_xticks(sorted(df_agg["n_components"].unique()))
    ax.set_xlabel("Number of components")
    ax.set_ylabel(metric_label)
    ax.set_title(
        f"Dimensionality trend ({task_type}, MCEM-MMIL)",
        fontweight="bold",
    )
    ax.legend(title="Feature method")
    ax.grid(alpha=0.25)

    out_file = output_dir / f"04_dimensionality_{task_type}.png"
    _savefig(out_file, width=8, height=5.5)

    print(f"  Saved: {out_file}")


############################################################
# Plot 6 — Win counts
############################################################

def _rank_desc_ties_min(values: list[float]) -> list[float]:
    """
    Equivalent to R rank(-values, ties.method = "min").
    Highest value gets rank 1.
    Ties get the minimum rank.
    """
    s = pd.Series(values, dtype=float)
    return s.rank(method="min", ascending=False).tolist()


def plot_win_counts(
    df_task: pd.DataFrame,
    task_type: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)

    wide = build_config_wide(df_task)

    needed = ["Naive", "EM-MMIL", "MCEM-MMIL"]

    if not all(col in wide.columns for col in needed):
        print(
            f"  Skipping plot 6 for {task_type} "
            "(missing one of Naive/EM/MCEM columns)."
        )
        return

    rank_rows = []

    for _, row in wide.iterrows():
        vals = [
            row["Naive"],
            row["EM-MMIL"],
            row["MCEM-MMIL"],
        ]

        ranks = _rank_desc_ties_min(vals)

        rank_rows.extend(
            [
                {
                    "experiment_id": row["experiment_id"],
                    "model_short": "Naive",
                    "rank_pos": ranks[0],
                },
                {
                    "experiment_id": row["experiment_id"],
                    "model_short": "EM-MMIL",
                    "rank_pos": ranks[1],
                },
                {
                    "experiment_id": row["experiment_id"],
                    "model_short": "MCEM-MMIL",
                    "rank_pos": ranks[2],
                },
            ]
        )

    ranks_long = pd.DataFrame(rank_rows)

    ranks_long["rank_label"] = ranks_long["rank_pos"].map(
        {
            1.0: "1st",
            2.0: "2nd",
            3.0: "3rd",
        }
    ).fillna(ranks_long["rank_pos"].astype(str))

    win_counts = (
        ranks_long.groupby(["model_short", "rank_label"])
        .size()
        .reset_index(name="n_configs")
    )

    total_configs = ranks_long["experiment_id"].nunique()

    win_counts_wide = (
        win_counts.pivot_table(
            index="model_short",
            columns="rank_label",
            values="n_configs",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )

    for col in ["1st", "2nd", "3rd"]:
        if col not in win_counts_wide.columns:
            win_counts_wide[col] = 0

    win_counts_wide["total_configs"] = total_configs
    win_counts_wide["pct_first"] = (
        100 * win_counts_wide["1st"] / total_configs
    ).round(1)

    csv_file = output_dir / f"06_win_counts_{task_type}.csv"
    win_counts_wide.to_csv(csv_file, index=False)

    print(f"  Saved: {csv_file}")

    fig, ax = plt.subplots()

    bottom = np.zeros(len(MODEL_SHORT_ORDER), dtype=float)
    x = np.arange(len(MODEL_SHORT_ORDER))

    for rank_label in ["1st", "2nd", "3rd"]:
        vals = []

        for model in MODEL_SHORT_ORDER:
            row = win_counts[
                (win_counts["model_short"] == model)
                & (win_counts["rank_label"] == rank_label)
            ]

            vals.append(
                int(row["n_configs"].iloc[0])
                if not row.empty
                else 0
            )

        vals = np.array(vals, dtype=float)

        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.6,
            color=RANK_COLORS[rank_label],
            label=rank_label,
            edgecolor="white",
        )

        for i, val in enumerate(vals):
            if val > 0:
                ax.text(
                    i,
                    bottom[i] + val / 2,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=10,
                )

        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT_ORDER)
    ax.set_ylabel("Number of configurations")
    ax.set_title(
        f"Win counts across configurations ({task_type})",
        fontweight="bold",
    )
    ax.legend(title="Rank within config", loc="best")
    ax.grid(axis="y", alpha=0.25)

    out_file = output_dir / f"06_win_counts_{task_type}.png"
    _savefig(out_file, width=7, height=5.5)

    print(f"  Saved: {out_file}")


############################################################
# Plot 7 — Paired differences
############################################################

def plot_paired_differences(
    df_task: pd.DataFrame,
    task_type: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)

    wide = build_config_wide(df_task)

    needed = ["Naive", "EM-MMIL", "MCEM-MMIL"]

    if not all(col in wide.columns for col in needed):
        print(
            f"  Skipping plot 7 for {task_type} "
            "(missing one of Naive/EM/MCEM columns)."
        )
        return

    metric_label = _metric_label(df_task)

    diff_long = pd.DataFrame(
        {
            "experiment_id": np.concatenate(
                [
                    wide["experiment_id"].to_numpy(),
                    wide["experiment_id"].to_numpy(),
                ]
            ),
            "comparison": (
                ["MCEM - Naive"] * wide.shape[0]
                + ["MCEM - EM-MMIL"] * wide.shape[0]
            ),
            "delta": np.concatenate(
                [
                    (
                        wide["MCEM-MMIL"]
                        - wide["Naive"]
                    ).to_numpy(dtype=float),
                    (
                        wide["MCEM-MMIL"]
                        - wide["EM-MMIL"]
                    ).to_numpy(dtype=float),
                ]
            ),
        }
    )

    diff_long = diff_long.dropna(subset=["delta"]).copy()

    diff_stats = (
        diff_long.groupby("comparison")
        .agg(
            n_configs=("delta", "size"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            pct_positive=("delta", lambda x: 100 * np.mean(x > 0)),
        )
        .reset_index()
    )

    csv_out = diff_long.merge(
        diff_stats[
            [
                "comparison",
                "mean_delta",
                "median_delta",
                "pct_positive",
            ]
        ],
        on="comparison",
        how="left",
    )

    csv_file = output_dir / f"07_paired_differences_{task_type}.csv"
    csv_out.to_csv(csv_file, index=False)

    print(f"  Saved: {csv_file}")

    comparisons = ["MCEM - Naive", "MCEM - EM-MMIL"]

    fig, axes = plt.subplots(
        len(comparisons),
        1,
        figsize=(8, 6),
        squeeze=False,
    )

    for ax, comparison in zip(axes[:, 0], comparisons):
        sub = diff_long[diff_long["comparison"] == comparison].copy()

        if sub.empty:
            ax.set_title(comparison)
            continue

        vals = sub["delta"].to_numpy(dtype=float)

        counts, bins, _ = ax.hist(
            vals,
            bins=20,
            color="#4C72B0",
            edgecolor="white",
            alpha=0.85,
        )

        ax.axvline(
            0,
            linestyle="--",
            color="black",
        )

        stat_row = diff_stats[diff_stats["comparison"] == comparison].iloc[0]

        ax.axvline(
            stat_row["mean_delta"],
            color="#C44E52",
            linewidth=1.0,
        )

        stats_label = (
            f"n = {int(stat_row['n_configs'])}\n"
            f"mean delta = {stat_row['mean_delta']:+.4f}\n"
            f"median delta = {stat_row['median_delta']:+.4f}\n"
            f"fraction > 0 = {stat_row['pct_positive']:.0f}%"
        )

        x_pos = np.nanmax(vals)
        y_pos = np.nanmax(counts) * 0.95 if len(counts) else 1

        ax.text(
            x_pos,
            y_pos,
            stats_label,
            ha="right",
            va="top",
            fontsize=9,
        )

        ax.set_title(comparison, fontweight="bold")
        ax.set_ylabel("Number of configurations")
        ax.grid(axis="y", alpha=0.25)

    axes[-1, 0].set_xlabel(
        f"Difference in {metric_label.lower()}"
    )

    fig.suptitle(
        f"Paired differences ({task_type}): MCEM-MMIL vs alternatives",
        fontweight="bold",
    )

    out_file = output_dir / f"07_paired_differences_{task_type}.png"
    _savefig(out_file, width=8, height=6)

    print(f"  Saved: {out_file}")

    print(f"\n  Paired-difference summary for {task_type}:")
    print(diff_stats.to_string(index=False))


############################################################
# Appendix CSV — top-10 configurations
############################################################

def write_appendix_top_configs(
    df_task: pd.DataFrame,
    task_type: str,
    output_dir: str | Path,
    top_n: int = 10,
) -> None:
    output_dir = Path(output_dir)

    df_best = collapse_best_aggregation(df_task)

    cols_to_show = [
        "experiment_id",
        "feature_method",
        "n_components",
        "model_backend",
        "glmnet_alpha",
        "model_short",
        "aggregation",
        "primary_metric",
    ]

    if task_type == "binary":
        extras = ["auroc"]
    else:
        extras = [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "multiclass_logloss",
        ]

    cols_to_show = list(
        dict.fromkeys(
            cols_to_show
            + [
                col
                for col in extras
                if col in df_best.columns
            ]
        )
    )

    out = (
        df_best.sort_values(
            "primary_metric",
            ascending=False,
            na_position="last",
        )
        .head(top_n)
        .loc[:, cols_to_show]
        .copy()
    )

    out.insert(0, "rank", np.arange(1, out.shape[0] + 1))

    out_file = output_dir / f"99_appendix_top_configs_{task_type}.csv"
    out.to_csv(out_file, index=False)

    print(f"  Saved: {out_file}")


############################################################
# Driver
############################################################

def run_visualization_for_task(
    df: pd.DataFrame,
    task_type: str,
    output_dir: str | Path,
) -> None:
    df_task = df[df["task_type"] == task_type].copy()

    if df_task.empty:
        print(f"\nNo rows for task = {task_type}, skipping.")
        return

    print("\n----------------------------------------")
    print(f"Task: {task_type} ({df_task.shape[0]} rows)")
    print("----------------------------------------")

    df_task = prepare_task_metric(df_task, task_type)

    models_present = (
        df_task["model_short"]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )

    print("Models present in data:", ", ".join(models_present))

    plot_main_comparison(df_task, task_type, output_dir)
    plot_feature_method(df_task, task_type, output_dir)
    plot_model_backend(df_task, task_type, output_dir)
    plot_dimensionality(df_task, task_type, output_dir)
    plot_win_counts(df_task, task_type, output_dir)
    plot_paired_differences(df_task, task_type, output_dir)
    write_appendix_top_configs(df_task, task_type, output_dir, top_n=10)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize MMIL / MCEM-MMIL ablation results."
    )

    parser.add_argument(
        "input_csv",
        help="Path to Output/summary/all_experiment_metrics.csv",
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help=(
            "Output visualization directory. "
            "Defaults to <csv_dir>/../viz."
        ),
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if args.output_dir is None:
        output_dir = input_csv.parent / ".." / "viz"
    else:
        output_dir = Path(args.output_dir)

    output_dir = ensure_dir(output_dir)

    print(f"Input CSV:   {input_csv.resolve()}")
    print(f"Output dir:  {output_dir.resolve()}")

    df = pd.read_csv(input_csv)
    validate_input_df(df)

    print(
        f"\nLoaded {df.shape[0]} rows across "
        f"{df['experiment_id'].nunique()} unique experiments."
    )

    print("Task breakdown:")
    task_breakdown = (
        df[["experiment_id", "task_type"]]
        .drop_duplicates()
        .groupby("task_type")
        .size()
        .reset_index(name="n_experiments")
    )
    print(task_breakdown.to_string(index=False))

    for task_type in ["binary", "categorical"]:
        run_visualization_for_task(
            df=df,
            task_type=task_type,
            output_dir=output_dir,
        )

    print("\nAll visualizations written to:")
    print(f"  {output_dir.resolve()}")


if __name__ == "__main__":
    main()