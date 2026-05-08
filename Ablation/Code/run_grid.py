############################################################
# run_grid.py
# Run a grid of binary and categorical MMIL / MCEM ablation experiments.
#
# Python version of 07_run_grid.R
#
# Key behavior:
#   Resume / skip no longer depends only on the unstable EXP001 / EXP002
#   prefix. If an old completed directory has the same experiment content
#   after removing the EXP###_ prefix, it is recognized and skipped.
#
# Usage:
#   python run_grid.py config/experiment_grid.yml
#   python run_grid.py config/experiment_grid.yml --force-rerun
############################################################

from __future__ import annotations

import argparse
import gc
import os
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from utils import (
    coalesce,
    ensure_dir,
    normalize_categorical_scheme,
    normalize_task_type,
)
from run_one_experiment import (
    prepare_shared_data_features_and_dfs,
    run_one_experiment,
)


############################################################
# Read grid config
############################################################

def read_grid_config(grid_config_file: str | Path) -> dict[str, Any]:
    grid_config_file = Path(grid_config_file)

    if not grid_config_file.exists():
        raise FileNotFoundError(f"Grid config file not found: {grid_config_file}")

    with open(grid_config_file, "r", encoding="utf-8") as f:
        grid_config = yaml.safe_load(f)

    if grid_config is None:
        grid_config = {}

    return grid_config


############################################################
# Normalize task config
############################################################

def normalize_task_grid_item(task_item: Any) -> dict[str, Any]:
    if isinstance(task_item, str):
        task_type = normalize_task_type(task_item)

        return {
            "task_type": task_type,
            "categorical_scheme": "three_stage",
        }

    if isinstance(task_item, dict):
        task_type = normalize_task_type(
            coalesce(task_item.get("task_type"), "binary")
        )

        categorical_scheme = normalize_categorical_scheme(
            coalesce(task_item.get("categorical_scheme"), "three_stage")
        )

        return {
            "task_type": task_type,
            "categorical_scheme": categorical_scheme,
        }

    raise ValueError("Each task entry must be either a string or a dict.")


############################################################
# Build one experiment config from grid elements
############################################################

def build_single_experiment_config(
    grid_config: dict[str, Any],
    experiment_id: str,
    task_config: dict[str, Any],
    feature_method: str,
    n_components: int,
    model_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    project = coalesce(grid_config.get("project"), {})
    defaults = coalesce(grid_config.get("defaults"), {})

    task_type = normalize_task_type(
        coalesce(task_config.get("task_type"), "binary")
    )

    categorical_scheme = normalize_categorical_scheme(
        coalesce(
            task_config.get("categorical_scheme"),
            coalesce(defaults.get("categorical_scheme"), "three_stage"),
        )
    )

    experiment = {
        "experiment_id": experiment_id,

        "task_type": task_type,
        "categorical_scheme": categorical_scheme,

        "feature_method": feature_method,
        "n_components": n_components,
        "seed": seed,

        "test_fraction": coalesce(defaults.get("test_fraction"), 0.25),

        "min_cell_fraction": coalesce(defaults.get("min_cell_fraction"), 0.01),
        "n_hvg": coalesce(defaults.get("n_hvg"), 2000),
        "scale_factor": coalesce(defaults.get("scale_factor"), 10000),

        "nmf_method": coalesce(defaults.get("nmf_method"), "brunet"),
        "nmf_nrun": coalesce(defaults.get("nmf_nrun"), 1),

        "rho": coalesce(defaults.get("rho"), 0.70),
        "label_strength": coalesce(defaults.get("label_strength"), 0.70),

        "max_iter": coalesce(defaults.get("max_iter"), 30),
        "tol": coalesce(defaults.get("tol"), 1e-4),
        "mcem_samples": coalesce(defaults.get("mcem_samples"), 30),

        "save_plots": coalesce(defaults.get("save_plots"), True),
    }

    backend = dict(model_config)

    single_config = {
        "project": project,
        "experiment": experiment,
        "backend": backend,
    }

    return single_config


############################################################
# Expand grid config
############################################################

def expand_experiment_grid(
    grid_config: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_tasks = coalesce(grid_config.get("tasks"), ["binary"])

    task_configs = [
        normalize_task_grid_item(task_item)
        for task_item in raw_tasks
    ]

    features = coalesce(grid_config.get("features"), {})

    methods = coalesce(features.get("methods"), ["PCA", "ICA", "NMF"])
    n_components_values = coalesce(features.get("n_components"), [20])

    models = coalesce(
        grid_config.get("models"),
        [
            {
                "model_name": "glmnet_ridge",
                "backend": "glmnet",
                "glmnet_alpha": 0,
            }
        ],
    )

    seeds = coalesce(grid_config.get("seeds"), [2026])

    experiment_configs = []
    experiment_counter = 1

    for task_config in task_configs:
        for feature_method in methods:
            for n_components in n_components_values:
                for model_config in models:
                    for seed in seeds:
                        task_type = task_config["task_type"]
                        categorical_scheme = task_config["categorical_scheme"]
                        model_name = coalesce(
                            model_config.get("model_name"),
                            model_config.get("backend"),
                        )

                        if task_type == "categorical":
                            task_name = f"{task_type}_{categorical_scheme}"
                        else:
                            task_name = task_type

                        experiment_id = (
                            f"EXP{experiment_counter:03d}_"
                            f"{task_name}_"
                            f"{feature_method}"
                            f"k{n_components}_"
                            f"{model_name}_"
                            f"seed{seed}"
                        )

                        single_config = build_single_experiment_config(
                            grid_config=grid_config,
                            experiment_id=experiment_id,
                            task_config=task_config,
                            feature_method=feature_method,
                            n_components=int(n_components),
                            model_config=model_config,
                            seed=int(seed),
                        )

                        experiment_configs.append(single_config)
                        experiment_counter += 1

    return experiment_configs


############################################################
# Save grid expansion table
############################################################

def make_grid_table(
    experiment_configs: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for cfg in experiment_configs:
        rows.append(
            {
                "experiment_id": cfg["experiment"]["experiment_id"],

                "task_type": cfg["experiment"]["task_type"],
                "categorical_scheme": cfg["experiment"]["categorical_scheme"],

                "feature_method": cfg["experiment"]["feature_method"],
                "n_components": cfg["experiment"]["n_components"],

                "model_name": cfg["backend"].get("model_name", np.nan),
                "backend": cfg["backend"]["backend"],
                "glmnet_alpha": cfg["backend"].get("glmnet_alpha", np.nan),

                "seed": cfg["experiment"]["seed"],

                "rho": cfg["experiment"]["rho"],
                "label_strength": cfg["experiment"]["label_strength"],
                "mcem_samples": cfg["experiment"]["mcem_samples"],
            }
        )

    return pd.DataFrame(rows)


############################################################
# Resume helpers
############################################################

def get_grid_out_dir(grid_config: dict[str, Any]) -> str:
    project = coalesce(grid_config.get("project"), {})
    proj_dir = coalesce(project.get("proj_dir"), os.getcwd())
    out_dir = coalesce(project.get("out_dir"), str(Path(proj_dir) / "Output"))
    return str(out_dir)


def get_experiment_out_dir_for_cfg(
    grid_config: dict[str, Any],
    cfg: dict[str, Any],
) -> str:
    out_dir = get_grid_out_dir(grid_config)
    task_type = normalize_task_type(cfg["experiment"]["task_type"])
    experiment_id = cfg["experiment"]["experiment_id"]

    return str(
        Path(out_dir)
        / task_type
        / "experiments"
        / experiment_id
    )


def is_experiment_complete(experiment_out_dir: str | Path) -> bool:
    summary_file = Path(experiment_out_dir) / "experiment_summary.csv"

    if not summary_file.exists():
        return False

    if summary_file.stat().st_size == 0:
        return False

    try:
        df = pd.read_csv(summary_file)
        return df.shape[0] > 0
    except Exception:
        return False


def load_existing_summary_safe(
    experiment_out_dir: str | Path,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    summary_file = Path(experiment_out_dir) / "experiment_summary.csv"

    try:
        out = pd.read_csv(summary_file)
    except Exception:
        return None

    # If summary is loaded from an old EXP-numbered directory,
    # rewrite experiment_id to the current grid ID so combined summaries
    # match the current expanded grid table.
    if cfg is not None and "experiment_id" in out.columns:
        out["experiment_id"] = cfg["experiment"]["experiment_id"]

    return out


def strip_experiment_counter_prefix(experiment_id: str) -> str:
    return re.sub(r"^EXP[0-9]+_", "", str(experiment_id))


def find_existing_experiment_dir(
    grid_config: dict[str, Any],
    cfg: dict[str, Any],
    force_rerun: bool = False,
) -> dict[str, Any]:
    expected_dir = get_experiment_out_dir_for_cfg(grid_config, cfg)

    if force_rerun:
        return {
            "complete": False,
            "experiment_dir": expected_dir,
            "expected_dir": expected_dir,
            "match_type": "force_rerun",
        }

    # 1. First try exact current directory.
    if is_experiment_complete(expected_dir):
        return {
            "complete": True,
            "experiment_dir": expected_dir,
            "expected_dir": expected_dir,
            "match_type": "exact",
        }

    # 2. Then try old directories with different EXP### prefix.
    out_dir = get_grid_out_dir(grid_config)
    task_type = normalize_task_type(cfg["experiment"]["task_type"])
    experiments_root = Path(out_dir) / task_type / "experiments"

    if not experiments_root.exists():
        return {
            "complete": False,
            "experiment_dir": expected_dir,
            "expected_dir": expected_dir,
            "match_type": "none",
        }

    target_key = strip_experiment_counter_prefix(
        cfg["experiment"]["experiment_id"]
    )

    candidate_dirs = [
        p for p in experiments_root.iterdir()
        if p.is_dir()
    ]

    matched_dirs = [
        p for p in candidate_dirs
        if strip_experiment_counter_prefix(p.name) == target_key
    ]

    if not matched_dirs:
        return {
            "complete": False,
            "experiment_dir": expected_dir,
            "expected_dir": expected_dir,
            "match_type": "none",
        }

    completed_dirs = [
        p for p in matched_dirs
        if is_experiment_complete(p)
    ]

    if not completed_dirs:
        return {
            "complete": False,
            "experiment_dir": expected_dir,
            "expected_dir": expected_dir,
            "match_type": "matched_but_incomplete",
        }

    # If multiple completed dirs match, use the newest summary file.
    completed_dirs = sorted(
        completed_dirs,
        key=lambda p: (p / "experiment_summary.csv").stat().st_mtime,
        reverse=True,
    )

    best_dir = completed_dirs[0]

    return {
        "complete": True,
        "experiment_dir": str(best_dir),
        "expected_dir": expected_dir,
        "match_type": "suffix",
    }


############################################################
# Feature signature & grouping
############################################################

def make_feature_group_signature(cfg: dict[str, Any]) -> str:
    e = cfg["experiment"]

    if e["task_type"] == "categorical":
        task_part = f"{e['task_type']}:{e['categorical_scheme']}"
    else:
        task_part = e["task_type"]

    parts = [
        task_part,
        e["seed"],
        e["test_fraction"],
        e["feature_method"],
        e["n_components"],
        e["min_cell_fraction"],
        e["n_hvg"],
        e["scale_factor"],
        e["nmf_method"],
        e["nmf_nrun"],
    ]

    return "|".join(map(str, parts))


def group_experiments_by_features(
    experiment_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    signatures = [
        make_feature_group_signature(cfg)
        for cfg in experiment_configs
    ]

    unique_sigs = []
    for sig in signatures:
        if sig not in unique_sigs:
            unique_sigs.append(sig)

    groups = []

    for g_idx, sig in enumerate(unique_sigs, start=1):
        indices = [
            i for i, s in enumerate(signatures)
            if s == sig
        ]

        groups.append(
            {
                "group_index": g_idx,
                "signature": sig,
                "indices": indices,
                "experiments": [experiment_configs[i] for i in indices],
            }
        )

    return groups


############################################################
# Save split summaries / failures / skipped logs
############################################################

def save_task_split_summaries(
    all_summary_df: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(out_dir)

    global_summary_dir = out_dir / "summary"
    binary_summary_dir = out_dir / "binary" / "summary"
    categorical_summary_dir = out_dir / "categorical" / "summary"

    ensure_dir(global_summary_dir)
    ensure_dir(binary_summary_dir)
    ensure_dir(categorical_summary_dir)

    all_summary_file = global_summary_dir / "all_experiment_metrics.csv"
    all_summary_df.to_csv(all_summary_file, index=False)

    binary_summary_df = all_summary_df[
        all_summary_df["task_type"] == "binary"
    ].copy()

    binary_summary_file = binary_summary_dir / "all_binary_experiment_metrics.csv"
    binary_summary_df.to_csv(binary_summary_file, index=False)

    categorical_summary_df = all_summary_df[
        all_summary_df["task_type"] == "categorical"
    ].copy()

    categorical_summary_file = (
        categorical_summary_dir / "all_categorical_experiment_metrics.csv"
    )
    categorical_summary_df.to_csv(categorical_summary_file, index=False)

    print("\nSaved combined experiment metrics:")
    print(all_summary_file)
    print(binary_summary_file)
    print(categorical_summary_file)

    return {
        "all_summary_file": str(all_summary_file),
        "binary_summary_file": str(binary_summary_file),
        "categorical_summary_file": str(categorical_summary_file),
    }


def save_task_split_failures(
    failed_experiments: list[pd.DataFrame],
    out_dir: str | Path,
) -> dict[str, str] | None:
    if len(failed_experiments) == 0:
        return None

    out_dir = Path(out_dir)

    global_summary_dir = out_dir / "summary"
    binary_summary_dir = out_dir / "binary" / "summary"
    categorical_summary_dir = out_dir / "categorical" / "summary"

    ensure_dir(global_summary_dir)
    ensure_dir(binary_summary_dir)
    ensure_dir(categorical_summary_dir)

    failed_df = pd.concat(failed_experiments, axis=0, ignore_index=True)

    failed_file = global_summary_dir / "failed_experiments.csv"
    failed_df.to_csv(failed_file, index=False)

    failed_binary_file = binary_summary_dir / "failed_binary_experiments.csv"
    failed_df[failed_df["task_type"] == "binary"].to_csv(
        failed_binary_file,
        index=False,
    )

    failed_categorical_file = (
        categorical_summary_dir / "failed_categorical_experiments.csv"
    )
    failed_df[failed_df["task_type"] == "categorical"].to_csv(
        failed_categorical_file,
        index=False,
    )

    print("\nSome experiments failed. Saved failure logs:")
    print(failed_file)
    print(failed_binary_file)
    print(failed_categorical_file)

    return {
        "failed_file": str(failed_file),
        "failed_binary_file": str(failed_binary_file),
        "failed_categorical_file": str(failed_categorical_file),
    }


def save_task_split_skipped(
    skipped_experiments: list[pd.DataFrame],
    out_dir: str | Path,
) -> dict[str, str] | None:
    if len(skipped_experiments) == 0:
        return None

    out_dir = Path(out_dir)

    global_summary_dir = out_dir / "summary"
    binary_summary_dir = out_dir / "binary" / "summary"
    categorical_summary_dir = out_dir / "categorical" / "summary"

    ensure_dir(global_summary_dir)
    ensure_dir(binary_summary_dir)
    ensure_dir(categorical_summary_dir)

    skipped_df = pd.concat(skipped_experiments, axis=0, ignore_index=True)

    skipped_file = global_summary_dir / "skipped_experiments.csv"
    skipped_df.to_csv(skipped_file, index=False)

    skipped_binary_file = binary_summary_dir / "skipped_binary_experiments.csv"
    skipped_df[skipped_df["task_type"] == "binary"].to_csv(
        skipped_binary_file,
        index=False,
    )

    skipped_categorical_file = (
        categorical_summary_dir / "skipped_categorical_experiments.csv"
    )
    skipped_df[skipped_df["task_type"] == "categorical"].to_csv(
        skipped_categorical_file,
        index=False,
    )

    print("\nSaved skipped experiment logs:")
    print(skipped_file)
    print(skipped_binary_file)
    print(skipped_categorical_file)

    return {
        "skipped_file": str(skipped_file),
        "skipped_binary_file": str(skipped_binary_file),
        "skipped_categorical_file": str(skipped_categorical_file),
    }


############################################################
# Pre-flight scan
############################################################

def scan_experiments_for_resume(
    experiment_configs: list[dict[str, Any]],
    grid_config: dict[str, Any],
    force_rerun: bool = False,
) -> pd.DataFrame:
    rows = []

    for i, cfg in enumerate(experiment_configs, start=1):
        found = find_existing_experiment_dir(
            grid_config=grid_config,
            cfg=cfg,
            force_rerun=force_rerun,
        )

        rows.append(
            {
                "index": i,
                "experiment_id": cfg["experiment"]["experiment_id"],
                "task_type": cfg["experiment"]["task_type"],
                "categorical_scheme": cfg["experiment"]["categorical_scheme"],
                "feature_method": cfg["experiment"]["feature_method"],
                "n_components": cfg["experiment"]["n_components"],
                "model_name": cfg["backend"].get("model_name", np.nan),
                "backend": cfg["backend"]["backend"],
                "seed": cfg["experiment"]["seed"],
                "expected_experiment_dir": found["expected_dir"],
                "experiment_dir": found["experiment_dir"],
                "resume_match_type": found["match_type"],
                "already_complete": bool(found["complete"]),
                "will_run": not bool(found["complete"]),
            }
        )

    return pd.DataFrame(rows)


############################################################
# Helpers used by run loop
############################################################

def record_skipped_experiment(
    cfg: dict[str, Any],
    expected_dir: str,
    loaded_summary_ok: bool,
    match_type: str | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": cfg["experiment"]["experiment_id"],
                "task_type": cfg["experiment"]["task_type"],
                "categorical_scheme": cfg["experiment"]["categorical_scheme"],
                "feature_method": cfg["experiment"]["feature_method"],
                "n_components": cfg["experiment"]["n_components"],
                "model_name": cfg["backend"].get("model_name", np.nan),
                "backend": cfg["backend"]["backend"],
                "seed": cfg["experiment"]["seed"],
                "experiment_dir": expected_dir,
                "resume_match_type": match_type,
                "loaded_summary": loaded_summary_ok,
            }
        ]
    )


def record_failed_experiment(
    cfg: dict[str, Any],
    error_message: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": cfg["experiment"]["experiment_id"],
                "task_type": cfg["experiment"]["task_type"],
                "categorical_scheme": cfg["experiment"]["categorical_scheme"],
                "feature_method": cfg["experiment"]["feature_method"],
                "n_components": cfg["experiment"]["n_components"],
                "model_name": cfg["backend"].get("model_name", np.nan),
                "backend": cfg["backend"]["backend"],
                "seed": cfg["experiment"]["seed"],
                "error_message": error_message,
            }
        ]
    )


############################################################
# Main grid runner
############################################################

def run_experiment_grid(
    grid_config_file: str | Path,
    force_rerun: bool | None = None,
) -> dict[str, Any]:
    grid_config = read_grid_config(grid_config_file)

    if force_rerun is None:
        force_rerun = bool(grid_config.get("force_rerun", False))
    else:
        force_rerun = bool(force_rerun)

    out_dir = get_grid_out_dir(grid_config)

    global_summary_dir = Path(out_dir) / "summary"
    binary_summary_dir = Path(out_dir) / "binary" / "summary"
    categorical_summary_dir = Path(out_dir) / "categorical" / "summary"

    ensure_dir(global_summary_dir)
    ensure_dir(binary_summary_dir)
    ensure_dir(categorical_summary_dir)

    experiment_configs = expand_experiment_grid(grid_config)
    grid_table = make_grid_table(experiment_configs)

    ############################################################
    # Save expanded grid tables
    ############################################################

    grid_table_file = global_summary_dir / "expanded_experiment_grid.csv"
    grid_table.to_csv(grid_table_file, index=False)

    binary_grid_table_file = (
        binary_summary_dir / "expanded_binary_experiment_grid.csv"
    )

    categorical_grid_table_file = (
        categorical_summary_dir / "expanded_categorical_experiment_grid.csv"
    )

    grid_table[grid_table["task_type"] == "binary"].to_csv(
        binary_grid_table_file,
        index=False,
    )

    grid_table[grid_table["task_type"] == "categorical"].to_csv(
        categorical_grid_table_file,
        index=False,
    )

    print("\nExpanded experiment grid:")
    print(grid_table)

    print("\nSaved expanded grid tables:")
    print(grid_table_file)
    print(binary_grid_table_file)
    print(categorical_grid_table_file)

    ############################################################
    # Pre-flight resume scan
    ############################################################

    resume_status = scan_experiments_for_resume(
        experiment_configs=experiment_configs,
        grid_config=grid_config,
        force_rerun=force_rerun,
    )

    resume_status_file = global_summary_dir / "resume_status_preflight.csv"
    resume_status.to_csv(resume_status_file, index=False)

    n_total = int(resume_status.shape[0])
    n_skip = int(resume_status["already_complete"].sum())
    n_run = int(resume_status["will_run"].sum())

    print("\n############################################################")
    print("Resume / skip pre-flight")
    print("############################################################")
    print(f"force_rerun:        {force_rerun}")
    print(f"Total experiments:  {n_total}")
    print(f"Already complete:   {n_skip} (will be skipped)")
    print(f"Pending:            {n_run} (will be run)")
    print(f"Pre-flight log:     {resume_status_file}")
    print("############################################################")

    ############################################################
    # Group experiments by feature signature
    ############################################################

    groups = group_experiments_by_features(experiment_configs)

    print(
        f"\n[grouping] {len(experiment_configs)} experiments grouped into "
        f"{len(groups)} feature group(s)."
    )
    print(
        f"[grouping] Feature extraction will run {len(groups)}x "
        f"instead of {len(experiment_configs)}x."
    )

    all_summaries: list[pd.DataFrame] = []
    failed_experiments: list[pd.DataFrame] = []
    skipped_experiments: list[pd.DataFrame] = []

    ############################################################
    # Loop over groups
    ############################################################

    for g_idx, group in enumerate(groups, start=1):
        n_in_group = len(group["experiments"])
        group_status = resume_status.iloc[group["indices"]].reset_index(drop=True)

        n_pending_in_group = int(group_status["will_run"].sum())
        n_skip_in_group = int(group_status["already_complete"].sum())

        print("\n############################################################")
        print(
            f"Group {g_idx} / {len(groups)} | signature: {group['signature']}"
        )
        print(
            f"  Experiments in group: {n_in_group} "
            f"(skip: {n_skip_in_group}, run: {n_pending_in_group})"
        )
        print("############################################################")

        ########################################################
        # Case A: all complete
        ########################################################

        if n_pending_in_group == 0:
            print("[group skip] All experiments in group already complete.")

            for j in range(n_in_group):
                cfg_j = group["experiments"][j]
                existing_dir_j = group_status.loc[j, "experiment_dir"]
                match_type_j = group_status.loc[j, "resume_match_type"]

                existing = load_existing_summary_safe(existing_dir_j, cfg_j)

                if existing is not None:
                    all_summaries.append(existing)

                skipped_experiments.append(
                    record_skipped_experiment(
                        cfg=cfg_j,
                        expected_dir=existing_dir_j,
                        loaded_summary_ok=existing is not None,
                        match_type=match_type_j,
                    )
                )

            continue

        ########################################################
        # Case B: at least one pending, compute shared data/features
        ########################################################

        print("\n[group prepare] Computing shared data + features...")

        try:
            shared = prepare_shared_data_features_and_dfs(
                config=group["experiments"][0]
            )

        except Exception as exc:
            print("\n[group failed] Shared data/feature preparation failed:")
            print(str(exc))

            for j in range(n_in_group):
                cfg_j = group["experiments"][j]
                existing_dir_j = group_status.loc[j, "experiment_dir"]
                match_type_j = group_status.loc[j, "resume_match_type"]

                if bool(group_status.loc[j, "already_complete"]):
                    existing = load_existing_summary_safe(existing_dir_j, cfg_j)

                    if existing is not None:
                        all_summaries.append(existing)

                    skipped_experiments.append(
                        record_skipped_experiment(
                            cfg=cfg_j,
                            expected_dir=existing_dir_j,
                            loaded_summary_ok=existing is not None,
                            match_type=match_type_j,
                        )
                    )

                else:
                    failed_experiments.append(
                        record_failed_experiment(
                            cfg=cfg_j,
                            error_message=(
                                "Group feature preparation failed; "
                                "experiment not attempted. "
                                f"Original error: {exc}"
                            ),
                        )
                    )

            continue

        ########################################################
        # Run / skip experiments in group
        ########################################################

        for j in range(n_in_group):
            cfg_j = group["experiments"][j]
            existing_dir_j = group_status.loc[j, "experiment_dir"]
            match_type_j = group_status.loc[j, "resume_match_type"]
            experiment_id = cfg_j["experiment"]["experiment_id"]

            if bool(group_status.loc[j, "already_complete"]):
                print(
                    f"\n  [skip] {experiment_id} — already complete "
                    f"({match_type_j})."
                )

                existing = load_existing_summary_safe(existing_dir_j, cfg_j)

                if existing is not None:
                    all_summaries.append(existing)

                skipped_experiments.append(
                    record_skipped_experiment(
                        cfg=cfg_j,
                        expected_dir=existing_dir_j,
                        loaded_summary_ok=existing is not None,
                        match_type=match_type_j,
                    )
                )

                continue

            backend = cfg_j["backend"]["backend"]

            alpha_msg = ""
            if backend == "glmnet":
                alpha_msg = f", alpha = {cfg_j['backend'].get('glmnet_alpha', np.nan)}"

            print(
                f"\n  [run] {experiment_id} "
                f"(group {g_idx} / {len(groups)}, item {j + 1} / {n_in_group})"
            )
            print(f"        backend: {backend}{alpha_msg}")

            try:
                result_j = run_one_experiment(
                    config=cfg_j,
                    r_dir="R",
                    source_scripts=False,
                    precomputed_data=shared["data"],
                    precomputed_features=shared["features"],
                    precomputed_model_dfs=shared["model_dfs"],
                )

            except Exception as exc:
                print("\nExperiment failed:")
                print(experiment_id)
                print("Error message:")
                print(str(exc))

                traceback.print_exc()

                failed_experiments.append(
                    record_failed_experiment(
                        cfg=cfg_j,
                        error_message=str(exc),
                    )
                )

                result_j = None

            if result_j is not None:
                all_summaries.append(result_j["summary"])

        del shared
        gc.collect()

    ############################################################
    # Save combined summaries
    ############################################################

    summary_files = None

    if len(all_summaries) > 0:
        all_summary_df = pd.concat(
            all_summaries,
            axis=0,
            ignore_index=True,
        )

        summary_files = save_task_split_summaries(
            all_summary_df=all_summary_df,
            out_dir=out_dir,
        )

        print("\nTop results by metric_value:")
        print(
            all_summary_df.sort_values(
                "metric_value",
                ascending=False,
                na_position="last",
            )
            .head(20)
            .to_string(index=False)
        )

        if "auroc" in all_summary_df.columns:
            print("\nTop binary results by AUROC:")
            print(
                all_summary_df[
                    all_summary_df["task_type"] == "binary"
                ]
                .sort_values(
                    "auroc",
                    ascending=False,
                    na_position="last",
                )
                .head(20)
                .to_string(index=False)
            )

        if "accuracy" in all_summary_df.columns:
            print("\nTop categorical results by accuracy:")
            print(
                all_summary_df[
                    all_summary_df["task_type"] == "categorical"
                ]
                .sort_values(
                    "accuracy",
                    ascending=False,
                    na_position="last",
                )
                .head(20)
                .to_string(index=False)
            )

    else:
        print("\nNo successful experiments to summarize.")

    failure_files = save_task_split_failures(
        failed_experiments=failed_experiments,
        out_dir=out_dir,
    )

    skipped_files = save_task_split_skipped(
        skipped_experiments=skipped_experiments,
        out_dir=out_dir,
    )

    ############################################################
    # Final summary banner
    ############################################################

    n_run_attempted = n_run
    n_run_failed = len(failed_experiments)
    n_run_succeeded = n_run_attempted - n_run_failed
    n_skipped = len(skipped_experiments)
    n_groups = len(groups)

    print("\n############################################################")
    print("Grid run completed.")
    print("############################################################")
    print(f"Total experiments in grid:     {n_total}")
    print(f"Feature groups:                {n_groups}")
    print(f"Skipped already complete:      {n_skipped}")
    print(f"Attempted this run:            {n_run_attempted}")
    print(f"  succeeded:                   {n_run_succeeded}")
    print(f"  failed:                      {n_run_failed}")
    print("############################################################")

    return {
        "grid_table": grid_table,
        "resume_status": resume_status,
        "groups": groups,
        "summaries": all_summaries,
        "failed": failed_experiments,
        "skipped": skipped_experiments,
        "summary_files": summary_files,
        "failure_files": failure_files,
        "skipped_files": skipped_files,
        "counts": {
            "total": n_total,
            "groups": n_groups,
            "skipped": n_skipped,
            "attempted": n_run_attempted,
            "succeeded": n_run_succeeded,
            "failed": n_run_failed,
        },
    }


############################################################
# Command-line entry point
############################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MMIL / MCEM experiment grid."
    )

    parser.add_argument(
        "grid_config_file",
        help="Path to experiment grid YAML config.",
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore completed experiment summaries and rerun everything.",
    )

    args = parser.parse_args()

    run_experiment_grid(
        grid_config_file=args.grid_config_file,
        force_rerun=True if args.force_rerun else None,
    )


if __name__ == "__main__":
    main()