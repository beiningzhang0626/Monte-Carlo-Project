############################################################
# run_one_experiment.py
# Run one MMIL / MCEM ablation experiment.
#
# Python version of 06_run_one_experiment.R
#
# Supports:
#   1. binary
#   2. categorical
#
# Output structure:
#   Output/
#   ├── binary/
#   │   └── experiments/
#   └── categorical/
#       └── experiments/
#
# Usage from command line:
#   python run_one_experiment.py config/single_experiment.yml
#
# Resource-sharing extension:
#   run_one_experiment() optionally accepts precomputed_data,
#   precomputed_features, and precomputed_model_dfs. When provided,
#   the corresponding pipeline stage is skipped and reused.
############################################################

from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
import yaml

from utils import (
    coalesce,
    ensure_dir,
    get_categorical_class_levels,
    make_experiment_name,
    normalize_categorical_scheme,
    normalize_task_type,
)
from load_and_split import load_and_split_data
from feature_extractors import (
    build_feature_modeling_dfs,
    extract_features,
)
from mmil_wrappers import (
    run_binary_mmil,
    run_categorical_mmil,
    save_mmil_outputs,
)
from patient_level_evaluation import evaluate_predictions


############################################################
# Source helper
############################################################

def source_pipeline_scripts(r_dir: str = "R") -> bool:
    """
    Kept only for conceptual compatibility with the R version.

    In Python, modules are imported at the top of this file, so there is
    no equivalent need to source scripts dynamically.
    """
    return True


############################################################
# Small helper functions
############################################################

def is_absolute_path(path: str | Path | None) -> bool:
    if path is None:
        return False

    path_str = str(path)

    # Unix / macOS absolute path.
    if Path(path_str).is_absolute():
        return True

    # Windows absolute path, e.g. C:/... or C:\...
    if re.match(r"^[A-Za-z]:[\\/]", path_str):
        return True

    # UNC path.
    if path_str.startswith("\\\\"):
        return True

    return False


def resolve_path(path: str | Path | None, base_dir: str | Path) -> str | None:
    if path is None:
        return None

    if is_absolute_path(path):
        return str(path)

    return str(Path(base_dir) / str(path))


def _yaml_safe(obj: Any) -> Any:
    """
    Convert objects to YAML-serializable plain Python types.
    """
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, PureWindowsPath):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_yaml_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [_yaml_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) else val

    if isinstance(obj, np.ndarray):
        return [_yaml_safe(v) for v in obj.tolist()]

    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple, np.ndarray)) else False:
        return None

    return obj


############################################################
# Read and normalize one experiment config
############################################################

def read_experiment_config(config_file: str | Path) -> dict[str, Any]:
    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def normalize_experiment_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    raw_config = raw_config or {}

    project = coalesce(raw_config.get("project"), {})
    experiment = coalesce(raw_config.get("experiment"), {})
    backend = coalesce(raw_config.get("backend"), {})

    ############################################################
    # Project paths
    ############################################################

    proj_dir = coalesce(project.get("proj_dir"), os.getcwd())
    in_dir = coalesce(project.get("in_dir"), str(Path(proj_dir) / "Input"))
    out_dir = coalesce(project.get("out_dir"), str(Path(proj_dir) / "Output"))

    expr_file = coalesce(
        project.get("expr_file"),
        "GSE131907_pilot_expression_subset_dt.rds",
    )

    meta_file = coalesce(
        project.get("meta_file"),
        "GSE131907_pilot_sampled_cell_metadata_with_clinical.csv",
    )

    expr_file = resolve_path(expr_file, in_dir)
    meta_file = resolve_path(meta_file, in_dir)

    ############################################################
    # Task settings
    ############################################################

    task_type = normalize_task_type(
        coalesce(experiment.get("task_type"), "binary")
    )

    categorical_scheme = normalize_categorical_scheme(
        coalesce(experiment.get("categorical_scheme"), "three_stage")
    )

    class_levels = get_categorical_class_levels(
        categorical_scheme=categorical_scheme,
    )

    ############################################################
    # General experiment settings
    ############################################################

    seed = coalesce(experiment.get("seed"), 2026)
    test_fraction = coalesce(experiment.get("test_fraction"), 0.25)

    feature_method = coalesce(experiment.get("feature_method"), "PCA")
    n_components = coalesce(experiment.get("n_components"), 20)

    min_cell_fraction = coalesce(experiment.get("min_cell_fraction"), 0.01)
    n_hvg = coalesce(experiment.get("n_hvg"), 2000)
    scale_factor = coalesce(experiment.get("scale_factor"), 10000)

    nmf_method = coalesce(experiment.get("nmf_method"), "brunet")
    nmf_nrun = coalesce(experiment.get("nmf_nrun"), 1)

    ############################################################
    # MMIL settings
    ############################################################

    rho = coalesce(experiment.get("rho"), 0.70)
    label_strength = coalesce(experiment.get("label_strength"), 0.70)

    max_iter = coalesce(experiment.get("max_iter"), 30)
    tol = coalesce(experiment.get("tol"), 1e-4)
    mcem_samples = coalesce(experiment.get("mcem_samples"), 30)

    save_plots = coalesce(experiment.get("save_plots"), True)

    ############################################################
    # Backend settings
    ############################################################

    model_backend = coalesce(backend.get("backend"), "glmnet")
    glmnet_alpha = backend.get("glmnet_alpha", np.nan)

    ############################################################
    # Experiment ID
    ############################################################

    tmp_config = {
        "task_type": task_type,
        "categorical_scheme": categorical_scheme,
        "feature_method": feature_method,
        "n_components": n_components,
        "model_backend": model_backend,
        "glmnet_alpha": glmnet_alpha,
        "seed": seed,
    }

    experiment_id = coalesce(
        experiment.get("experiment_id"),
        make_experiment_name(tmp_config),
    )

    ############################################################
    # Output directories
    ############################################################

    task_out_dir = str(Path(out_dir) / task_type)

    experiment_out_dir = str(
        Path(task_out_dir) / "experiments" / str(experiment_id)
    )

    ensure_dir(experiment_out_dir)

    ############################################################
    # Backend config
    ############################################################

    backend_config = copy.deepcopy(backend)
    backend_config["backend"] = model_backend

    if model_backend == "glmnet" and backend_config.get("glmnet_alpha") is None:
        backend_config["glmnet_alpha"] = 0

    ############################################################
    # Normalized config object
    ############################################################

    normalized = {
        "experiment_id": experiment_id,

        "task_type": task_type,
        "categorical_scheme": categorical_scheme,
        "class_levels": class_levels,

        "proj_dir": proj_dir,
        "in_dir": in_dir,
        "out_dir": out_dir,
        "task_out_dir": task_out_dir,
        "experiment_out_dir": experiment_out_dir,

        "expr_file": expr_file,
        "meta_file": meta_file,

        "seed": int(seed),
        "test_fraction": float(test_fraction),

        "feature_method": feature_method,
        "n_components": int(n_components),
        "min_cell_fraction": float(min_cell_fraction),
        "n_hvg": int(n_hvg),
        "scale_factor": float(scale_factor),
        "nmf_method": nmf_method,
        "nmf_nrun": int(nmf_nrun),

        "rho": float(rho),
        "label_strength": float(label_strength),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "mcem_samples": int(mcem_samples),

        "save_plots": bool(save_plots),

        "model_backend": model_backend,
        "glmnet_alpha": glmnet_alpha,
        "backend_config": backend_config,
    }

    return normalized


############################################################
# Save config used
############################################################

def save_config_used(
    raw_config: dict[str, Any],
    normalized_config: dict[str, Any],
) -> dict[str, str]:
    out_dir = Path(normalized_config["experiment_out_dir"])
    ensure_dir(out_dir)

    raw_file = out_dir / "config_used_raw.yml"
    normalized_file = out_dir / "config_used_normalized.yml"

    with open(raw_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            _yaml_safe(raw_config),
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    with open(normalized_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            _yaml_safe(normalized_config),
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    return {
        "raw_file": str(raw_file),
        "normalized_file": str(normalized_file),
    }


############################################################
# Validation helpers for precomputed objects
############################################################

def validate_precomputed_features(
    cfg: dict[str, Any],
    precomputed_features: dict[str, Any] | None,
) -> bool:
    if precomputed_features is None:
        return True

    if precomputed_features.get("feature_method") is not None:
        if precomputed_features["feature_method"] != cfg["feature_method"]:
            raise ValueError(
                "precomputed_features['feature_method'] "
                f"('{precomputed_features['feature_method']}') does not match "
                f"cfg['feature_method'] ('{cfg['feature_method']}')."
            )

    if precomputed_features.get("n_components") is not None:
        if precomputed_features["n_components"] != cfg["n_components"]:
            raise ValueError(
                "precomputed_features['n_components'] "
                f"({precomputed_features['n_components']}) does not match "
                f"cfg['n_components'] ({cfg['n_components']})."
            )

    return True


def validate_precomputed_data(
    cfg: dict[str, Any],
    precomputed_data: dict[str, Any] | None,
) -> bool:
    if precomputed_data is None:
        return True

    if precomputed_data.get("task_type") is not None:
        if precomputed_data["task_type"] != cfg["task_type"]:
            raise ValueError(
                "precomputed_data['task_type'] "
                f"('{precomputed_data['task_type']}') does not match "
                f"cfg['task_type'] ('{cfg['task_type']}')."
            )

    if (
        precomputed_data.get("categorical_scheme") is not None
        and cfg["task_type"] == "categorical"
    ):
        if precomputed_data["categorical_scheme"] != cfg["categorical_scheme"]:
            raise ValueError(
                "precomputed_data['categorical_scheme'] "
                f"('{precomputed_data['categorical_scheme']}') does not match "
                f"cfg['categorical_scheme'] ('{cfg['categorical_scheme']}')."
            )

    return True


############################################################
# Build experiment summary table
############################################################

def build_experiment_summary(
    config: dict[str, Any],
    eval_result: dict[str, Any],
) -> pd.DataFrame:
    ############################################################
    # Binary summary
    ############################################################

    if eval_result["task_type"] == "binary":
        summary_df = eval_result["best_test_auc"].copy()

        summary_df["metric_name"] = "auroc"
        summary_df["metric_value"] = summary_df["auroc"]

        summary_df["experiment_id"] = config["experiment_id"]
        summary_df["task_type"] = config["task_type"]
        summary_df["categorical_scheme"] = np.nan
        summary_df["feature_method"] = config["feature_method"]
        summary_df["n_components"] = config["n_components"]
        summary_df["model_backend"] = config["model_backend"]
        summary_df["glmnet_alpha"] = config["glmnet_alpha"]
        summary_df["seed"] = config["seed"]
        summary_df["rho"] = config["rho"]
        summary_df["label_strength"] = np.nan
        summary_df["mcem_samples"] = config["mcem_samples"]

        summary_df["multiclass_logloss"] = np.nan
        summary_df["accuracy"] = np.nan
        summary_df["balanced_accuracy"] = np.nan
        summary_df["macro_f1"] = np.nan

        columns = [
            "experiment_id",
            "task_type",
            "categorical_scheme",
            "feature_method",
            "n_components",
            "model_backend",
            "glmnet_alpha",
            "seed",
            "rho",
            "label_strength",
            "mcem_samples",
            "model",
            "aggregation",
            "n_patients",
            "n_early",
            "n_advanced",
            "auroc",
            "multiclass_logloss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "metric_name",
            "metric_value",
        ]

        return summary_df.loc[:, columns]

    ############################################################
    # Categorical summary
    ############################################################

    if eval_result["task_type"] == "categorical":
        summary_df = eval_result["best_test_metrics"].copy()

        summary_df["metric_name"] = "accuracy"
        summary_df["metric_value"] = summary_df["accuracy"]

        summary_df["experiment_id"] = config["experiment_id"]
        summary_df["task_type"] = config["task_type"]
        summary_df["categorical_scheme"] = config["categorical_scheme"]
        summary_df["feature_method"] = config["feature_method"]
        summary_df["n_components"] = config["n_components"]
        summary_df["model_backend"] = config["model_backend"]
        summary_df["glmnet_alpha"] = config["glmnet_alpha"]
        summary_df["seed"] = config["seed"]
        summary_df["rho"] = np.nan
        summary_df["label_strength"] = config["label_strength"]
        summary_df["mcem_samples"] = config["mcem_samples"]

        summary_df["n_early"] = np.nan
        summary_df["n_advanced"] = np.nan
        summary_df["auroc"] = np.nan

        columns = [
            "experiment_id",
            "task_type",
            "categorical_scheme",
            "feature_method",
            "n_components",
            "model_backend",
            "glmnet_alpha",
            "seed",
            "rho",
            "label_strength",
            "mcem_samples",
            "model",
            "aggregation",
            "n_patients",
            "n_early",
            "n_advanced",
            "auroc",
            "multiclass_logloss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "metric_name",
            "metric_value",
        ]

        return summary_df.loc[:, columns]

    raise ValueError("Unsupported eval_result['task_type'].")


############################################################
# Helper: prepare shared data + features + model_dfs once
############################################################

def prepare_shared_data_features_and_dfs(
    config: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """
    Used by run_grid.py to build shared objects once for a group of
    experiments that have identical data / feature configurations.
    """
    if isinstance(config, (str, Path)):
        raw_config = read_experiment_config(config)
        cfg = normalize_experiment_config(raw_config)

    elif config.get("experiment_id") is not None and config.get("task_out_dir") is not None:
        cfg = config

    else:
        raw_config = config
        cfg = normalize_experiment_config(raw_config)

    print("\n[group prepare] Loading data and extracting features...")
    print(f"  Task type:        {cfg['task_type']}")

    if cfg["task_type"] == "categorical":
        print(f"  Categorical:      {cfg['categorical_scheme']}")

    print(f"  Feature method:   {cfg['feature_method']}")
    print(f"  N components:     {cfg['n_components']}")
    print(f"  Seed:             {cfg['seed']}")

    data_obj = load_and_split_data(cfg)

    feature_obj = extract_features(
        train_expr=data_obj["train_expr"],
        test_expr=data_obj["test_expr"],
        feature_method=cfg["feature_method"],
        n_components=cfg["n_components"],
        min_cell_fraction=cfg["min_cell_fraction"],
        n_hvg=cfg["n_hvg"],
        scale_factor=cfg["scale_factor"],
        seed=cfg["seed"],
        nmf_method=cfg["nmf_method"],
        nmf_nrun=cfg["nmf_nrun"],
    )

    model_dfs = build_feature_modeling_dfs(
        train_meta=data_obj["train_meta"],
        test_meta=data_obj["test_meta"],
        feature_result=feature_obj,
    )

    return {
        "data": data_obj,
        "features": feature_obj,
        "model_dfs": model_dfs,
    }


############################################################
# Main function: run one experiment
############################################################

def run_one_experiment(
    config: str | Path | dict[str, Any],
    r_dir: str = "R",
    source_scripts: bool = False,
    precomputed_data: dict[str, Any] | None = None,
    precomputed_features: dict[str, Any] | None = None,
    precomputed_model_dfs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source_scripts:
        source_pipeline_scripts(r_dir)

    if isinstance(config, (str, Path)):
        raw_config = read_experiment_config(config)
    else:
        raw_config = config

    cfg = normalize_experiment_config(raw_config)

    print("\n============================================================")
    print("Running experiment:")
    print(cfg["experiment_id"])
    print("============================================================")

    print("\nExperiment settings:")
    print("Task type:", cfg["task_type"])

    if cfg["task_type"] == "categorical":
        print("Categorical scheme:", cfg["categorical_scheme"])
        print("Class levels:", ", ".join(cfg["class_levels"]))

    print("Feature method:", cfg["feature_method"])
    print("Number of components:", cfg["n_components"])
    print("Model backend:", cfg["model_backend"])

    if cfg["model_backend"] == "glmnet":
        print("glmnet alpha:", cfg["backend_config"].get("glmnet_alpha"))

    print("Seed:", cfg["seed"])
    print("Task output directory:", cfg["task_out_dir"])
    print("Experiment output directory:", cfg["experiment_out_dir"])

    save_config_used(
        raw_config=raw_config,
        normalized_config=cfg,
    )

    ############################################################
    # Validate precomputed args
    ############################################################

    validate_precomputed_data(cfg, precomputed_data)
    validate_precomputed_features(cfg, precomputed_features)

    ############################################################
    # 1. Load data and split by patient, or reuse precomputed
    ############################################################

    if precomputed_data is None:
        data_obj = load_and_split_data(cfg)
    else:
        print("\n[reuse] Using precomputed data object; skipping load_and_split_data.")
        data_obj = precomputed_data

    ############################################################
    # 2. Feature extraction, or reuse precomputed
    ############################################################

    if precomputed_features is None:
        feature_obj = extract_features(
            train_expr=data_obj["train_expr"],
            test_expr=data_obj["test_expr"],
            feature_method=cfg["feature_method"],
            n_components=cfg["n_components"],
            min_cell_fraction=cfg["min_cell_fraction"],
            n_hvg=cfg["n_hvg"],
            scale_factor=cfg["scale_factor"],
            seed=cfg["seed"],
            nmf_method=cfg["nmf_method"],
            nmf_nrun=cfg["nmf_nrun"],
        )
    else:
        print("\n[reuse] Using precomputed feature object; skipping extract_features.")
        feature_obj = precomputed_features

    ############################################################
    # 3. Build modeling data frames, or reuse precomputed
    ############################################################

    if precomputed_model_dfs is None:
        model_dfs = build_feature_modeling_dfs(
            train_meta=data_obj["train_meta"],
            test_meta=data_obj["test_meta"],
            feature_result=feature_obj,
        )
    else:
        print("\n[reuse] Using precomputed model_dfs; skipping build_feature_modeling_dfs.")
        model_dfs = precomputed_model_dfs

    ############################################################
    # 4. Run MMIL according to task type
    ############################################################

    if cfg["task_type"] == "binary":
        mmil_result = run_binary_mmil(
            train_df=model_dfs["train_df"],
            test_df=model_dfs["test_df"],
            feature_cols=model_dfs["feature_cols"],
            backend_config=cfg["backend_config"],
            rho=cfg["rho"],
            max_iter=cfg["max_iter"],
            tol=cfg["tol"],
            mcem_samples=cfg["mcem_samples"],
            seed=cfg["seed"],
        )

    elif cfg["task_type"] == "categorical":
        mmil_result = run_categorical_mmil(
            train_df=model_dfs["train_df"],
            test_df=model_dfs["test_df"],
            feature_cols=model_dfs["feature_cols"],
            backend_config=cfg["backend_config"],
            class_levels=cfg["class_levels"],
            label_strength=cfg["label_strength"],
            max_iter=cfg["max_iter"],
            tol=cfg["tol"],
            mcem_samples=cfg["mcem_samples"],
            seed=cfg["seed"],
        )

    else:
        raise ValueError(f"Unsupported task_type: {cfg['task_type']}")

    ############################################################
    # 5. Save cell-level predictions and logs
    ############################################################

    saved = save_mmil_outputs(
        mmil_result=mmil_result,
        out_dir=cfg["experiment_out_dir"],
        pred_filename="cell_predictions.csv",
    )

    ############################################################
    # 6. Patient-level evaluation
    ############################################################

    eval_result = evaluate_predictions(
        pred_file=saved["pred_file"],
        out_dir=cfg["experiment_out_dir"],
        task_type=cfg["task_type"],
        class_levels=cfg["class_levels"],
        save_plots=cfg["save_plots"],
    )

    ############################################################
    # 7. Save experiment-level summary
    ############################################################

    summary_df = build_experiment_summary(
        config=cfg,
        eval_result=eval_result,
    )

    summary_file = Path(cfg["experiment_out_dir"]) / "experiment_summary.csv"

    summary_df.to_csv(summary_file, index=False)

    print("\nSaved experiment summary:")
    print(summary_file)

    print("\nExperiment completed:")
    print(cfg["experiment_id"])

    return {
        "config": cfg,
        "data": data_obj,
        "features": feature_obj,
        "model_dfs": model_dfs,
        "mmil_result": mmil_result,
        "eval_result": eval_result,
        "summary": summary_df,
        "files": {
            "prediction_file": saved["pred_file"],
            "summary_file": str(summary_file),
            "task_out_dir": cfg["task_out_dir"],
            "experiment_out_dir": cfg["experiment_out_dir"],
        },
    }


############################################################
# Command-line entry point
############################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one MMIL / MCEM experiment."
    )

    parser.add_argument(
        "config_file",
        help="Path to single experiment YAML config.",
    )

    args = parser.parse_args()

    run_one_experiment(
        config=args.config_file,
        r_dir="R",
        source_scripts=False,
    )


if __name__ == "__main__":
    main()