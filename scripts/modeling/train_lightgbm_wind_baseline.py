#!/usr/bin/env python3
"""Train LightGBM tabular baselines for ballast-preview wind forecasting.

This baseline uses the same chronological splits and train-only scaling as the
recurrent models. Each 12-step input sequence is flattened into one feature row.
The regression target is the future 6-step u/v sequence, optionally represented
as a residual relative to the latest observed scaled u/v. Event probabilities
are trained with one binary LightGBM classifier per event.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from train_ballast_lstm import (
    best_f1_thresholds,
    event_metrics,
    inverse_scale_uv,
    regression_metrics,
    shape_checks,
    uv_feature_indices,
)


SPLITS = ["train", "validation", "test"]


@dataclass
class LightGBMConfig:
    dataset_dir: str
    output_dir: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    threads: int
    seed: int
    residual_regression: bool
    train_sample_limit: int | None
    threshold_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_dwd_helgoland"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/lightgbm_tabular_residual"),
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=0.20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--threshold-mode", choices=["fixed_0.5", "best_f1"], default="best_f1")
    parser.add_argument("--no-residual-regression", action="store_true")
    return parser.parse_args()


def load_arrays(dataset_dir: Path, split: str) -> dict[str, np.ndarray]:
    return {
        "X": np.load(dataset_dir / f"X_{split}.npy", mmap_mode="r"),
        "y_uv": np.load(dataset_dir / f"y_uv_{split}.npy", mmap_mode="r"),
        "y_uv_raw": np.load(dataset_dir / f"y_uv_raw_{split}.npy", mmap_mode="r"),
        "y_speed_dir_raw": np.load(dataset_dir / f"y_speed_dir_raw_{split}.npy", mmap_mode="r"),
        "y_event": np.load(dataset_dir / f"y_event_{split}.npy", mmap_mode="r"),
    }


def load_dataset(dataset_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict, dict]:
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    scaler = json.loads((dataset_dir / "scaler_train.json").read_text(encoding="utf-8"))
    arrays = {split: load_arrays(dataset_dir, split) for split in SPLITS}
    shape_checks(arrays, metadata)
    return arrays, metadata, scaler


def flatten_features(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])


def regression_target(
    y_uv: np.ndarray,
    x: np.ndarray,
    residual_regression: bool,
    uv_indices: tuple[int, int],
) -> np.ndarray:
    target = np.asarray(y_uv, dtype=np.float32)
    if residual_regression:
        current_uv = np.asarray(x[:, -1, list(uv_indices)], dtype=np.float32)[:, None, :]
        target = target - current_uv
    return target.reshape(target.shape[0], target.shape[1] * target.shape[2])


def full_scaled_prediction(
    pred_flat: np.ndarray,
    x: np.ndarray,
    future_steps: int,
    residual_regression: bool,
    uv_indices: tuple[int, int],
) -> np.ndarray:
    pred = pred_flat.reshape(pred_flat.shape[0], future_steps, 2).astype(np.float32)
    if residual_regression:
        current_uv = np.asarray(x[:, -1, list(uv_indices)], dtype=np.float32)[:, None, :]
        pred = pred + current_uv
    return pred


def choose_train_rows(n: int, limit: int | None, seed: int) -> np.ndarray | None:
    if limit is None or limit <= 0 or limit >= n:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def fit_regressors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> list[lgb.LGBMRegressor]:
    models: list[lgb.LGBMRegressor] = []
    for target_idx in range(y_train.shape[1]):
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            subsample_freq=1,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            random_state=args.seed + target_idx,
            n_jobs=args.threads,
            verbosity=-1,
        )
        model.fit(
            x_train,
            y_train[:, target_idx],
            eval_set=[(x_val, y_val[:, target_idx])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        models.append(model)
        print(
            f"regression target {target_idx + 1}/{y_train.shape[1]} "
            f"best_iteration={model.best_iteration_}",
            flush=True,
        )
    return models


def fit_event_classifiers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> list[lgb.LGBMClassifier]:
    models: list[lgb.LGBMClassifier] = []
    for target_idx in range(y_train.shape[1]):
        positive_rate = float(np.mean(y_train[:, target_idx]))
        scale_pos_weight = (1.0 - positive_rate) / max(positive_rate, 1e-4)
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            subsample_freq=1,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            scale_pos_weight=min(scale_pos_weight, 50.0),
            random_state=args.seed + 1000 + target_idx,
            n_jobs=args.threads,
            verbosity=-1,
        )
        model.fit(
            x_train,
            y_train[:, target_idx],
            eval_set=[(x_val, y_val[:, target_idx])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        models.append(model)
        print(
            f"event target {target_idx + 1}/{y_train.shape[1]} "
            f"positive_rate={positive_rate:.4f} best_iteration={model.best_iteration_}",
            flush=True,
        )
    return models


def predict_regression(models: list[lgb.LGBMRegressor], x: np.ndarray) -> np.ndarray:
    pred = np.empty((x.shape[0], len(models)), dtype=np.float32)
    for target_idx, model in enumerate(models):
        pred[:, target_idx] = model.predict(x, num_iteration=model.best_iteration_).astype(np.float32)
    return pred


def predict_events(models: list[lgb.LGBMClassifier], x: np.ndarray) -> np.ndarray:
    probs = np.empty((x.shape[0], len(models)), dtype=np.float32)
    for target_idx, model in enumerate(models):
        probs[:, target_idx] = model.predict_proba(x, num_iteration=model.best_iteration_)[:, 1].astype(np.float32)
    return probs


def feature_names(metadata: dict) -> list[str]:
    names = []
    for step in range(metadata["history_steps"]):
        minutes_before = (metadata["history_steps"] - 1 - step) * metadata["input_resolution_minutes"]
        for col in metadata["feature_columns"]:
            names.append(f"t_minus_{minutes_before:03d}m__{col}")
    return names


def save_feature_importance(models: list, names: list[str], output_path: Path) -> None:
    rows = []
    for model_idx, model in enumerate(models):
        importance = model.booster_.feature_importance(importance_type="gain")
        total = float(np.sum(importance)) or 1.0
        top_idx = np.argsort(importance)[::-1][:50]
        for rank, feature_idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "model_index": model_idx,
                    "rank": rank,
                    "feature": names[feature_idx],
                    "gain": float(importance[feature_idx]),
                    "gain_share": float(importance[feature_idx] / total),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, metadata, scaler = load_dataset(args.dataset_dir)
    uv_indices = uv_feature_indices(metadata)
    residual_regression = not args.no_residual_regression
    train_limit = args.train_sample_limit if args.train_sample_limit > 0 else None

    selected_train_rows = choose_train_rows(arrays["train"]["X"].shape[0], train_limit, args.seed)
    if selected_train_rows is None:
        train_x_source = arrays["train"]["X"]
        train_y_uv_source = arrays["train"]["y_uv"]
        train_event_source = arrays["train"]["y_event"]
        train_rows_used = arrays["train"]["X"].shape[0]
    else:
        train_x_source = arrays["train"]["X"][selected_train_rows]
        train_y_uv_source = arrays["train"]["y_uv"][selected_train_rows]
        train_event_source = arrays["train"]["y_event"][selected_train_rows]
        train_rows_used = len(selected_train_rows)

    x_train = flatten_features(train_x_source)
    y_train = regression_target(
        train_y_uv_source,
        train_x_source,
        residual_regression,
        uv_indices,
    )
    y_event_train = np.asarray(train_event_source, dtype=np.int8)
    x_val = flatten_features(arrays["validation"]["X"])
    y_val = regression_target(arrays["validation"]["y_uv"], arrays["validation"]["X"], residual_regression, uv_indices)
    y_event_val = np.asarray(arrays["validation"]["y_event"], dtype=np.int8)
    x_test = flatten_features(arrays["test"]["X"])

    config = LightGBMConfig(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        threads=args.threads,
        seed=args.seed,
        residual_regression=residual_regression,
        train_sample_limit=train_limit,
        threshold_mode=args.threshold_mode,
    )
    (args.output_dir / "lightgbm_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "lightgbm_version": lgb.__version__,
                "train_rows_used": int(train_rows_used),
                "validation_rows": int(x_val.shape[0]),
                "test_rows": int(x_test.shape[0]),
                "flattened_feature_count": int(x_train.shape[1]),
                "metadata_source": str(args.dataset_dir / "metadata.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Train rows used: {train_rows_used} / {arrays['train']['X'].shape[0]}")
    print(f"Flattened features: {x_train.shape[1]}")
    print("Training regression models...")
    reg_models = fit_regressors(x_train, y_train, x_val, y_val, args)
    print("Training event classifiers...")
    event_models = fit_event_classifiers(x_train, y_event_train, x_val, y_event_val, args)

    joblib.dump({"regression": reg_models, "events": event_models, "config": asdict(config)}, args.output_dir / "lightgbm_models.joblib")
    names = feature_names(metadata)
    save_feature_importance(reg_models, names, args.output_dir / "lightgbm_regression_feature_importance_top50.csv")
    save_feature_importance(event_models, names, args.output_dir / "lightgbm_event_feature_importance_top50.csv")

    model_name = "lightgbm_tabular"
    if residual_regression:
        model_name = "residual_lightgbm_tabular"

    val_pred_scaled = full_scaled_prediction(
        predict_regression(reg_models, x_val),
        arrays["validation"]["X"],
        metadata["future_steps"],
        residual_regression,
        uv_indices,
    )
    test_pred_scaled = full_scaled_prediction(
        predict_regression(reg_models, x_test),
        arrays["test"]["X"],
        metadata["future_steps"],
        residual_regression,
        uv_indices,
    )
    val_pred_raw = inverse_scale_uv(val_pred_scaled, scaler)
    test_pred_raw = inverse_scale_uv(test_pred_scaled, scaler)

    reg_rows = []
    reg_rows.extend(
        regression_metrics(
            "validation",
            model_name,
            val_pred_raw,
            np.asarray(arrays["validation"]["y_uv_raw"]),
            np.asarray(arrays["validation"]["y_speed_dir_raw"]),
        )
    )
    reg_rows.extend(
        regression_metrics(
            "test",
            model_name,
            test_pred_raw,
            np.asarray(arrays["test"]["y_uv_raw"]),
            np.asarray(arrays["test"]["y_speed_dir_raw"]),
        )
    )
    reg_metrics = pd.DataFrame(reg_rows)
    reg_metrics.to_csv(args.output_dir / "lightgbm_regression_metrics.csv", index=False)

    val_event_prob = predict_events(event_models, x_val)
    test_event_prob = predict_events(event_models, x_test)
    if args.threshold_mode == "best_f1":
        thresholds = best_f1_thresholds(np.asarray(arrays["validation"]["y_event"]), val_event_prob)
    else:
        thresholds = np.full(arrays["train"]["y_event"].shape[1], 0.5, dtype=np.float32)

    event_rows = []
    event_rows.extend(
        event_metrics(
            "validation",
            model_name,
            np.asarray(arrays["validation"]["y_event"]),
            val_event_prob,
            thresholds,
            metadata["event_columns"],
        )
    )
    event_rows.extend(
        event_metrics(
            "test",
            model_name,
            np.asarray(arrays["test"]["y_event"]),
            test_event_prob,
            thresholds,
            metadata["event_columns"],
        )
    )
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(args.output_dir / "lightgbm_event_metrics.csv", index=False)
    (args.output_dir / "lightgbm_event_thresholds.json").write_text(
        json.dumps(
            {
                "threshold_mode": args.threshold_mode,
                "thresholds": {name: float(thresholds[idx]) for idx, name in enumerate(metadata["event_columns"])},
                "selected_on": "validation",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "model": model_name,
        "regression_metrics_csv": str(args.output_dir / "lightgbm_regression_metrics.csv"),
        "event_metrics_csv": str(args.output_dir / "lightgbm_event_metrics.csv"),
        "models": str(args.output_dir / "lightgbm_models.joblib"),
    }
    (args.output_dir / "lightgbm_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nRegression metrics")
    print(reg_metrics.to_string(index=False))
    print("\nEvent metrics")
    print(event_frame.to_string(index=False))


if __name__ == "__main__":
    main()
