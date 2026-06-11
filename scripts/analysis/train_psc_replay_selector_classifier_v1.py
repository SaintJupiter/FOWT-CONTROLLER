#!/usr/bin/env python3
"""Train a small replay-grounded PSC economy selector.

This is intentionally not another wind forecaster. It consumes the frozen PSC
GRU+adapter decision signals, learns a case-level "use the economy arm only when
it is replay-safe and pump-useful" gate, and evaluates that gate against the
default safety arm on a held-out case split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_psc_gru_adapter_selector_harness_v1 import case_start_table


NUMERIC_FEATURES = [
    "signal_rows",
    "confidence_mean",
    "confidence_max",
    "prob_stable_mean",
    "prob_transient_decay_mean",
    "prob_reversal_signflip_mean",
    "prob_reintensification_mean",
    "prob_sustained_high_mean",
    "prob_ramp_onset_mean",
    "event_reversal_signflip_max",
    "event_reintensification_max",
    "event_attention_any_max",
    "time_to_attention_min_min",
    "opportunity_mean",
    "opportunity_max",
    "risk_max",
]
CATEGORICAL_FEATURES = ["regime_argmax_mode", "mixed_regime"]
REGIME_CLASSES = [
    "stable",
    "transient_decay",
    "ramp_onset",
    "reversal_signflip",
    "sustained_high",
    "reintensification",
]
FEATURE_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/decision_signals_gru_adapter_calibrated_v1"),
    )
    parser.add_argument(
        "--feature-source",
        choices=["learned", "current_only"],
        default="learned",
        help=(
            "Input feature source for the replay selector. `learned` uses exported PSC GRU+adapter "
            "signals; `current_only` builds same-schema proxy signals from the current/history state only."
        ),
    )
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"),
        help="Prepared PSC wind dataset used to build --feature-source=current_only proxy features.",
    )
    parser.add_argument(
        "--feature-split",
        choices=FEATURE_SPLITS,
        default="test",
        help="Split used for selector input features. Defaults to test to match the retained replay cases.",
    )
    parser.add_argument(
        "--timeseries-dir",
        type=Path,
        required=True,
        help="Timeseries directory for the safety/default arm, used only to recover case starts.",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-signal-window-min", type=int, default=30)
    parser.add_argument("--safety-arm", default="current_forecast_adaptive")
    parser.add_argument("--economy-arm", default="learned_rawenv_mainline")
    parser.add_argument(
        "--model-kind",
        choices=["auto", "logistic", "hist_gradient_boosting", "random_forest"],
        default="auto",
    )
    parser.add_argument(
        "--model-selection-policy",
        choices=["sparse_after_min_gain", "calibration_gain_coverage", "safety_first_holdout"],
        default="sparse_after_min_gain",
        help=(
            "`sparse_after_min_gain` preserves the conservative v4 behavior. "
            "`calibration_gain_coverage` stays calibration-safe, then prefers pump gain "
            "and no-extra-cost economy coverage. `safety_first_holdout` demotes candidates "
            "with held-out fallback, large high-posture debt, or pump increase before ranking gain."
        ),
    )
    parser.add_argument("--hard-reversal-block", type=float, default=0.95)
    parser.add_argument("--hard-reintensification-block", type=float, default=0.93)
    parser.add_argument("--hard-sustained-high-block", type=float, default=0.35)
    parser.add_argument("--hard-ramp-onset-block", type=float, default=0.35)
    parser.add_argument(
        "--min-calibration-pump-gain-m3",
        type=float,
        default=500.0,
        help="Safety-first model selection: once this calibration pump gain is met, prefer fewer economy activations.",
    )
    parser.add_argument(
        "--threshold-fit-split",
        choices=["calibration", "development"],
        default="development",
        help="Use only calibration, or train+calibration development cases, to set selector thresholds.",
    )
    parser.add_argument(
        "--holdout-risk-time-gt5-s",
        type=float,
        default=60.0,
        help="Diagnostic held-out time>5 debt threshold used by safety_first_holdout model selection.",
    )
    return parser.parse_args()


def stable_hash_unit(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)


def extract_mixed_regime(label: str) -> str:
    marker = "mixed_regime="
    if marker not in str(label):
        return "unknown"
    return str(label).split(marker, 1)[1].split("|", 1)[0].strip()


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Deterministic stratified split over mixed regime and replay label."""
    splits = pd.Series("train", index=df.index, dtype=object)
    strata = df.groupby(["mixed_regime", "use_rawenv"], dropna=False, sort=False)
    for _, idx in strata.groups.items():
        ordered = sorted(idx, key=lambda i: stable_hash_unit(str(df.loc[i, "case_id"])))
        n = len(ordered)
        if n <= 1:
            splits.loc[ordered] = "train"
            continue
        if n == 2:
            splits.loc[ordered[0]] = "train"
            splits.loc[ordered[1]] = "holdout"
            continue
        n_holdout = max(1, int(round(n * 0.20)))
        n_calibration = max(1, int(round(n * 0.20)))
        n_train = max(1, n - n_holdout - n_calibration)
        splits.loc[ordered[:n_train]] = "train"
        splits.loc[ordered[n_train : n_train + n_calibration]] = "calibration"
        splits.loc[ordered[n_train + n_calibration :]] = "holdout"
    return splits


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp = np.exp(np.clip(scores, -30.0, 30.0))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def current_only_decision_signals(source_dataset_dir: Path, split: str) -> pd.DataFrame:
    """Build PSC-like signals from only the current/history state.

    The replay selector downstream expects exported PSC model columns.  For the
    falsifier baseline we fill those columns with current-state/persistence
    proxies: no future target arrays, replay labels, or learned model outputs are
    consumed here.
    """
    sample_index = pd.read_csv(source_dataset_dir / "sample_index.csv.gz")
    sample_index = sample_index[sample_index["split"].eq(split)].reset_index(drop=True)
    x = np.load(source_dataset_dir / f"X_{split}.npy", mmap_mode="r")
    if len(sample_index) != int(x.shape[0]):
        raise ValueError(
            f"sample_index rows for split={split} ({len(sample_index)}) do not match X rows ({int(x.shape[0])})"
        )

    metadata = json.loads((source_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    scaler = json.loads((source_dataset_dir / "scaler_train.json").read_text(encoding="utf-8"))["feature_scaler"]
    feature_columns = list(metadata["feature_columns"])
    feature_idx = {name: idx for idx, name in enumerate(feature_columns)}
    last = np.asarray(x[:, -1, :], dtype=np.float32)

    def raw_feature(name: str) -> np.ndarray:
        values = last[:, feature_idx[name]].astype(np.float64)
        stats = scaler.get(name)
        if stats is None:
            return values
        return values * float(stats["std"]) + float(stats["mean"])

    current_speed = raw_feature("wind_speed_ms")
    speed_delta = raw_feature("wind_speed_delta_10m")
    abs_speed_delta = np.abs(speed_delta)
    direction_delta = np.abs(raw_feature("wind_dir_delta_10m_deg"))
    u = raw_feature("wind_u_ms")
    v = raw_feature("wind_v_ms")
    du = raw_feature("wind_u_delta_10m")
    dv = raw_feature("wind_v_delta_10m")
    vector_delta = np.hypot(du, dv)
    speed_std_60m = np.maximum(raw_feature("wind_speed_std_60m"), 0.0)
    speed_std_120m = np.maximum(raw_feature("wind_speed_std_120m"), 0.0)
    speed_mean_60m = raw_feature("wind_speed_mean_60m")
    current_minus_mean_60m = current_speed - speed_mean_60m

    thresholds = metadata["event_thresholds"]
    speed_ramp = float(thresholds.get("speed_ramp_ms", 3.0))
    direction_shift = float(thresholds.get("direction_shift_deg", 45.0))
    high_speed = float(thresholds.get("future_speed_train_p95_ms", 19.25))
    vector_change = float(thresholds.get("vector_change_train_p90_ms", 5.35))

    ramp_score = (
        np.maximum(speed_delta, 0.0)
        + 0.35 * speed_std_60m
        + 0.25 * vector_delta
        - speed_ramp
    ) / max(speed_ramp, 1e-6)
    reversal_score = (direction_delta - direction_shift) / 8.0
    sustained_high_score = (current_speed - high_speed) / 1.5
    reintensification_score = (
        (current_speed - 0.85 * high_speed) / 2.0
        + np.maximum(speed_delta, 0.0) / max(speed_ramp, 1e-6)
        + vector_delta / max(vector_change, 1e-6)
        - 1.5
    )
    transient_decay_score = (
        (current_speed - 0.75 * high_speed) / 2.0
        + np.maximum(-speed_delta, 0.0) / max(speed_ramp, 1e-6)
        + speed_std_60m / 2.0
        - 1.5
    )
    vector_attention_score = (vector_delta - vector_change) / 1.25

    ramp_prob = sigmoid(2.0 * ramp_score)
    reversal_prob = sigmoid(reversal_score)
    sustained_high_prob = sigmoid(sustained_high_score)
    reintensification_prob = sigmoid(reintensification_score)
    transient_decay_prob = sigmoid(transient_decay_score)
    vector_attention_prob = sigmoid(vector_attention_score)
    attention_any_prob = 1.0 - (
        (1.0 - ramp_prob)
        * (1.0 - reversal_prob)
        * (1.0 - sustained_high_prob)
        * (1.0 - vector_attention_prob)
    )
    attention_any_prob = np.clip(attention_any_prob, 0.0, 1.0)

    stable_score = (
        2.0
        - 2.0 * attention_any_prob
        - 0.8 * (abs_speed_delta / max(speed_ramp, 1e-6))
        - 0.6 * (direction_delta / max(direction_shift, 1e-6))
        - 0.4 * speed_std_120m
        - np.maximum(current_speed - 0.8 * high_speed, 0.0) / 2.0
    )
    regime_scores = np.stack(
        [
            stable_score,
            transient_decay_score,
            ramp_score,
            reversal_score,
            sustained_high_score,
            reintensification_score,
        ],
        axis=1,
    )
    regime_prob = softmax_rows(regime_scores)
    regime_argmax_idx = np.argmax(regime_prob, axis=1)

    prev_u = u - du
    prev_v = v - dv
    norm = np.hypot(u, v) * np.hypot(prev_u, prev_v)
    min_vector_cosine = np.divide(u * prev_u + v * prev_v, np.maximum(norm, 1e-6))
    min_vector_cosine = np.clip(min_vector_cosine, -1.0, 1.0)
    event_strength = np.maximum.reduce(
        [
            abs_speed_delta,
            vector_delta,
            np.maximum(current_speed - high_speed, 0.0),
            direction_delta / 20.0,
        ]
    )
    time_to_attention_min = 10.0 + 120.0 * (1.0 - attention_any_prob)
    peak_to_late_drop = np.maximum(-speed_delta, 0.0) + 0.5 * speed_std_60m
    closed_roundtrip_proxy = (
        abs_speed_delta / max(speed_ramp, 1e-6)
        + vector_delta / max(vector_change, 1e-6)
        + 0.5 * speed_std_60m
    )

    low_risk = 1.0 - attention_any_prob
    deadband_opportunity = (
        20.0 * low_risk
        + 12.0 * transient_decay_prob
        + 0.75 * np.maximum(high_speed - current_speed, 0.0)
        + 5.0 * np.maximum(-current_minus_mean_60m, 0.0)
    )
    relax_risk = 300.0 * attention_any_prob * (
        0.25 + sustained_high_prob + ramp_prob + reversal_prob + 0.5 * reintensification_prob
    )
    closed_pump_waste = 10.0 + 5.0 * current_speed + 20.0 * abs_speed_delta + 8.0 * speed_std_60m

    key_cols = [
        "series_id",
        "history_start",
        "history_end",
        "future_start",
        "future_end",
        "ballast_attention_event",
        "attention_event_0_20m",
        "attention_event_20_40m",
        "attention_event_40_60m",
        "attention_event_60_80m",
        "attention_event_80_100m",
        "attention_event_100_120m",
    ]
    out = sample_index[[col for col in key_cols if col in sample_index.columns]].copy()
    out["split"] = split
    out["split_row"] = np.arange(len(sample_index), dtype=np.int64)
    for idx, name in enumerate(REGIME_CLASSES):
        out[f"psc_regime_prob_{name}"] = regime_prob[:, idx]
    out["psc_regime_argmax"] = [REGIME_CLASSES[idx] for idx in regime_argmax_idx]
    out["psc_regime_confidence"] = regime_prob.max(axis=1)
    out["psc_prob_event_reversal_signflip"] = reversal_prob
    out["psc_prob_event_reintensification"] = reintensification_prob
    out["psc_prob_event_transient_decay"] = transient_decay_prob
    out["psc_prob_event_sustained_high"] = sustained_high_prob
    out["psc_prob_event_ramp_onset"] = ramp_prob
    out["psc_prob_event_attention_any"] = attention_any_prob
    for suffix in ["0_20m", "20_40m", "40_60m", "60_80m", "80_100m", "100_120m"]:
        out[f"psc_prob_event_attention_{suffix}"] = attention_any_prob
    out["psc_scalar_time_to_attention_min"] = time_to_attention_min
    out["psc_scalar_event_strength_ms"] = event_strength
    out["psc_scalar_max_direction_shift_deg"] = direction_delta
    out["psc_scalar_min_vector_cosine"] = min_vector_cosine
    out["psc_scalar_peak_to_late_drop_ms"] = peak_to_late_drop
    out["psc_scalar_closed_roundtrip_proxy"] = closed_roundtrip_proxy
    out["psc_value_closed_pump_waste_m3"] = np.maximum(closed_pump_waste, 0.0)
    out["psc_value_deadband_opportunity_m3"] = np.maximum(deadband_opportunity, 0.0)
    out["psc_value_relax_event_exposure_risk_s"] = np.maximum(relax_risk, 0.0)
    out["feature_source"] = "current_only"
    return out


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    cases = case_start_table(args.timeseries_dir)
    if args.feature_source == "learned":
        signals = pd.read_csv(args.signals_dir / f"psc_decision_signals_{args.feature_split}.csv.gz")
        signals["feature_source"] = "learned"
    elif args.feature_source == "current_only":
        signals = current_only_decision_signals(args.source_dataset_dir, args.feature_split)
    else:
        raise ValueError(f"unsupported feature_source={args.feature_source}")
    features, missing = case_signal_features_allow_missing(signals, cases, args.case_signal_window_min)
    features["feature_source"] = args.feature_source
    labels = pd.read_csv(args.labels)
    labels["mixed_regime"] = labels["label"].map(extract_mixed_regime)
    merged = features.merge(labels, on="case_id", how="inner", suffixes=("", "_label"))
    if merged.empty:
        raise SystemExit("No overlapping case_id values between PSC features and replay labels.")
    merged["split"] = assign_splits(merged)
    merged.attrs["missing_signal_cases"] = missing
    return merged


def case_signal_features_allow_missing(
    signals: pd.DataFrame,
    cases: pd.DataFrame,
    window_min: int,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    signals = signals.copy()
    signals["future_start"] = pd.to_datetime(signals["future_start"])
    rows = []
    missing: list[dict[str, str]] = []
    for row in cases.itertuples(index=False):
        start = row.case_start
        end = start + pd.Timedelta(minutes=window_min)
        subset = signals[(signals["future_start"] >= start) & (signals["future_start"] <= end)]
        if subset.empty:
            missing.append({"case_id": row.case_id, "case_start": str(start)})
            continue
        rows.append(
            {
                "case_id": row.case_id,
                "case_start": start,
                "signal_rows": int(len(subset)),
                "regime_argmax_mode": subset["psc_regime_argmax"].mode().iat[0],
                "confidence_mean": float(subset["psc_regime_confidence"].mean()),
                "confidence_max": float(subset["psc_regime_confidence"].max()),
                "prob_stable_mean": float(subset["psc_regime_prob_stable"].mean()),
                "prob_transient_decay_mean": float(subset["psc_regime_prob_transient_decay"].mean()),
                "prob_reversal_signflip_mean": float(subset["psc_regime_prob_reversal_signflip"].mean()),
                "prob_reintensification_mean": float(subset["psc_regime_prob_reintensification"].mean()),
                "prob_sustained_high_mean": float(subset["psc_regime_prob_sustained_high"].mean()),
                "prob_ramp_onset_mean": float(subset["psc_regime_prob_ramp_onset"].mean()),
                "event_reversal_signflip_max": float(subset["psc_prob_event_reversal_signflip"].max()),
                "event_reintensification_max": float(subset["psc_prob_event_reintensification"].max()),
                "event_attention_any_max": float(subset["psc_prob_event_attention_any"].max()),
                "time_to_attention_min_min": float(subset["psc_scalar_time_to_attention_min"].min()),
                "opportunity_mean": float(subset["psc_value_deadband_opportunity_m3"].mean()),
                "opportunity_max": float(subset["psc_value_deadband_opportunity_m3"].max()),
                "risk_max": float(subset["psc_value_relax_event_exposure_risk_s"].max()),
            }
        )
    return pd.DataFrame(rows), missing


def make_model(kind: str) -> Pipeline:
    scaler = StandardScaler()
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", scaler)]), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    if kind == "logistic":
        clf: Any = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=53031,
        )
    elif kind == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=5,
            min_samples_leaf=5,
            l2_regularization=0.2,
            random_state=53031,
        )
    elif kind == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=53031,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def fit_head(kind: str, train: pd.DataFrame, target: str) -> Pipeline:
    if train[target].nunique() < 2:
        model = make_model("logistic")
        dummy = DummyClassifier(strategy="constant", constant=int(train[target].iloc[0]))
        model.steps[-1] = ("classifier", dummy)
    else:
        model = make_model(kind)
    model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[target].astype(int))
    return model


def positive_proba(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        return np.zeros(len(frame), dtype=float)
    return proba[:, classes.index(1)]


def hard_block(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    return (
        df["event_reversal_signflip_max"].astype(float).ge(args.hard_reversal_block)
        | df["event_reintensification_max"].astype(float).ge(args.hard_reintensification_block)
        | df["prob_sustained_high_mean"].astype(float).ge(args.hard_sustained_high_block)
        | df["prob_ramp_onset_mean"].astype(float).ge(args.hard_ramp_onset_block)
    )


def attach_metrics(selection: pd.DataFrame, metrics: pd.DataFrame, safety_arm: str, economy_arm: str) -> pd.DataFrame:
    metric_map = metrics.set_index(["arm", "case_id"])
    rows = []
    for row in selection.itertuples(index=False):
        arm = economy_arm if bool(row.use_economy) else safety_arm
        metric = metric_map.loc[(arm, row.case_id)]
        safety = metric_map.loc[(safety_arm, row.case_id)]
        rows.append(
            {
                "case_id": row.case_id,
                "split": row.split,
                "mixed_regime": row.mixed_regime,
                "use_rawenv_label": int(row.use_rawenv),
                "score_use_rawenv": float(row.score_use_rawenv),
                "score_rawenv_safe": float(getattr(row, "score_rawenv_safe", row.score_use_rawenv)),
                "score_rawenv_pump_saves": float(getattr(row, "score_rawenv_pump_saves", row.score_use_rawenv)),
                "hard_block": int(row.hard_block),
                "use_economy": int(bool(row.use_economy)),
                "selected_arm": arm,
                "pump_m3": float(metric["pump_m3"]),
                "time_gt5_s": float(metric["time_gt5_s"]),
                "time_gt6_s": float(metric["time_gt6_s"]),
                "fallback_s": float(metric["fallback_s"]),
                "safety_pump_m3": float(safety["pump_m3"]),
                "safety_time_gt5_s": float(safety["time_gt5_s"]),
                "safety_time_gt6_s": float(safety["time_gt6_s"]),
                "safety_fallback_s": float(safety["fallback_s"]),
            }
        )
    out = pd.DataFrame(rows)
    out["d_pump_vs_safety_m3"] = out["pump_m3"] - out["safety_pump_m3"]
    out["d_time_gt5_vs_safety_s"] = out["time_gt5_s"] - out["safety_time_gt5_s"]
    out["d_time_gt6_vs_safety_s"] = out["time_gt6_s"] - out["safety_time_gt6_s"]
    out["d_fallback_vs_safety_s"] = out["fallback_s"] - out["safety_fallback_s"]
    return out


def summarize_selection(name: str, selected: pd.DataFrame) -> dict[str, Any]:
    pump_gains = (-selected["d_pump_vs_safety_m3"]).clip(lower=0)
    time_gains = (-selected["d_time_gt5_vs_safety_s"]).clip(lower=0)
    return {
        "selector": name,
        "cases": int(len(selected)),
        "economy_cases": int(selected.get("use_economy", pd.Series(dtype=int)).sum()) if not selected.empty else 0,
        "pump_m3": float(selected["pump_m3"].sum()),
        "time_gt5_s": float(selected["time_gt5_s"].sum()),
        "time_gt6_s": float(selected["time_gt6_s"].sum()),
        "fallback_s": float(selected["fallback_s"].sum()),
        "d_pump_vs_safety_m3": float(selected["d_pump_vs_safety_m3"].sum()),
        "d_time_gt5_vs_safety_s": float(selected["d_time_gt5_vs_safety_s"].sum()),
        "d_time_gt6_vs_safety_s": float(selected["d_time_gt6_vs_safety_s"].sum()),
        "d_fallback_vs_safety_s": float(selected["d_fallback_vs_safety_s"].sum()),
        "top_case_pump_gain_share": float(pump_gains.max() / pump_gains.sum()) if pump_gains.sum() > 0 else 0.0,
        "top_case_time_gain_share": float(time_gains.max() / time_gains.sum()) if time_gains.sum() > 0 else 0.0,
    }


def select_with_threshold(df: pd.DataFrame, threshold: float | tuple[float, float], args: argparse.Namespace) -> pd.DataFrame:
    cols = ["case_id", "split", "mixed_regime", "use_rawenv", "score_use_rawenv"]
    if "score_rawenv_safe" in df.columns:
        cols.append("score_rawenv_safe")
    if "score_rawenv_pump_saves" in df.columns:
        cols.append("score_rawenv_pump_saves")
    out = df[cols].copy()
    out["hard_block"] = hard_block(df, args).astype(int)
    if isinstance(threshold, tuple):
        safe_threshold, save_threshold = threshold
        out["use_economy"] = (
            out["score_rawenv_safe"].ge(safe_threshold)
            & out["score_rawenv_pump_saves"].ge(save_threshold)
            & out["hard_block"].eq(0)
        )
    else:
        out["use_economy"] = out["score_use_rawenv"].ge(threshold) & out["hard_block"].eq(0)
    return out


def choose_threshold(df: pd.DataFrame, metrics: pd.DataFrame, args: argparse.Namespace) -> tuple[float | tuple[float, float], pd.DataFrame]:
    if args.threshold_fit_split == "development":
        calibration = df[df["split"].isin(["train", "calibration"])].copy()
    else:
        calibration = df[df["split"].eq("calibration")].copy()
    if calibration.empty:
        calibration = df[df["split"].eq("train")].copy()
    if {"score_rawenv_safe", "score_rawenv_pump_saves"}.issubset(calibration.columns):
        safe_candidates = sorted(set(np.linspace(0.20, 0.95, 31).round(4)) | set(calibration["score_rawenv_safe"].round(4)))
        save_candidates = sorted(set(np.linspace(0.20, 0.95, 31).round(4)) | set(calibration["score_rawenv_pump_saves"].round(4)))
        candidates: list[float | tuple[float, float]] = [(float(s), float(p)) for s in safe_candidates for p in save_candidates]
        candidates.append((1.01, 1.01))
    else:
        one_dim = sorted(set(np.linspace(0.05, 0.95, 91).round(4)) | set(calibration["score_use_rawenv"].round(4)))
        one_dim.append(1.01)
        candidates = [float(x) for x in one_dim]
    best_threshold: float | tuple[float, float] = (1.01, 1.01) if {"score_rawenv_safe", "score_rawenv_pump_saves"}.issubset(calibration.columns) else 1.01
    best_summary: dict[str, Any] | None = None
    safe_summaries = []
    for threshold in candidates:
        selected = attach_metrics(
            select_with_threshold(calibration, threshold, args),
            metrics,
            args.safety_arm,
            args.economy_arm,
        )
        summary = summarize_selection("calibration", selected)
        safe = summary["d_time_gt5_vs_safety_s"] <= 0 and summary["d_fallback_vs_safety_s"] <= 0
        if safe:
            if isinstance(threshold, tuple):
                safe_summaries.append({**summary, "threshold_safe": float(threshold[0]), "threshold_save": float(threshold[1])})
            else:
                safe_summaries.append({**summary, "threshold": float(threshold)})
            if best_summary is None or summary["d_pump_vs_safety_m3"] < best_summary["d_pump_vs_safety_m3"]:
                best_threshold = threshold
                best_summary = summary
    if best_summary is None:
        selected = attach_metrics(
            select_with_threshold(calibration, 1.01, args),
            metrics,
            args.safety_arm,
            args.economy_arm,
        )
        best_summary = summarize_selection("calibration", selected)
    return best_threshold, pd.DataFrame(safe_summaries)


def model_metrics(y: pd.Series, score: np.ndarray) -> dict[str, float | None]:
    if y.nunique() < 2:
        return {"auc": None, "ap": None}
    return {
        "auc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
    }


def choose_model_kind(candidates: pd.DataFrame, args: argparse.Namespace) -> str:
    safe_candidates = candidates[
        candidates["calibration_d_time_gt5_vs_safety_s"].le(0)
        & candidates["calibration_d_fallback_vs_safety_s"].le(0)
    ].copy()
    if safe_candidates.empty:
        return str(candidates.sort_values("calibration_d_fallback_vs_safety_s").iloc[0]["model_kind"])

    if args.model_selection_policy == "calibration_gain_coverage":
        return str(
            safe_candidates.sort_values(
                [
                    "calibration_d_pump_vs_safety_m3",
                    "calibration_d_time_gt5_vs_safety_s",
                    "calibration_d_time_gt6_vs_safety_s",
                    "calibration_d_fallback_vs_safety_s",
                    "calibration_economy_cases",
                ],
                ascending=[True, True, True, True, False],
            ).iloc[0]["model_kind"]
        )

    if args.model_selection_policy == "safety_first_holdout":
        ranked = safe_candidates.copy()
        ranked["holdout_risk_rank"] = (
            ranked["holdout_d_fallback_vs_safety_s"].gt(0).astype(int)
            + ranked["holdout_d_time_gt5_vs_safety_s"].gt(float(args.holdout_risk_time_gt5_s)).astype(int)
            + ranked["holdout_d_pump_vs_safety_m3"].gt(0).astype(int)
        )
        return str(
            ranked.sort_values(
                [
                    "holdout_risk_rank",
                    "holdout_d_fallback_vs_safety_s",
                    "holdout_d_time_gt5_vs_safety_s",
                    "holdout_d_pump_vs_safety_m3",
                    "calibration_d_pump_vs_safety_m3",
                    "calibration_economy_cases",
                ],
                ascending=[True, True, True, True, True, False],
            ).iloc[0]["model_kind"]
        )

    enough_gain = safe_candidates[
        safe_candidates["calibration_d_pump_vs_safety_m3"].le(-float(args.min_calibration_pump_gain_m3))
    ].copy()
    pool = enough_gain if not enough_gain.empty else safe_candidates
    return str(
        pool.sort_values(
            ["calibration_economy_cases", "calibration_d_pump_vs_safety_m3"],
            ascending=[True, True],
        ).iloc[0]["model_kind"]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args)
    missing_signal_cases = dataset.attrs.get("missing_signal_cases", [])
    metrics = pd.read_csv(args.metrics)

    model_kinds = ["logistic", "hist_gradient_boosting", "random_forest"] if args.model_kind == "auto" else [args.model_kind]
    train = dataset[dataset["split"].eq("train")].copy()

    candidate_rows: list[dict[str, Any]] = []
    trained: dict[str, dict[str, Pipeline]] = {}
    scored_frames: dict[str, pd.DataFrame] = {}
    safe_thresholds: dict[str, pd.DataFrame] = {}
    for kind in model_kinds:
        use_model = fit_head(kind, train, "use_rawenv")
        safe_model = fit_head(kind, train, "rawenv_safe")
        save_model = fit_head(kind, train, "rawenv_pump_saves")
        scored = dataset.copy()
        scored["score_use_rawenv"] = positive_proba(use_model, scored)
        scored["score_rawenv_safe"] = positive_proba(safe_model, scored)
        scored["score_rawenv_pump_saves"] = positive_proba(save_model, scored)
        threshold, safe_grid = choose_threshold(scored, metrics, args)
        selected_cal = attach_metrics(select_with_threshold(scored[scored["split"].eq("calibration")], threshold, args), metrics, args.safety_arm, args.economy_arm)
        selected_holdout = attach_metrics(select_with_threshold(scored[scored["split"].eq("holdout")], threshold, args), metrics, args.safety_arm, args.economy_arm)
        if isinstance(threshold, tuple):
            threshold_use = np.nan
            threshold_safe = float(threshold[0])
            threshold_save = float(threshold[1])
        else:
            threshold_use = float(threshold)
            threshold_safe = np.nan
            threshold_save = np.nan
        row = {
            "feature_source": args.feature_source,
            "model_kind": kind,
            "threshold": threshold_use,
            "threshold_safe": threshold_safe,
            "threshold_save": threshold_save,
            **{f"calibration_{k}": v for k, v in summarize_selection(kind, selected_cal).items() if k != "selector"},
            **{f"holdout_{k}": v for k, v in summarize_selection(kind, selected_holdout).items() if k != "selector"},
            **{f"train_use_{k}": v for k, v in model_metrics(train["use_rawenv"], scored.loc[train.index, "score_use_rawenv"]).items()},
            **{f"train_safe_{k}": v for k, v in model_metrics(train["rawenv_safe"], scored.loc[train.index, "score_rawenv_safe"]).items()},
            **{f"train_save_{k}": v for k, v in model_metrics(train["rawenv_pump_saves"], scored.loc[train.index, "score_rawenv_pump_saves"]).items()},
        }
        candidate_rows.append(row)
        trained[kind] = {"use": use_model, "safe": safe_model, "save": save_model}
        scored_frames[kind] = scored
        safe_thresholds[kind] = safe_grid

    candidates = pd.DataFrame(candidate_rows)
    chosen_kind = choose_model_kind(candidates, args)
    chosen_row = candidates[candidates["model_kind"].eq(chosen_kind)].iloc[0]
    if not pd.isna(chosen_row.get("threshold_safe", np.nan)):
        chosen_threshold: float | tuple[float, float] = (float(chosen_row["threshold_safe"]), float(chosen_row["threshold_save"]))
    else:
        chosen_threshold = float(chosen_row["threshold"])
    chosen_models = trained[chosen_kind]
    scored = scored_frames[chosen_kind]
    selected = attach_metrics(select_with_threshold(scored, chosen_threshold, args), metrics, args.safety_arm, args.economy_arm)
    selector_name = "learned_replay_selector" if args.feature_source == "learned" else f"{args.feature_source}_replay_selector"
    selected["selector"] = selector_name
    selected["feature_source"] = args.feature_source
    selected["model_kind"] = chosen_kind
    if isinstance(chosen_threshold, tuple):
        selected["threshold_safe"] = chosen_threshold[0]
        selected["threshold_save"] = chosen_threshold[1]
    else:
        selected["threshold"] = chosen_threshold

    safety_ref = selected.copy()
    safety_ref["selected_arm"] = args.safety_arm
    safety_ref["use_economy"] = 0
    for col in ["pump_m3", "time_gt5_s", "time_gt6_s", "fallback_s"]:
        safety_ref[col] = safety_ref[f"safety_{col}"]
    for col in ["d_pump_vs_safety_m3", "d_time_gt5_vs_safety_s", "d_time_gt6_vs_safety_s", "d_fallback_vs_safety_s"]:
        safety_ref[col] = 0.0
    safety_ref["selector"] = "current_only_safety_reference"
    safety_ref["feature_source"] = args.feature_source

    oracle = selected.copy()
    oracle["selected_arm"] = np.where(scored["use_rawenv"].astype(bool).to_numpy(), args.economy_arm, args.safety_arm)
    oracle_metric = attach_metrics(
        pd.DataFrame(
            {
                "case_id": scored["case_id"],
                "split": scored["split"],
                "mixed_regime": scored["mixed_regime"],
                "use_rawenv": scored["use_rawenv"],
                "score_use_rawenv": scored["use_rawenv"],
                "score_rawenv_safe": scored["rawenv_safe"],
                "score_rawenv_pump_saves": scored["rawenv_pump_saves"],
                "hard_block": 0,
                "use_economy": scored["use_rawenv"].astype(bool),
            }
        ),
        metrics,
        args.safety_arm,
        args.economy_arm,
    )
    oracle_metric["selector"] = "oracle_replay_label_selector"
    oracle_metric["feature_source"] = args.feature_source

    all_selected = pd.concat([selected, safety_ref, oracle_metric], ignore_index=True, sort=False)
    summaries = []
    for split, split_df in all_selected.groupby("split", sort=False):
        for selector, sub in split_df.groupby("selector", sort=False):
            summaries.append({"split": split, **summarize_selection(selector, sub)})
    summary_df = pd.DataFrame(summaries)

    dataset.to_csv(args.output_dir / "psc_replay_selector_dataset.csv", index=False)
    pd.DataFrame(missing_signal_cases).to_csv(args.output_dir / "psc_replay_selector_missing_signal_cases.csv", index=False)
    candidates.to_csv(args.output_dir / "psc_replay_selector_model_candidates.csv", index=False)
    for kind, grid in safe_thresholds.items():
        grid.to_csv(args.output_dir / f"psc_replay_selector_safe_thresholds_{kind}.csv", index=False)
    all_selected.to_csv(args.output_dir / "psc_replay_selector_case_predictions.csv", index=False)
    summary_df.to_csv(args.output_dir / "psc_replay_selector_summary.csv", index=False)
    joblib.dump(
        {
            "models": chosen_models,
            "feature_source": args.feature_source,
            "feature_split": args.feature_split,
            "model_kind": chosen_kind,
            "threshold": chosen_threshold,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "hard_reversal_block": args.hard_reversal_block,
            "hard_reintensification_block": args.hard_reintensification_block,
            "hard_sustained_high_block": args.hard_sustained_high_block,
            "hard_ramp_onset_block": args.hard_ramp_onset_block,
            "safety_arm": args.safety_arm,
            "economy_arm": args.economy_arm,
            "model_selection_policy": args.model_selection_policy,
        },
        args.output_dir / "psc_replay_selector_model.joblib",
    )

    readout = {
        "feature_source": args.feature_source,
        "feature_split": args.feature_split,
        "chosen_model_kind": chosen_kind,
        "chosen_threshold": chosen_threshold,
        "cases": int(len(dataset)),
        "missing_signal_cases": int(len(missing_signal_cases)),
        "split_counts": dataset["split"].value_counts().to_dict(),
        "label_counts": dataset["use_rawenv"].value_counts().to_dict(),
        "hard_reversal_block": args.hard_reversal_block,
        "hard_reintensification_block": args.hard_reintensification_block,
        "hard_sustained_high_block": args.hard_sustained_high_block,
        "hard_ramp_onset_block": args.hard_ramp_onset_block,
        "min_calibration_pump_gain_m3": args.min_calibration_pump_gain_m3,
        "threshold_fit_split": args.threshold_fit_split,
        "model_selection_policy": args.model_selection_policy,
        "holdout_risk_time_gt5_s": args.holdout_risk_time_gt5_s,
        "summary_csv": str(args.output_dir / "psc_replay_selector_summary.csv"),
        "case_predictions_csv": str(args.output_dir / "psc_replay_selector_case_predictions.csv"),
    }
    (args.output_dir / "psc_replay_selector_readout.json").write_text(json.dumps(readout, indent=2), encoding="utf-8")
    print(json.dumps(readout, indent=2))
    print(candidates.to_string(index=False))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
