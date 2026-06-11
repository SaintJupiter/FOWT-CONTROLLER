#!/usr/bin/env python3
"""First PSC GRU+adapter selector harness on the retained mixed pool.

This is a control-chain selection test, not a new plant replay. It uses the
already-run deployable arms on the 10-case mixed pool and asks whether a fixed
precedence selector, fed by PSC GRU+adapter signals, chooses a better arm than
blind/current-only selectors. If this passes, the same selector should be wired
into a real replay provider.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CASE_RE = re.compile(r"^(?P<case_id>\d+_[^_]+(?:_[^_]+)*)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hhmmss>\d{6})_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/decision_signals_gru_adapter_v1"),
    )
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/wind_prediction/psc_degradation_ladder_20260531/degradation_ladder_case_metrics.csv"),
    )
    parser.add_argument(
        "--timeseries-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_degradation_ladder_20260531/learned_forecast_adaptive/timeseries"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_gru_adapter_selector_harness_v1"),
    )
    parser.add_argument("--case-signal-window-min", type=int, default=30)
    parser.add_argument("--confidence-quantile", type=float, default=0.60)
    parser.add_argument("--opportunity-quantile", type=float, default=0.80)
    parser.add_argument("--risk-quantile", type=float, default=0.95)
    parser.add_argument(
        "--safety-arm",
        choices=["blind_deadband", "current_forecast_adaptive"],
        default="blind_deadband",
        help="Deployable arm used for safety abstain/default tracking.",
    )
    parser.add_argument(
        "--economy-arm",
        choices=["learned_forecast_adaptive", "learned_rawenv_mainline", "current_rawenv_mainline"],
        default="learned_forecast_adaptive",
        help="Deployable arm used when the learned selector chooses RELAX.",
    )
    parser.add_argument(
        "--policy",
        choices=["precedence_v1", "fix_a_high_precision_abstain"],
        default="precedence_v1",
    )
    return parser.parse_args()


def case_start_table(timeseries_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(timeseries_dir.glob("*_timeseries.csv")):
        match = CASE_RE.match(path.name)
        if not match:
            continue
        case_id = match.group("case_id")
        date = match.group("date")
        hhmmss = match.group("hhmmss")
        start = pd.to_datetime(f"{date} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}")
        rows.append({"case_id": case_id, "case_start": start, "timeseries_file": str(path)})
    return pd.DataFrame(rows)


def _threshold_for_precision(y: np.ndarray, score: np.ndarray, target_precision: float, min_recall: float = 0.05) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0.50, 0.995, 100)))
    best = float(np.quantile(score, 0.99))
    best_recall = -1.0
    for threshold in candidates:
        pred = score >= threshold
        tp = int((pred & (y >= 0.5)).sum())
        fp = int((pred & (y < 0.5)).sum())
        fn = int(((~pred) & (y >= 0.5)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision >= target_precision and recall >= min_recall and recall > best_recall:
            best = float(threshold)
            best_recall = recall
    return best


def _threshold_for_min_recall(y: np.ndarray, score: np.ndarray, target_recall: float) -> float:
    """Highest-precision threshold that still catches a target recall level."""
    candidates = np.unique(np.quantile(score, np.linspace(0.05, 0.995, 200)))
    best = float(np.quantile(score, 0.50))
    best_precision = -1.0
    for threshold in candidates:
        pred = score >= threshold
        tp = int((pred & (y >= 0.5)).sum())
        fp = int((pred & (y < 0.5)).sum())
        fn = int(((~pred) & (y >= 0.5)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if recall >= target_recall and precision > best_precision:
            best = float(threshold)
            best_precision = precision
    return best


def calibration_thresholds(
    signals_dir: Path,
    source_dataset_dir: Path,
    confidence_q: float,
    opportunity_q: float,
    risk_q: float,
) -> dict[str, float]:
    validation = pd.read_csv(signals_dir / "psc_decision_signals_validation.csv.gz")
    stable_mask = validation["psc_regime_prob_stable"] >= validation["psc_regime_prob_stable"].quantile(0.50)
    sample_index = pd.read_csv(source_dataset_dir / "sample_index.csv.gz")
    sample_index = sample_index[sample_index["split"].eq("validation")].reset_index(drop=True)
    reversal_truth = sample_index["direction_shift_ge_45deg"].to_numpy(dtype=np.float32)
    reint_truth = sample_index["vector_change_ge_train_p90"].to_numpy(dtype=np.float32)
    rev_hp = _threshold_for_precision(
        reversal_truth,
        validation["psc_prob_event_reversal_signflip"].to_numpy(dtype=np.float32),
        0.80,
    )
    rein_hp = _threshold_for_precision(
        reint_truth,
        validation["psc_prob_event_reintensification"].to_numpy(dtype=np.float32),
        0.80,
    )
    rev_economy_cap = _threshold_for_min_recall(
        reversal_truth,
        validation["psc_prob_event_reversal_signflip"].to_numpy(dtype=np.float32),
        0.70,
    )

    return {
        "confidence_c": float(validation["psc_regime_confidence"].quantile(confidence_q)),
        "opportunity_p": float(validation.loc[stable_mask, "psc_value_deadband_opportunity_m3"].quantile(opportunity_q)),
        "risk_veto": float(validation["psc_value_relax_event_exposure_risk_s"].quantile(risk_q)),
        "reversal_high_precision": rev_hp,
        "reintensification_high_precision": rein_hp,
        "reversal_economy_cap": rev_economy_cap,
        # Structural economy eligibility caps: a strong economy arm is not
        # allowed in event/onset or sustained-high regimes. The 0.35 boundary is
        # deliberately coarse: above it, a 6-way regime head is signaling that
        # the class is materially present rather than background noise.
        "ramp_onset_economy_cap": 0.35,
        "sustained_high_economy_cap": 0.35,
    }


def case_signal_features(signals: pd.DataFrame, cases: pd.DataFrame, window_min: int) -> pd.DataFrame:
    signals = signals.copy()
    signals["future_start"] = pd.to_datetime(signals["future_start"])
    rows = []
    for row in cases.itertuples(index=False):
        start = row.case_start
        end = start + pd.Timedelta(minutes=window_min)
        subset = signals[(signals["future_start"] >= start) & (signals["future_start"] <= end)]
        if subset.empty:
            raise SystemExit(f"No PSC signals found for {row.case_id} around {start}.")
        out = {
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
        rows.append(out)
    return pd.DataFrame(rows)


def select_mode(
    features: pd.Series,
    thresholds: dict[str, float],
    learned: bool,
    safety_arm: str,
    economy_arm: str,
    policy: str,
) -> tuple[str, str, str]:
    c = thresholds["confidence_c"]
    p = thresholds["opportunity_p"]
    risk_veto = thresholds["risk_veto"]

    if learned:
        high_conf = features["confidence_max"] >= c
        rev = max(features["prob_reversal_signflip_mean"], features["event_reversal_signflip_max"])
        rein = max(features["prob_reintensification_mean"], features["event_reintensification_max"])
        if policy == "fix_a_high_precision_abstain":
            if rev >= thresholds["reversal_high_precision"] or rein >= thresholds["reintensification_high_precision"]:
                return "ABSTAIN_TRACK", safety_arm, "high_precision_adverse_abstain"
            if (
                features["opportunity_mean"] >= p
                and rev < thresholds["reversal_economy_cap"]
                and features["prob_ramp_onset_mean"] < thresholds["ramp_onset_economy_cap"]
                and features["prob_sustained_high_mean"] < thresholds["sustained_high_economy_cap"]
                and features["risk_max"] < risk_veto
            ):
                return "RELAX", economy_arm, "non_adverse_high_opportunity"
            if features["opportunity_mean"] >= p and features["prob_ramp_onset_mean"] >= thresholds["ramp_onset_economy_cap"]:
                return "TRACK_DEFAULT", safety_arm, "high_opportunity_ramp_onset_blocked"
            if features["opportunity_mean"] >= p and features["prob_sustained_high_mean"] >= thresholds["sustained_high_economy_cap"]:
                return "TRACK_DEFAULT", safety_arm, "high_opportunity_sustained_high_blocked"
            if features["opportunity_mean"] >= p and features["risk_max"] >= risk_veto:
                return "TRACK_DEFAULT", safety_arm, "high_opportunity_risk_veto"
            if features["opportunity_mean"] >= p:
                return "TRACK_DEFAULT", safety_arm, "high_opportunity_event_cap_blocked"
            return "TRACK_DEFAULT", safety_arm, "non_adverse_no_opportunity"

        if rev >= 0.75 or rein >= 0.75:
            return "ABSTAIN_TRACK", safety_arm, "high_event_probability_safety_abstain"
        if high_conf and (rev >= 0.65 or rein >= 0.65):
            return "ABSTAIN_TRACK", safety_arm, "safety_abstain_reversal_or_reintensification"
        if high_conf and features["prob_sustained_high_mean"] >= 0.35:
            return "PRE_FLOOR_WATCH", "learned_forecast_adaptive", "sustained_high_watch"
        if high_conf and features["prob_transient_decay_mean"] >= 0.35 and features["time_to_attention_min_min"] >= 20:
            return "RELIEF", "learned_forecast_adaptive", "transient_decay_relief"
        if (
            high_conf
            and features["prob_stable_mean"] >= 0.25
            and features["opportunity_mean"] >= p
            and features["risk_max"] < risk_veto
        ):
            return "RELAX", economy_arm, "stable_high_opportunity"
        return "TRACK_DEFAULT", safety_arm, "default_low_conf_or_no_opportunity"

    # Current-only selector: intentionally cannot use PSC future regime/event/
    # value outputs. In this cheap harness, the retained current-forecast arm is
    # the available current-only comparator.
    return "CURRENT_ONLY_REFERENCE", "current_forecast_adaptive", "current_only_reference_arm"


def summarize_arm(name: str, cases: pd.DataFrame) -> dict[str, float | str | int]:
    totals = {
        "arm": name,
        "cases": int(len(cases)),
        "pump_m3": float(cases["pump_m3"].sum()),
        "time_gt5_s": float(cases["time_gt5_s"].sum()),
        "time_gt6_s": float(cases["time_gt6_s"].sum()),
        "fallback_s": float(cases["fallback_s"].sum()),
        "latch_switches": float(cases["latch_switches"].sum()),
    }
    return totals


def concentration(selected: pd.DataFrame, metric_col: str) -> float:
    values = selected[metric_col].to_numpy(dtype=float)
    positive = values[values > 0]
    if positive.size == 0:
        return 0.0
    return float(positive.max() / positive.sum())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = calibration_thresholds(
        args.signals_dir,
        args.source_dataset_dir,
        args.confidence_quantile,
        args.opportunity_quantile,
        args.risk_quantile,
    )
    cases = case_start_table(args.timeseries_dir)
    signals = pd.read_csv(args.signals_dir / "psc_decision_signals_test.csv.gz")
    features = case_signal_features(signals, cases, args.case_signal_window_min)
    features.to_csv(args.output_dir / "psc_case_signal_features.csv", index=False)

    metrics = pd.read_csv(args.metrics)
    metric_map = metrics.set_index(["arm", "case_id"])

    selections = []
    for row in features.itertuples(index=False):
        feature_row = pd.Series(row._asdict())
        for selector_name, learned in (("learned_psc_gru_adapter", True), ("current_only_proxy_selector", False)):
            mode, arm, reason = select_mode(
                feature_row,
                thresholds,
                learned=learned,
                safety_arm=args.safety_arm,
                economy_arm=args.economy_arm,
                policy=args.policy,
            )
            metric = metric_map.loc[(arm, row.case_id)].to_dict()
            selections.append(
                {
                    "selector": selector_name,
                    "case_id": row.case_id,
                    "selected_mode": mode,
                    "selected_arm": arm,
                    "reason": reason,
                    **{k: metric[k] for k in ["label", "pump_m3", "time_gt5_s", "time_gt6_s", "p95_axis_deg", "max_axis_deg", "fallback_s", "latch_switches"]},
                }
            )
    selected = pd.DataFrame(selections)

    # Add fixed reference arms.
    references = []
    for arm in ("blind_deadband", "current_forecast_adaptive", "learned_forecast_adaptive", "oracle_selector_v0"):
        sub = metrics[metrics["arm"].eq(arm)].copy()
        sub["selector"] = arm
        sub["selected_mode"] = arm
        sub["selected_arm"] = arm
        sub["reason"] = "reference"
        references.append(sub[selected.columns])
    all_cases = pd.concat([selected, *references], ignore_index=True)

    blind = all_cases[all_cases["selector"].eq("blind_deadband")][["case_id", "pump_m3", "time_gt5_s", "time_gt6_s", "fallback_s"]]
    blind = blind.rename(
        columns={
            "pump_m3": "blind_pump_m3",
            "time_gt5_s": "blind_time_gt5_s",
            "time_gt6_s": "blind_time_gt6_s",
            "fallback_s": "blind_fallback_s",
        }
    )
    all_cases = all_cases.merge(blind, on="case_id", how="left")
    all_cases["d_pump_vs_blind_m3"] = all_cases["pump_m3"] - all_cases["blind_pump_m3"]
    all_cases["d_time_gt5_vs_blind_s"] = all_cases["time_gt5_s"] - all_cases["blind_time_gt5_s"]
    all_cases["d_time_gt6_vs_blind_s"] = all_cases["time_gt6_s"] - all_cases["blind_time_gt6_s"]
    all_cases["d_fallback_vs_blind_s"] = all_cases["fallback_s"] - all_cases["blind_fallback_s"]
    all_cases.to_csv(args.output_dir / "psc_selector_harness_case_metrics.csv", index=False)

    summaries = []
    for selector, sub in all_cases.groupby("selector", sort=False):
        summary = summarize_arm(selector, sub)
        summary["d_pump_vs_blind_m3"] = float(sub["d_pump_vs_blind_m3"].sum())
        summary["d_time_gt5_vs_blind_s"] = float(sub["d_time_gt5_vs_blind_s"].sum())
        summary["d_time_gt6_vs_blind_s"] = float(sub["d_time_gt6_vs_blind_s"].sum())
        summary["d_fallback_vs_blind_s"] = float(sub["d_fallback_vs_blind_s"].sum())
        summary["top_case_time_gain_share"] = concentration(sub.assign(time_gain=-sub["d_time_gt5_vs_blind_s"]), "time_gain")
        summary["top_case_pump_gain_share"] = concentration(sub.assign(pump_gain=-sub["d_pump_vs_blind_m3"]), "pump_gain")
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "psc_selector_harness_summary.csv", index=False)

    falsifier = {
        "learned_vs_current_mode_diff_cases": int(
            (
                selected.pivot(index="case_id", columns="selector", values="selected_mode")["learned_psc_gru_adapter"]
                != selected.pivot(index="case_id", columns="selector", values="selected_mode")["current_only_proxy_selector"]
            ).sum()
        ),
        "learned_vs_current_arm_diff_cases": int(
            (
                selected.pivot(index="case_id", columns="selector", values="selected_arm")["learned_psc_gru_adapter"]
                != selected.pivot(index="case_id", columns="selector", values="selected_arm")["current_only_proxy_selector"]
            ).sum()
        ),
    }
    learned_sum = summary_df[summary_df["arm"].eq("learned_psc_gru_adapter")].iloc[0].to_dict()
    current_sum = summary_df[summary_df["arm"].eq("current_only_proxy_selector")].iloc[0].to_dict()
    falsifier["learned_minus_current_pump_m3"] = float(learned_sum["pump_m3"] - current_sum["pump_m3"])
    falsifier["learned_minus_current_time_gt5_s"] = float(learned_sum["time_gt5_s"] - current_sum["time_gt5_s"])
    falsifier["learned_minus_current_fallback_s"] = float(learned_sum["fallback_s"] - current_sum["fallback_s"])

    readout = {
        "thresholds_from_validation": thresholds,
        "case_signal_window_min": args.case_signal_window_min,
        "safety_arm": args.safety_arm,
        "economy_arm": args.economy_arm,
        "policy": args.policy,
        "falsifier": falsifier,
        "summary_csv": str(args.output_dir / "psc_selector_harness_summary.csv"),
        "case_metrics_csv": str(args.output_dir / "psc_selector_harness_case_metrics.csv"),
    }
    (args.output_dir / "psc_selector_harness_readout.json").write_text(json.dumps(readout, indent=2), encoding="utf-8")
    print(json.dumps(readout, indent=2))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
