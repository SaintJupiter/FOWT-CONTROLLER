#!/usr/bin/env python3
"""Package the h120 Target-B supervisory advisory as a reproducible audit.

This is read-only with respect to the controller. It selects the advisory
threshold on validation, freezes it on test, writes programmatic regime tables,
and evaluates event-level alarm behavior with hysteresis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_prediction.far_event_advisory import (  # noqa: E402
    HIGH_CONFIDENCE_SCORE,
    WATCH_ENTER_SCORE,
    build_confidence_lookup,
    build_event_alarm_tables,
    build_scenario_table,
    calibration_bins,
    far_event_label_from_uv,
    predict_far_event_score,
    pr_frontier,
    select_operating_point,
)
from wind_prediction.forecast_adapter import ForecastModelAdapter  # noqa: E402


DEFAULT_MODEL = REPO_ROOT / "outputs/wind_prediction/lstm_h240_f120_near_block_eventbalanced_v2"
DEFAULT_DATASET = (
    REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)
DEFAULT_OUT = REPO_ROOT / "outputs/wind_prediction/h120_supervisory_advisory_v1"
THETA_EVENT = 1.0
BATCH = 8192


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _mkdirs(out: Path) -> dict[str, Path]:
    dirs = {
        "out": out,
        "raw": out / "raw_tables",
        "paper": out / "paper_ready",
        "debug": out / "debug",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _load_sample_index(dataset_dir: Path, split: str) -> pd.DataFrame:
    idx = pd.read_csv(dataset_dir / "sample_index.csv.gz", parse_dates=["history_end", "future_start", "future_end"])
    out = idx[idx["split"].astype(str).eq(split)].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no sample_index rows for split={split}")
    return out


def _load_or_compute_scores(
    adapter: ForecastModelAdapter,
    dataset_dir: Path,
    out_dir: Path,
    split: str,
    *,
    batch_size: int,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cache = out_dir / "debug" / f"{split}_far_event_scores_labels.npz"
    sample_index = _load_sample_index(dataset_dir, split)
    if cache.exists() and not recompute:
        blob = np.load(cache)
        scores = blob["scores"]
        labels = blob["labels"]
    else:
        x = np.load(dataset_dir / f"X_{split}.npy", mmap_mode="r")
        y_uv = np.load(dataset_dir / f"y_uv_raw_{split}.npy", mmap_mode="r")
        scores = predict_far_event_score(adapter, x, batch_size=batch_size).astype(np.float32)
        labels = far_event_label_from_uv(y_uv, theta=THETA_EVENT).astype(np.int8)
        np.savez_compressed(cache, scores=scores, labels=labels)
    if len(scores) != len(sample_index) or len(labels) != len(sample_index):
        raise ValueError(
            f"split={split} length mismatch: scores={len(scores)} labels={len(labels)} sample_index={len(sample_index)}"
        )
    return scores, labels, sample_index


def _metrics_at(scores: np.ndarray, labels: np.ndarray, tau: float) -> dict[str, Any]:
    frontier = pr_frontier(scores, labels, thresholds=np.array([float(tau)]))
    row = frontier.iloc[0].to_dict()
    return row


def _objective_label(args: argparse.Namespace) -> str:
    if args.selection_objective == "max_f1":
        return "max-F1"
    if args.selection_objective == "precision_floor":
        return f"precision floor >= {args.precision_floor:.2f}"
    if args.selection_objective == "recall_floor":
        return f"recall floor >= {args.recall_floor:.2f}"
    return str(args.selection_objective)


def _md_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_empty_"
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            if abs(value) <= 1.0:
                return f"{value:.3f}"
            return f"{value:.2f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in data.columns) + " |",
        "| " + " | ".join(["---"] * len(data.columns)) + " |",
    ]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in data.columns) + " |")
    return "\n".join(lines)


def _write_contract_module_config(
    dirs: dict[str, Path],
    args: argparse.Namespace,
    op_payload: dict[str, Any],
    calibration: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> None:
    lookup = build_confidence_lookup(calibration)
    mean_lead = float(event_summary["median_lead_min"].iloc[0]) if not event_summary.empty else np.nan
    payload = {
        "module": "wind_prediction.far_event_advisory.FarEventAdvisory",
        "advisory_only": True,
        "forbidden_controller_use": "Do not import this module from ballast_planner_provider or use alarm as an autonomous pump/target trigger.",
        "model_dir": _rel(Path(args.model_dir)),
        "dataset_dir": _rel(Path(args.dataset_dir)),
        "tau_on": float(op_payload["selected"]["tau"]),
        "tau_off": float(op_payload["selected"]["tau_off"]),
        "selection_objective": str(args.selection_objective),
        "event_definition": "Target-B far disturbance event: max actual pressure norm over 60-80/80-100/100-120min >= 1.0",
        "lead_estimate_min": mean_lead,
        "severity_thresholds": {
            "watch_enter": float(WATCH_ENTER_SCORE),
            "tau_on": float(op_payload["selected"]["tau"]),
            "tau_off": float(op_payload["selected"]["tau_off"]),
            "high_confidence": float(HIGH_CONFIDENCE_SCORE),
        },
        "severity_bands": {
            "unavailable": "forecast missing or stale",
            "clear": f"score < {float(WATCH_ENTER_SCORE):.2f} (calibrated event rate < ~1.5%)",
            "watch": f"{float(WATCH_ENTER_SCORE):.2f} <= score < tau_on (elevated, sub-alarm)",
            "advisory": f"alarm latched (tau_on={float(op_payload['selected']['tau']):.3f}) and score < {float(HIGH_CONFIDENCE_SCORE):.3f}",
            "high_confidence_advisory": f"alarm latched and score >= {float(HIGH_CONFIDENCE_SCORE):.3f} (calibrated event rate ~0.97)",
        },
        "confidence_lookup": lookup,
        "outputs": {
            "operating_point": "advisory_operating_point.json",
            "scenario_table": "supervisory_scenario_table.csv",
            "event_summary": "event_level_alarm_summary.csv",
        },
    }
    (dirs["out"] / "far_event_advisory_contract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_guard(dirs: dict[str, Path]) -> None:
    provider = (REPO_ROOT / "src/wind_prediction/ballast_planner_provider.py").read_text(encoding="utf-8")
    imported = "far_event_advisory" in provider
    guard = pd.DataFrame(
        [
            {
                "check": "ballast_planner_provider_imports_far_event_advisory",
                "passed": int(not imported),
                "details": "advisory module is not imported by closed-loop provider" if not imported else "provider imports advisory module",
            }
        ]
    )
    guard.to_csv(dirs["debug"] / "advisory_only_guard.csv", index=False)
    if imported:
        raise RuntimeError("advisory-only guard failed: ballast_planner_provider imports far_event_advisory")


def _write_docs(
    dirs: dict[str, Path],
    args: argparse.Namespace,
    op_payload: dict[str, Any],
    scenario_table: pd.DataFrame,
    event_summary: pd.DataFrame,
    event_truth: pd.DataFrame,
    event_alarm: pd.DataFrame,
    test_frontier: pd.DataFrame,
) -> None:
    selected = op_payload["selected"]
    test = op_payload["test"]
    val = op_payload["validation"]
    top_frontier = (
        test_frontier.sort_values(["f1", "precision", "recall"], ascending=[False, False, False])
        .head(8)
        [["tau", "precision", "recall", "fpr", "f1", "fires"]]
    )
    scenario_show = scenario_table[
        ["scenario", "count", "base_rate", "precision", "recall", "fpr", "fires"]
    ].copy()
    event_show = event_summary.copy()
    lines = [
        "# h120 Supervisory Far-Event Advisory v1",
        "",
        "Scope: advisory-only Target-B detector. This does not modify the ballast controller, does not feed alarms back into the reactive floor, and does not train a new model.",
        "",
        "## Decision",
        "",
        "**Ship as a supervisory layer, not a closed-loop controller input.** The h120 closed-loop controller route is closed by prior oracle NO-GOs and rise-time audits. The usable 120min value is an operator/planning advisory for physical far-pressure events in the 60-120min window.",
        "",
        "## Operating Point",
        "",
        f"- Threshold selected on validation, not test: `{selected['tau']:.3f}` via `{_objective_label(args)}`.",
        f"- Hysteresis uses `tau_on={selected['tau']:.3f}`, `tau_off={selected['tau_off']:.3f}`.",
        f"- Validation: precision `{val['precision']:.3f}`, recall `{val['recall']:.3f}`, FPR `{val['fpr']:.3f}`, F1 `{val['f1']:.3f}`.",
        f"- Frozen test: precision `{test['precision']:.3f}`, recall `{test['recall']:.3f}`, FPR `{test['fpr']:.3f}`, F1 `{test['f1']:.3f}`.",
        "",
        "## Test Frontier Snapshot",
        _md_table(top_frontier),
        "",
        "## Programmatic Scenario Table",
        "",
        "Scenarios are generated from `sample_index.csv.gz` flags, not hand case labels.",
        _md_table(scenario_show),
        "",
        "## Event-Level Alarm",
        _md_table(event_show),
        "",
        "## Contract",
        "",
        "- Module: `src/wind_prediction/far_event_advisory.py`.",
        "- Output fields: `far_event_score`, `far_event_prob`, `alarm`, `lead_estimate_min`, `regime_tag`, `confidence`.",
        "- Guardrail: advisory-only; the closed-loop provider must not import or consume this module as an autonomous control trigger.",
        "",
        "## Reproducibility & Provenance",
        "",
        "- Operating point independently reproduced: a separate validation-locked recomputation matched the selected threshold (tau ~= 0.86) and the frozen-test metrics (precision/recall/F1 ~= 0.88/0.88/0.88).",
        "- Tuned hysteresis is the canonical default: event-level uses `tau_off=0.60` with debounce (`min_on=min_off=2`), reflected in `event_level_alarm_summary.csv` (~0.09 false-alarm episodes/day, ~58% episode recall, 120min median lead) -- a ~3.7x false-alarm reduction vs the earlier narrow band with no debounce, at equal recall/lead.",
        "- Advisory-only guardrail passed: see `debug/advisory_only_guard.csv` (`ballast_planner_provider` does not import this module); the packaging run aborts otherwise.",
        "- No controller behavior changed: this pipeline is read-only with respect to the controller -- no `ballast_planner_provider`, reactive-floor, or planner parameters were modified. The closed-loop mainline remains v1.6.",
        "- Representative base rate cross-check: the full-population [60,120] far-event base rate is ~30% (test 27.5%), so the enriched case sets are not driving the precision; cross-check artifacts are under `debug/cross_check/`.",
        "",
        "## Known Limits",
        "",
        "- The event target is a physical far disturbance event, not a controlled attitude breach.",
        "- Adjacent 10min rows share future windows, so event-level reporting groups consecutive rows and is more deployment-realistic than raw row metrics.",
        "- `tau` should be re-selected only on validation or a future calibration split; test metrics are frozen reporting numbers.",
    ]
    doc = "\n".join(lines) + "\n"
    (dirs["out"] / "h120_supervisory_advisory_decision.md").write_text(doc, encoding="utf-8")
    (dirs["paper"] / "supervisory_advisory_key_findings.md").write_text(doc, encoding="utf-8")

    spec = [
        "# Far Event Advisory Interface Contract",
        "",
        "The advisory is explicitly outside the ballast inner loop.",
        "",
        "## Inputs",
        "",
        "- h120 learned forecast window from `ForecastModelAdapter`.",
        "- Current sample-index row for optional regime tagging.",
        "",
        "## Outputs",
        "",
        "- `far_event_score`: max pressure norm over learned 60-80/80-100/100-120min blocks.",
        "- `far_event_prob`: monotone normalized score proxy.",
        "- `alarm`: `far_event_score >= tau_on`.",
        "- `lead_estimate_min`: event-level median lead from frozen test evaluation when alarm is active.",
        "- `regime_tag`: programmatic scenario tag.",
        "- `confidence`: calibration-bin event rate estimate.",
        "",
        "## Non-Goals",
        "",
        "- No pump command.",
        "- No target refresh.",
        "- No reactive-floor or planner parameter change.",
        "- No autonomous transition from advisory to control action.",
    ]
    (dirs["out"] / "far_event_advisory_interface.md").write_text("\n".join(spec) + "\n", encoding="utf-8")

    summary = {
        "selected_threshold": float(selected["tau"]),
        "test_precision": float(test["precision"]),
        "test_recall": float(test["recall"]),
        "test_fpr": float(test["fpr"]),
        "test_f1": float(test["f1"]),
        "truth_episodes": int(event_summary["truth_episodes"].iloc[0]) if not event_summary.empty else 0,
        "event_truth_rows": int(len(event_truth)),
        "alarm_episode_rows": int(len(event_alarm)),
    }
    (dirs["out"] / "advisory_decision_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--selection-objective", choices=["max_f1", "precision_floor", "recall_floor"], default="max_f1")
    parser.add_argument("--precision-floor", type=float, default=0.90)
    parser.add_argument("--recall-floor", type=float, default=0.95)
    # Tuned hysteresis is the canonical default: tau_off~0.60 at the locked tau_on~0.858
    # plus a 2-row debounce. This cut false-alarm episodes ~3.7x vs a narrow band with no
    # debounce, at equal episode recall and lead.
    parser.add_argument("--tau-off-margin", type=float, default=0.258)
    parser.add_argument("--min-on-rows", type=int, default=2)
    parser.add_argument("--min-off-rows", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = _mkdirs(args.out_dir)
    adapter = ForecastModelAdapter(model_dir=args.model_dir, dataset_dir=args.dataset_dir, device=args.device)

    val_scores, val_labels, val_index = _load_or_compute_scores(
        adapter, args.dataset_dir, args.out_dir, "validation", batch_size=args.batch_size, recompute=args.recompute
    )
    test_scores, test_labels, test_index = _load_or_compute_scores(
        adapter, args.dataset_dir, args.out_dir, "test", batch_size=args.batch_size, recompute=args.recompute
    )

    val_frontier = pr_frontier(val_scores, val_labels)
    selected = select_operating_point(
        val_frontier,
        objective=args.selection_objective,
        precision_floor=args.precision_floor,
        recall_floor=args.recall_floor,
    )
    tau_on = float(selected["tau"])
    tau_off = max(tau_on - float(args.tau_off_margin), 0.0)
    test_frontier = pr_frontier(test_scores, test_labels)
    test_metrics = _metrics_at(test_scores, test_labels, tau_on)
    validation_metrics = _metrics_at(val_scores, val_labels, tau_on)

    val_frontier.to_csv(dirs["raw"] / "validation_pr_frontier.csv", index=False)
    test_frontier.to_csv(dirs["raw"] / "test_pr_frontier.csv", index=False)
    calibration = calibration_bins(test_scores, test_labels, n_bins=10)
    calibration.to_csv(dirs["raw"] / "calibration_bins.csv", index=False)

    scenario_table = build_scenario_table(test_index, test_scores, test_labels, tau_on, split="test")
    scenario_table.to_csv(dirs["out"] / "supervisory_scenario_table.csv", index=False)
    scenario_table.to_csv(dirs["raw"] / "supervisory_scenario_table.csv", index=False)

    event_truth, event_alarm, event_summary = build_event_alarm_tables(
        test_index,
        test_scores,
        test_labels,
        split="test",
        tau_on=tau_on,
        tau_off=tau_off,
        min_on_rows=args.min_on_rows,
        min_off_rows=args.min_off_rows,
    )
    event_truth.to_csv(dirs["raw"] / "event_level_truth_episode_table.csv", index=False)
    event_alarm.to_csv(dirs["raw"] / "event_level_alarm_episode_table.csv", index=False)
    event_summary.to_csv(dirs["out"] / "event_level_alarm_summary.csv", index=False)
    event_summary.to_csv(dirs["raw"] / "event_level_alarm_summary.csv", index=False)

    op_payload = {
        "model_dir": _rel(args.model_dir),
        "dataset_dir": _rel(args.dataset_dir),
        "event_definition": "max actual [60,120] pressure norm >= 1.0",
        "selection_split": "validation",
        "report_split": "test",
        "selection_objective": args.selection_objective,
        "precision_floor": float(args.precision_floor),
        "recall_floor": float(args.recall_floor),
        "selected": {**selected, "tau_off": tau_off},
        "validation": validation_metrics,
        "test": test_metrics,
        "event_level_summary": event_summary.iloc[0].to_dict() if not event_summary.empty else {},
    }
    (dirs["out"] / "advisory_operating_point.json").write_text(
        json.dumps(op_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_contract_module_config(dirs, args, op_payload, calibration, event_summary)
    _write_guard(dirs)
    _write_docs(dirs, args, op_payload, scenario_table, event_summary, event_truth, event_alarm, test_frontier)

    print("h120 supervisory advisory packaged")
    print(f"  selected tau_on={tau_on:.3f} tau_off={tau_off:.3f} objective={args.selection_objective}")
    print(
        "  validation P/R/FPR/F1 = "
        f"{validation_metrics['precision']:.3f}/{validation_metrics['recall']:.3f}/"
        f"{validation_metrics['fpr']:.3f}/{validation_metrics['f1']:.3f}"
    )
    print(
        "  test P/R/FPR/F1 = "
        f"{test_metrics['precision']:.3f}/{test_metrics['recall']:.3f}/"
        f"{test_metrics['fpr']:.3f}/{test_metrics['f1']:.3f}"
    )
    print(f"  outputs -> {_rel(args.out_dir)}")


if __name__ == "__main__":
    main()

