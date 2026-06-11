#!/usr/bin/env python3
"""Read-only h120-vs-h60 incremental value audit for supervisory advisory.

Question:
    Does the 120min forecast output add supervisory information beyond the
    0-60min/near-horizon output?

Target:
    Target-B far disturbance event:
    max actual pressure norm over [60,120]min >= theta_event.

Scores compared on the same validation/test rows:
    near_to_far_score = max learned pressure norm over [0,60]min
    far_score         = max learned pressure norm over [60,120]min

Both scores use the same h240_f120 learned model.  This is not a controller
experiment and does not modify ballast control behavior.
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
    build_event_alarm_tables,
    build_scenario_table,
    pr_frontier,
    select_operating_point,
)
from wind_prediction.forecast_adapter import ForecastModelAdapter  # noqa: E402


DEFAULT_MODEL = REPO_ROOT / "outputs/wind_prediction/lstm_h240_f120_near_block_eventbalanced_v2"
DEFAULT_DATASET = (
    REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)
DEFAULT_OUT = REPO_ROOT / "outputs/wind_prediction/h120_vs_h60_supervisory_incremental_value_v1"
THETA_EVENT = 1.0
NEAR_BLOCKS = ((0, 2), (2, 4), (4, 6))
FAR_BLOCKS = ((6, 8), (8, 10), (10, 12))


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _mkdirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "out": out_dir,
        "raw": out_dir / "raw_tables",
        "paper": out_dir / "paper_ready",
        "debug": out_dir / "debug",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _load_sample_index(dataset_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(dataset_dir / "sample_index.csv.gz", parse_dates=["history_end", "future_start", "future_end"])
    out = frame[frame["split"].astype(str).eq(split)].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no sample_index rows for split={split!r}")
    return out


def _block_scores(uv: np.ndarray, blocks: tuple[tuple[int, int], ...], wind_ref: float = 12.0, cap: float = 1.5) -> np.ndarray:
    arr = np.asarray(uv, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] < 12 or arr.shape[2] != 2:
        raise ValueError(f"expected uv shape [N, >=12, 2], got {arr.shape}")
    cols: list[np.ndarray] = []
    for start, end in blocks:
        speed = np.sqrt(arr[:, start:end, 0] ** 2 + arr[:, start:end, 1] ** 2).mean(axis=1)
        cols.append(np.clip((np.maximum(speed, 0.0) / float(wind_ref)) ** 2, 0.0, float(cap)))
    return np.stack(cols, axis=1).astype(np.float32)


def _predict_uv_raw(
    adapter: ForecastModelAdapter,
    x_window: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    adapter._lazy_load_runtime()
    torch = adapter._torch
    x = np.asarray(x_window, dtype=np.float32)
    out: list[np.ndarray] = []
    for start in range(0, len(x), int(batch_size)):
        xb = torch.from_numpy(x[start : start + int(batch_size)]).to(adapter._device)
        with torch.no_grad():
            pred_uv_scaled, _ = adapter._model(xb)
            pred_uv_scaled = adapter._full_scaled_prediction(pred_uv_scaled, xb)
        pred_uv_raw = adapter._inverse_scale_uv(pred_uv_scaled.detach().cpu().numpy().astype(np.float32))
        out.append(pred_uv_raw.astype(np.float32))
    return np.concatenate(out, axis=0)


def _load_or_compute_split(
    adapter: ForecastModelAdapter,
    dataset_dir: Path,
    dirs: dict[str, Path],
    split: str,
    *,
    batch_size: int,
    recompute: bool,
    theta_event: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cache = dirs["debug"] / f"{split}_near_far_scores_labels.npz"
    sample_index = _load_sample_index(dataset_dir, split)
    if cache.exists() and not recompute:
        blob = np.load(cache)
        data = {name: blob[name] for name in blob.files}
    else:
        x = np.load(dataset_dir / f"X_{split}.npy", mmap_mode="r")
        y_uv = np.load(dataset_dir / f"y_uv_raw_{split}.npy", mmap_mode="r")
        pred_uv = _predict_uv_raw(adapter, x, batch_size=batch_size)
        pred_near_blocks = _block_scores(pred_uv, NEAR_BLOCKS)
        pred_far_blocks = _block_scores(pred_uv, FAR_BLOCKS)
        actual_near_blocks = _block_scores(np.asarray(y_uv), NEAR_BLOCKS)
        actual_far_blocks = _block_scores(np.asarray(y_uv), FAR_BLOCKS)
        data = {
            "near_score": pred_near_blocks.max(axis=1).astype(np.float32),
            "far_score": pred_far_blocks.max(axis=1).astype(np.float32),
            "actual_near_score": actual_near_blocks.max(axis=1).astype(np.float32),
            "actual_far_score": actual_far_blocks.max(axis=1).astype(np.float32),
            "far_label": (actual_far_blocks.max(axis=1) >= float(theta_event)).astype(np.int8),
            "near_label": (actual_near_blocks.max(axis=1) >= float(theta_event)).astype(np.int8),
        }
        np.savez_compressed(cache, **data)
    n = len(sample_index)
    for key, value in data.items():
        if len(value) != n:
            raise ValueError(f"split={split} length mismatch for {key}: {len(value)} vs sample_index={n}")
    return sample_index, data


def _metrics_at(scores: np.ndarray, labels: np.ndarray, tau: float) -> dict[str, Any]:
    return pr_frontier(scores, labels, thresholds=np.array([float(tau)])).iloc[0].to_dict()


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = pd.Series(all_scores).rank(method="average").to_numpy()
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _select_and_eval(
    name: str,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    objective: str,
    precision_floor: float,
    recall_floor: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    val_frontier = pr_frontier(val_scores, val_labels)
    selected = select_operating_point(
        val_frontier,
        objective=objective,
        precision_floor=precision_floor,
        recall_floor=recall_floor,
    )
    tau = float(selected["tau"])
    test_frontier = pr_frontier(test_scores, test_labels)
    val_at = _metrics_at(val_scores, val_labels, tau)
    test_at = _metrics_at(test_scores, test_labels, tau)
    row = {
        "score_name": name,
        "selection_objective": objective,
        "tau": tau,
        "validation_precision": float(val_at["precision"]),
        "validation_recall": float(val_at["recall"]),
        "validation_fpr": float(val_at["fpr"]),
        "validation_f1": float(val_at["f1"]),
        "validation_auc": _auc(val_labels, val_scores),
        "test_precision": float(test_at["precision"]),
        "test_recall": float(test_at["recall"]),
        "test_fpr": float(test_at["fpr"]),
        "test_f1": float(test_at["f1"]),
        "test_auc": _auc(test_labels, test_scores),
        "test_support": int(test_at["support"]),
        "test_negative": int(test_at["negative"]),
        "test_fires": int(test_at["fires"]),
        "test_tp": int(test_at["tp"]),
        "test_fp": int(test_at["fp"]),
        "test_fn": int(test_at["fn"]),
        "test_tn": int(test_at["tn"]),
    }
    return row, val_frontier, test_frontier


def _unique_detection_table(
    labels: np.ndarray,
    near_scores: np.ndarray,
    far_scores: np.ndarray,
    *,
    near_tau: float,
    far_tau: float,
) -> pd.DataFrame:
    labels = np.asarray(labels).astype(bool)
    near_fire = np.asarray(near_scores) >= float(near_tau)
    far_fire = np.asarray(far_scores) >= float(far_tau)
    groups = {
        "both_fire": near_fire & far_fire,
        "far_only_fire": far_fire & ~near_fire,
        "near_only_fire": near_fire & ~far_fire,
        "neither_fire": ~(near_fire | far_fire),
    }
    rows = []
    for name, mask in groups.items():
        total = int(mask.sum())
        tp = int((mask & labels).sum())
        fp = int((mask & ~labels).sum())
        rows.append(
            {
                "bucket_group": name,
                "rows": total,
                "true_far_events": tp,
                "non_events": fp,
                "event_rate": float(tp / total) if total else 0.0,
                "share_of_all_true_events": float(tp / labels.sum()) if labels.sum() else 0.0,
                "share_of_all_fires": float(total / max(int((near_fire | far_fire).sum()), 1)),
            }
        )
    return pd.DataFrame(rows)


def _event_summary_for_score(
    sample_index: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    tau: float,
    tau_off: float,
    min_on_rows: int,
    min_off_rows: int,
) -> pd.DataFrame:
    _, _, summary = build_event_alarm_tables(
        sample_index,
        scores,
        labels,
        split="test",
        tau_on=float(tau),
        tau_off=float(tau_off),
        min_on_rows=int(min_on_rows),
        min_off_rows=int(min_off_rows),
    )
    summary.insert(0, "score_name", name)
    return summary


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            if abs(value) <= 1.0:
                return f"{value:.3f}"
            return f"{value:.2f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _write_docs(
    dirs: dict[str, Path],
    compare: pd.DataFrame,
    event_summary: pd.DataFrame,
    unique: pd.DataFrame,
    scenario_near: pd.DataFrame,
    scenario_far: pd.DataFrame,
    payload: dict[str, Any],
) -> None:
    delta = payload["incremental_delta"]
    verdict = payload["verdict"]
    lines = [
        "# h120 vs h60 Supervisory Incremental Value v1",
        "",
        "Scope: read-only audit.  This does not touch the ballast controller, the reactive floor, or the advisory runtime path.",
        "",
        "## Question",
        "",
        "Does the 120min output add supervisory information beyond a 0-60min / near-only advisory?",
        "",
        "## Method",
        "",
        "- Common target: `actual max pressure norm over [60,120]min >= 1.0`.",
        "- Near-only comparator: `max learned pressure norm over [0,60]min`.",
        "- h120 far advisory: `max learned pressure norm over [60,120]min`.",
        "- Thresholds are selected on validation and frozen on test for each score independently.",
        "",
        "## Row-Level Comparison",
        _md_table(
            compare[
                [
                    "score_name",
                    "tau",
                    "test_precision",
                    "test_recall",
                    "test_fpr",
                    "test_f1",
                    "test_auc",
                    "test_fires",
                    "test_tp",
                    "test_fp",
                ]
            ]
        ),
        "",
        "## Event-Level Comparison",
        _md_table(
            event_summary[
                [
                    "score_name",
                    "alarm_duty_cycle",
                    "truth_episodes",
                    "truth_hits",
                    "truth_recall",
                    "median_lead_min",
                    "mean_lead_min",
                    "alarm_episodes",
                    "false_alarm_episodes_per_day",
                ]
            ]
        ),
        "",
        "## Unique Detection",
        _md_table(unique),
        "",
        "## Scenario Tables",
        "",
        "### Near-only",
        _md_table(scenario_near[["scenario", "count", "base_rate", "precision", "recall", "fpr", "fires"]]),
        "",
        "### h120 far",
        _md_table(scenario_far[["scenario", "count", "base_rate", "precision", "recall", "fpr", "fires"]]),
        "",
        "## Decision",
        "",
        verdict,
        "",
        "## Key Delta",
        "",
        f"- Precision delta (far - near): `{delta['precision']:.3f}`.",
        f"- Recall delta (far - near): `{delta['recall']:.3f}`.",
        f"- FPR delta (far - near): `{delta['fpr']:.3f}`.",
        f"- F1 delta (far - near): `{delta['f1']:.3f}`.",
        f"- AUC delta (far - near): `{delta['auc']:.3f}`.",
        f"- Event recall delta (far - near): `{delta['event_recall']:.3f}`.",
        f"- False-alarm episodes/day delta (far - near): `{delta['false_alarm_per_day']:.3f}`.",
        "",
        "## Interpretation",
        "",
        "If the far score materially beats the near-only score on the same [60,120] target, the 120min output is adding genuine supervisory information rather than merely repackaging 0-60min persistence.  If the two are similar, the h120 claim should be weakened to a generic far-event detector rather than a horizon-increment result.",
    ]
    doc = "\n".join(lines) + "\n"
    (dirs["out"] / "h120_vs_h60_supervisory_incremental_decision.md").write_text(doc, encoding="utf-8")
    (dirs["paper"] / "h120_vs_h60_incremental_key_findings.md").write_text(doc, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--selection-objective", choices=["max_f1", "precision_floor", "recall_floor"], default="max_f1")
    parser.add_argument("--precision-floor", type=float, default=0.90)
    parser.add_argument("--recall-floor", type=float, default=0.95)
    parser.add_argument("--tau-off-margin", type=float, default=0.258)
    parser.add_argument("--min-on-rows", type=int, default=2)
    parser.add_argument("--min-off-rows", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = _mkdirs(args.out_dir)
    adapter = ForecastModelAdapter(model_dir=args.model_dir, dataset_dir=args.dataset_dir, device=args.device)

    val_index, val = _load_or_compute_split(
        adapter,
        args.dataset_dir,
        dirs,
        "validation",
        batch_size=args.batch_size,
        recompute=args.recompute,
        theta_event=THETA_EVENT,
    )
    test_index, test = _load_or_compute_split(
        adapter,
        args.dataset_dir,
        dirs,
        "test",
        batch_size=args.batch_size,
        recompute=args.recompute,
        theta_event=THETA_EVENT,
    )

    rows = []
    frontiers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, key in (("near_only_0_60", "near_score"), ("h120_far_60_120", "far_score")):
        row, val_frontier, test_frontier = _select_and_eval(
            name,
            val[key],
            val["far_label"],
            test[key],
            test["far_label"],
            objective=args.selection_objective,
            precision_floor=args.precision_floor,
            recall_floor=args.recall_floor,
        )
        rows.append(row)
        frontiers[name] = (val_frontier, test_frontier)
        val_frontier.to_csv(dirs["raw"] / f"validation_pr_frontier_{name}.csv", index=False)
        test_frontier.to_csv(dirs["raw"] / f"test_pr_frontier_{name}.csv", index=False)
    compare = pd.DataFrame(rows)
    compare.to_csv(dirs["out"] / "horizon_incremental_compare_table.csv", index=False)
    compare.to_csv(dirs["raw"] / "horizon_incremental_compare_table.csv", index=False)

    near_tau = float(compare.loc[compare["score_name"].eq("near_only_0_60"), "tau"].iloc[0])
    far_tau = float(compare.loc[compare["score_name"].eq("h120_far_60_120"), "tau"].iloc[0])
    near_event = _event_summary_for_score(
        test_index,
        test["near_score"],
        test["far_label"],
        name="near_only_0_60",
        tau=near_tau,
        tau_off=max(near_tau - float(args.tau_off_margin), 0.0),
        min_on_rows=args.min_on_rows,
        min_off_rows=args.min_off_rows,
    )
    far_event = _event_summary_for_score(
        test_index,
        test["far_score"],
        test["far_label"],
        name="h120_far_60_120",
        tau=far_tau,
        tau_off=max(far_tau - float(args.tau_off_margin), 0.0),
        min_on_rows=args.min_on_rows,
        min_off_rows=args.min_off_rows,
    )
    event_summary = pd.concat([near_event, far_event], ignore_index=True)
    event_summary.to_csv(dirs["out"] / "horizon_incremental_event_summary.csv", index=False)
    event_summary.to_csv(dirs["raw"] / "horizon_incremental_event_summary.csv", index=False)

    unique = _unique_detection_table(
        test["far_label"],
        test["near_score"],
        test["far_score"],
        near_tau=near_tau,
        far_tau=far_tau,
    )
    unique.to_csv(dirs["out"] / "horizon_incremental_unique_detection_table.csv", index=False)
    unique.to_csv(dirs["raw"] / "horizon_incremental_unique_detection_table.csv", index=False)

    scenario_near = build_scenario_table(test_index, test["near_score"], test["far_label"], near_tau, split="test")
    scenario_far = build_scenario_table(test_index, test["far_score"], test["far_label"], far_tau, split="test")
    scenario_near.insert(1, "score_name", "near_only_0_60")
    scenario_far.insert(1, "score_name", "h120_far_60_120")
    scenario = pd.concat([scenario_near, scenario_far], ignore_index=True)
    scenario.to_csv(dirs["out"] / "horizon_incremental_scenario_table.csv", index=False)
    scenario.to_csv(dirs["raw"] / "horizon_incremental_scenario_table.csv", index=False)

    near_row = compare[compare["score_name"].eq("near_only_0_60")].iloc[0]
    far_row = compare[compare["score_name"].eq("h120_far_60_120")].iloc[0]
    near_ev = event_summary[event_summary["score_name"].eq("near_only_0_60")].iloc[0]
    far_ev = event_summary[event_summary["score_name"].eq("h120_far_60_120")].iloc[0]
    delta = {
        "precision": float(far_row["test_precision"] - near_row["test_precision"]),
        "recall": float(far_row["test_recall"] - near_row["test_recall"]),
        "fpr": float(far_row["test_fpr"] - near_row["test_fpr"]),
        "f1": float(far_row["test_f1"] - near_row["test_f1"]),
        "auc": float(far_row["test_auc"] - near_row["test_auc"]),
        "event_recall": float(far_ev["truth_recall"] - near_ev["truth_recall"]),
        "false_alarm_per_day": float(far_ev["false_alarm_episodes_per_day"] - near_ev["false_alarm_episodes_per_day"]),
    }
    far_only_true_share = float(unique.loc[unique["bucket_group"].eq("far_only_fire"), "share_of_all_true_events"].iloc[0])
    verdict = (
        "The h120 far score adds incremental supervisory value over the near-only comparator."
        if delta["f1"] > 0.02 and delta["auc"] > 0.02 and far_only_true_share > 0.05
        else "The h120 far score does not show a strong incremental advantage over the near-only comparator under this audit."
    )
    payload = {
        "model_dir": _rel(args.model_dir),
        "dataset_dir": _rel(args.dataset_dir),
        "target": "actual max pressure norm over [60,120]min >= 1.0",
        "selection_split": "validation",
        "report_split": "test",
        "selection_objective": args.selection_objective,
        "near_tau": near_tau,
        "far_tau": far_tau,
        "incremental_delta": delta,
        "far_only_true_event_share": far_only_true_share,
        "verdict": verdict,
    }
    (dirs["out"] / "horizon_incremental_value_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_docs(dirs, compare, event_summary, unique, scenario_near, scenario_far, payload)

    print("h120 vs h60 supervisory incremental audit complete")
    print(f"  near tau={near_tau:.3f}  far tau={far_tau:.3f}")
    print(
        "  near test P/R/FPR/F1/AUC = "
        f"{near_row['test_precision']:.3f}/{near_row['test_recall']:.3f}/"
        f"{near_row['test_fpr']:.3f}/{near_row['test_f1']:.3f}/{near_row['test_auc']:.3f}"
    )
    print(
        "  far  test P/R/FPR/F1/AUC = "
        f"{far_row['test_precision']:.3f}/{far_row['test_recall']:.3f}/"
        f"{far_row['test_fpr']:.3f}/{far_row['test_f1']:.3f}/{far_row['test_auc']:.3f}"
    )
    print(f"  delta F1={delta['f1']:+.3f} AUC={delta['auc']:+.3f} event_recall={delta['event_recall']:+.3f}")
    print(f"  outputs -> {_rel(args.out_dir)}")


if __name__ == "__main__":
    main()
