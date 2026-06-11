#!/usr/bin/env python3
"""Audit whether true 60-120 minute future wind contains extra control value.

This is an oracle-data diagnostic only. It does not run or modify the
controller. The goal is to estimate how often the second hour contains pressure
relief, intensification, or direction reversal that is not visible in the first
60 minutes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.ballast_planner import PlannerConfig, norm_term, pressure_proxy_vec  # noqa: E402


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(str(value).strip(), TIMESTAMP_FMT)


def _load_canonical(path: Path, split: str) -> tuple[list[datetime], np.ndarray]:
    timestamps: list[datetime] = []
    uv_rows: list[tuple[float, float]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("split", "")).strip() != split:
                continue
            timestamps.append(_parse_ts(row["timestamp"]))
            uv_rows.append((float(row["wind_u_ms"]), float(row["wind_v_ms"])))
    return timestamps, np.asarray(uv_rows, dtype=np.float32)


def _blocks_for_window(uv_12: np.ndarray, cfg: PlannerConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, start, end in (
        ("0_20min", 0, 2),
        ("20_40min", 2, 4),
        ("40_60min", 4, 6),
        ("60_80min", 6, 8),
        ("80_100min", 8, 10),
        ("100_120min", 10, 12),
    ):
        vec = pressure_proxy_vec(uv_12[start:end], cfg)
        rows.append(
            {
                "block": name,
                "vec": vec,
                "norm": norm_term(vec, cfg),
            }
        )
    return rows


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1e-9)


def _direction_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.linalg.norm(a)) <= 1e-9 or float(np.linalg.norm(b)) <= 1e-9:
        return 0.0
    cosv = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))))


def _analyze_window(ts: datetime, uv_12: np.ndarray, cfg: PlannerConfig) -> dict[str, Any]:
    blocks = _blocks_for_window(uv_12, cfg)
    norms = np.asarray([b["norm"] for b in blocks], dtype=float)
    vecs = [np.asarray(b["vec"], dtype=float) for b in blocks]
    near = norms[:3]
    far = norms[3:]
    near_max = float(np.max(near))
    near_last = float(near[-1])
    near_mean = float(np.mean(near))
    far_min = float(np.min(far))
    far_max = float(np.max(far))
    far_mean = float(np.mean(far))
    far_last = float(far[-1])
    near_trend = float(near[-1] - near[0])
    far_trend = float(far[-1] - far[0])
    dir_60_120 = _direction_delta_deg(vecs[2], vecs[5])
    dot_60_120 = float(np.dot(vecs[2], vecs[5]))

    hidden_relief = (
        near_max >= 1.0
        and far_min <= max(0.75, near_max - 0.35)
        and far_min <= near_last - 0.25
    )
    hidden_intensification = (
        near_max <= 0.85
        and far_max >= 1.10
        and far_max >= near_max + 0.35
    )
    far_reversal = near_max >= 0.55 and far_max >= 0.55 and dot_60_120 < 0.0
    far_direction_shift = near_max >= 0.55 and far_max >= 0.55 and dir_60_120 >= 60.0

    return {
        "timestamp": ts.strftime(TIMESTAMP_FMT),
        "near_norm_0_20": float(norms[0]),
        "near_norm_20_40": float(norms[1]),
        "near_norm_40_60": float(norms[2]),
        "far_norm_60_80": float(norms[3]),
        "far_norm_80_100": float(norms[4]),
        "far_norm_100_120": float(norms[5]),
        "near_max_norm": near_max,
        "near_mean_norm": near_mean,
        "near_last_norm": near_last,
        "far_min_norm": far_min,
        "far_max_norm": far_max,
        "far_mean_norm": far_mean,
        "far_last_norm": far_last,
        "far_min_over_near_max": _safe_ratio(far_min, near_max),
        "far_max_over_near_max": _safe_ratio(far_max, near_max),
        "near_trend_norm": near_trend,
        "far_trend_norm": far_trend,
        "dir_shift_60_to_120_deg": dir_60_120,
        "dot_60_to_120": dot_60_120,
        "hidden_relief_60_120": int(hidden_relief),
        "hidden_intensification_60_120": int(hidden_intensification),
        "far_reversal_60_120": int(far_reversal),
        "far_direction_shift_60_120": int(far_direction_shift),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    near = np.asarray([r["near_max_norm"] for r in rows], dtype=float)
    far_min = np.asarray([r["far_min_norm"] for r in rows], dtype=float)
    far_max = np.asarray([r["far_max_norm"] for r in rows], dtype=float)
    hidden_relief = np.asarray([r["hidden_relief_60_120"] for r in rows], dtype=bool)
    hidden_intensity = np.asarray([r["hidden_intensification_60_120"] for r in rows], dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axes[0].scatter(near, far_min, s=4, alpha=0.22)
    axes[0].scatter(near[hidden_relief], far_min[hidden_relief], s=6, alpha=0.55, label="hidden relief")
    axes[0].plot([0, max(near.max(), 1.0)], [0, max(near.max(), 1.0)], color="black", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("0-60min max norm")
    axes[0].set_ylabel("60-120min min norm")
    axes[0].set_title("Second-hour relief visibility")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].scatter(near, far_max, s=4, alpha=0.22)
    axes[1].scatter(near[hidden_intensity], far_max[hidden_intensity], s=6, alpha=0.55, color="tab:red", label="hidden intensification")
    axes[1].plot([0, max(far_max.max(), 1.0)], [0, max(far_max.max(), 1.0)], color="black", lw=0.8, alpha=0.5)
    axes[1].set_xlabel("0-60min max norm")
    axes[1].set_ylabel("60-120min max norm")
    axes[1].set_title("Second-hour intensification")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    bins = np.linspace(0, max(float(near.max()), float(far_max.max()), 1.0), 40)
    axes[2].hist(near, bins=bins, alpha=0.55, label="0-60 max")
    axes[2].hist(far_max, bins=bins, alpha=0.55, label="60-120 max")
    axes[2].set_xlabel("Pressure norm")
    axes[2].set_ylabel("Window count")
    axes[2].set_title("Near vs far pressure distribution")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    fig.suptitle("Oracle 120min horizon value audit from true FINO1 wind")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / "oracle_120min_horizon_value_audit.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-windows", type=int, default=30000)
    parser.add_argument("--case-csv", default="outputs/wind_prediction/postureaction_probe_cases_2case.csv")
    parser.add_argument("--out-dir", default="outputs/wind_prediction/oracle_120min_horizon_value_audit_v1")
    args = parser.parse_args()

    dataset_dir = (REPO_ROOT / args.dataset_dir).resolve()
    meta = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    canonical = dataset_dir.parent / "fino1_platform_10min" / "canonical_observations_10min.csv.gz"
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps, uv = _load_canonical(canonical, args.split)
    index = {ts: i for i, ts in enumerate(timestamps)}
    cfg = PlannerConfig(pressure_sign_multiplier=-1.0, envelope_use_discount=False)

    valid_starts = [i for i in range(0, len(timestamps) - 12) if timestamps[i + 12] - timestamps[i] == timedelta(minutes=120)]
    if args.max_windows > 0 and len(valid_starts) > args.max_windows:
        pick = np.linspace(0, len(valid_starts) - 1, num=int(args.max_windows), dtype=int)
        valid_starts = [valid_starts[int(i)] for i in pick]

    rows = [_analyze_window(timestamps[i], uv[i : i + 12], cfg) for i in valid_starts]
    _write_csv(out_dir / "oracle_120min_window_metrics.csv", rows)
    fig_path = _plot(out_dir, rows)

    case_rows: list[dict[str, Any]] = []
    case_csv = (REPO_ROOT / args.case_csv).resolve()
    if case_csv.exists():
        with case_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = _parse_ts(row["timestamp"])
                idx = index.get(ts)
                rec = {"case_id": row.get("case_id", ""), "label": row.get("label", ""), "timestamp": row["timestamp"]}
                if idx is None or idx + 12 > len(uv):
                    rec["status"] = "missing_or_incomplete_120min"
                else:
                    rec.update(_analyze_window(ts, uv[idx : idx + 12], cfg))
                    rec["status"] = "ok"
                case_rows.append(rec)
        _write_csv(out_dir / "oracle_120min_case_metrics.csv", case_rows)

    n = max(len(rows), 1)
    summary = {
        "split": args.split,
        "n_windows": len(rows),
        "max_windows": int(args.max_windows),
        "hidden_relief_ratio": float(np.mean([r["hidden_relief_60_120"] for r in rows])) if rows else 0.0,
        "hidden_intensification_ratio": float(np.mean([r["hidden_intensification_60_120"] for r in rows])) if rows else 0.0,
        "far_reversal_ratio": float(np.mean([r["far_reversal_60_120"] for r in rows])) if rows else 0.0,
        "far_direction_shift_ratio": float(np.mean([r["far_direction_shift_60_120"] for r in rows])) if rows else 0.0,
        "near_max_mean": float(np.mean([r["near_max_norm"] for r in rows])) if rows else 0.0,
        "far_max_mean": float(np.mean([r["far_max_norm"] for r in rows])) if rows else 0.0,
        "figure": str(fig_path),
        "future_resolution_minutes": float(meta.get("input_resolution_minutes", 10.0)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out_dir / 'oracle_120min_window_metrics.csv'}", flush=True)
    print(f"wrote {out_dir / 'oracle_120min_case_metrics.csv'}", flush=True)
    print(f"wrote {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
