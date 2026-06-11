#!/usr/bin/env python3
"""Five-layer diagnostic summary for learned vs reactive_current control."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _q(values: np.ndarray, p: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values.astype(float), p))


def _episodes(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    m = mask.astype(bool)
    return int(np.sum(m & np.concatenate([[True], ~m[:-1]])))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_one(folder: Path, subdir: str, case_prefix: str, suffix: str) -> Path | None:
    files = sorted((folder / subdir).glob(f"{case_prefix}_*_{suffix}.csv"))
    return files[0] if files else None


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        raw = row.get(key, "")
        if raw == "":
            return default
        return float(raw)
    except Exception:
        return default


def _case_metrics(case: str, source: str, ts: list[dict[str, str]], log: list[dict[str, str]]) -> dict[str, Any]:
    pitch = np.asarray([_f(r, "pitch_deg") for r in ts], dtype=float)
    roll = np.asarray([_f(r, "roll_deg") for r in ts], dtype=float)
    fallback = np.asarray([_f(r, "preview_primary_safety_fallback") > 0.5 for r in ts], dtype=bool)
    pump_rate = np.asarray([_f(r, "pump_total_rate_m3_min") for r in ts], dtype=float)
    cmd_clip = np.asarray([_f(r, "alloc_clip_any") > 0.5 for r in ts], dtype=bool)
    target_err = np.asarray(
        [
            np.mean(
                [
                    abs(_f(r, "err_tank1_kg")),
                    abs(_f(r, "err_tank2_kg")),
                    abs(_f(r, "err_tank3_kg")),
                ]
            )
            for r in ts
        ],
        dtype=float,
    )

    first_actions = [str(r.get("first_action", "")) for r in log]
    action_active = [a for a in first_actions if a and a != "hold"]
    target_refreshed = np.asarray([_f(r, "prediction_primary_target_refreshed") > 0.5 for r in log], dtype=bool)
    target_reused = np.asarray([_f(r, "prediction_primary_target_reused") > 0.5 for r in log], dtype=bool)
    if len(log) > 1:
        targets = np.asarray(
            [
                [
                    _f(r, "prediction_primary_target_t1_kg"),
                    _f(r, "prediction_primary_target_t2_kg"),
                    _f(r, "prediction_primary_target_t3_kg"),
                ]
                for r in log
            ],
            dtype=float,
        )
        target_step_change = float(np.mean(np.linalg.norm(np.diff(targets, axis=0), axis=1) > 1.0))
    else:
        target_step_change = float("nan")

    return {
        "case": case,
        "source": source,
        "pump_work_m3": float(np.sum(pump_rate) / 60.0) if pump_rate.size else 0.0,
        "pitch_abs_p95_deg": _q(np.abs(pitch), 0.95),
        "roll_abs_p95_deg": _q(np.abs(roll), 0.95),
        "pitch_abs_mean_deg": float(np.mean(np.abs(pitch))) if pitch.size else float("nan"),
        "roll_abs_mean_deg": float(np.mean(np.abs(roll))) if roll.size else float("nan"),
        "planner_bucket_count": int(len(log)),
        "planner_active_bucket_ratio": float(len(action_active) / max(len(first_actions), 1)),
        "planner_first_action_counts": ";".join(
            f"{a}:{first_actions.count(a)}" for a in sorted(set(first_actions)) if a
        ),
        "target_refresh_ratio": float(np.mean(target_refreshed)) if target_refreshed.size else float("nan"),
        "target_reuse_ratio": float(np.mean(target_reused)) if target_reused.size else float("nan"),
        "target_step_change_ratio": float(target_step_change),
        "execution_clip_ratio": float(np.mean(cmd_clip)) if cmd_clip.size else 0.0,
        "target_error_mean_kg": float(np.mean(target_err)) if target_err.size else 0.0,
        "target_error_p95_kg": _q(target_err, 0.95),
        "safety_fallback_ratio": float(np.mean(fallback)) if fallback.size else 0.0,
        "safety_fallback_episodes": _episodes(fallback),
        "safety_fallback_pump_m3": float(pump_rate[fallback].sum() / 60.0) if pump_rate.size else 0.0,
        "safety_fallback_pump_share": float(pump_rate[fallback].sum() / max(pump_rate.sum(), 1e-9)) if pump_rate.size else 0.0,
    }


def _pair_rows(learned_dir: Path, reactive_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        ("fr_relief_09", "01_fr_relief_09"),
        ("lowrisk_clean", "02_lowrisk_clean"),
    ]
    for case_name, prefix in cases:
        for source, folder in [("learned", learned_dir), ("reactive_current", reactive_dir)]:
            ts_path = _find_one(folder, "timeseries", prefix, "prediction_primary_econ_timeseries")
            log_path = _find_one(folder, "planner_logs", prefix, "prediction_primary_econ_planner_log")
            if ts_path is None or log_path is None:
                continue
            rows.append(_case_metrics(case_name, source, _read_csv_rows(ts_path), _read_csv_rows(log_path)))
    return rows


def _compare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {(r["case"], r["source"]): r for r in rows}
    out: list[dict[str, Any]] = []
    metrics = [
        "pump_work_m3",
        "pitch_abs_p95_deg",
        "roll_abs_p95_deg",
        "planner_active_bucket_ratio",
        "target_refresh_ratio",
        "execution_clip_ratio",
        "target_error_p95_kg",
        "safety_fallback_ratio",
        "safety_fallback_pump_share",
    ]
    for case in sorted({r["case"] for r in rows}):
        l = by.get((case, "learned"))
        r = by.get((case, "reactive_current"))
        if not l or not r:
            continue
        row = {"case": case}
        for m in metrics:
            row[f"learned_{m}"] = l[m]
            row[f"reactive_{m}"] = r[m]
            row[f"delta_{m}"] = float(l[m]) - float(r[m])
        out.append(row)
    return out


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

    metrics = [
        ("pump_work_m3", "Pump work (m3)"),
        ("pitch_abs_p95_deg", "|Pitch| p95 (deg)"),
        ("roll_abs_p95_deg", "|Roll| p95 (deg)"),
        ("safety_fallback_ratio", "Safety fallback ratio"),
        ("execution_clip_ratio", "Execution clip ratio"),
    ]
    cases = sorted({r["case"] for r in rows})
    sources = ["learned", "reactive_current"]
    fig, axes = plt.subplots(len(metrics), len(cases), figsize=(11, 12), squeeze=False)
    for col, case in enumerate(cases):
        case_rows = {r["source"]: r for r in rows if r["case"] == case}
        for row_idx, (key, title) in enumerate(metrics):
            ax = axes[row_idx, col]
            vals = [case_rows.get(s, {}).get(key, np.nan) for s in sources]
            ax.bar(["learned", "reactive"], vals, color=["#1f77b4", "#ff7f0e"])
            ax.set_title(f"{case}: {title}")
            ax.grid(True, axis="y", alpha=0.25)
            if row_idx == len(metrics) - 1:
                ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Five-layer diagnostic summary: learned vs reactive_current")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "five_layer_learned_vs_reactive_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-dir", default="outputs/wind_prediction/fivelayer_learned_vs_reactive_learned_2case_2h_v1")
    parser.add_argument("--reactive-dir", default="outputs/wind_prediction/fivelayer_learned_vs_reactive_reactive_2case_2h_v1")
    parser.add_argument("--out-dir", default="outputs/wind_prediction/fivelayer_learned_vs_reactive_diagnosis_v1")
    args = parser.parse_args()

    learned_dir = (REPO_ROOT / args.learned_dir).resolve()
    reactive_dir = (REPO_ROOT / args.reactive_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _pair_rows(learned_dir, reactive_dir)
    compare = _compare_rows(rows)
    _write_csv(out_dir / "five_layer_source_metrics.csv", rows)
    _write_csv(out_dir / "five_layer_pair_compare.csv", compare)
    fig_path = _plot(out_dir, rows)
    print(f"wrote {out_dir / 'five_layer_source_metrics.csv'}")
    print(f"wrote {out_dir / 'five_layer_pair_compare.csv'}")
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
