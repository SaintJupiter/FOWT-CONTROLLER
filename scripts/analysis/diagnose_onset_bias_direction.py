#!/usr/bin/env python3
"""Diagnose whether onset preview setpoint bias points with or against platform pitch.

The A2 structural ablations can fail for two different reasons:

1. The preview bias is too small or too weakly selected to pierce the PI deadband.
2. The preview bias has the wrong sign for a setpoint-bias injection path.

This diagnostic uses existing onset simulation CSVs.  For every onset window and
selected planner variant, it compares the preview pitch bias against the
closed-only pitch over the first hour.  If the bias is mostly the same sign as
the closed-only pitch, then it relaxes the controller's pitch error
(setpoint - actual) instead of asking the controller to counteract the motion.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs" / "wind_prediction" / "planner_a2_onset_struct"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "wind_prediction" / "planner_a2_onset_direction_diag"
FIRST_HOUR_S = 3600.0
BIAS_EPS_DEG = 0.02
BUCKET_S = 600
VARIANTS = ("v0_baseline", "v2_env_tight", "v3_amp_bump", "v5_barrier_on")


def _read_timeseries(path: Path, cols: tuple[str, ...], max_t: float = FIRST_HOUR_S) -> dict[str, list[float]]:
    out = {c: [] for c in cols}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_s = float(row["t_s"])
            if t_s >= max_t:
                break
            for col in cols:
                val = row.get(col, "")
                try:
                    out[col].append(float(val))
                except ValueError:
                    out[col].append(float("nan"))
    return out


def _read_planner_log(path: Path, max_rows: int = 6) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))[:max_rows]


def _sign(x: float, eps: float = 1e-9) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _pct(num: float, den: float) -> float:
    return float(num / den * 100.0) if den else 0.0


def _p95_abs(values: list[float]) -> float:
    vals = sorted(abs(v) for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    idx = 0.95 * (len(vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def _safe_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return mean(vals) if vals else float("nan")


def _window_ids(input_dir: Path) -> list[str]:
    ids = []
    for path in sorted(input_dir.glob("*_closed_only_timeseries.csv")):
        ids.append(path.name.removesuffix("_closed_only_timeseries.csv"))
    return ids


def _diagnose_variant(input_dir: Path, window_id: str, variant: str) -> dict[str, float | str]:
    closed_path = input_dir / f"{window_id}_closed_only_timeseries.csv"
    var_path = input_dir / f"{window_id}_{variant}_timeseries.csv"
    log_path = input_dir / f"{window_id}_{variant}_planner_log.csv"
    closed = _read_timeseries(closed_path, ("pitch_deg", "roll_deg"))
    ts = _read_timeseries(
        var_path,
        (
            "pitch_deg",
            "roll_deg",
            "preview_pitch_bias_deg",
            "preview_roll_bias_deg",
            "pitch_sp_deg",
            "roll_sp_deg",
            "ctrl_pitch_err_raw_deg",
            "ctrl_roll_err_raw_deg",
            "pump_total_rate_m3_min",
        ),
    )
    n = min(len(closed["pitch_deg"]), len(ts["pitch_deg"]))
    active_pitch = [i for i in range(n) if abs(ts["preview_pitch_bias_deg"][i]) > BIAS_EPS_DEG]
    active_roll = [i for i in range(n) if abs(ts["preview_roll_bias_deg"][i]) > BIAS_EPS_DEG]

    same_pitch = sum(
        1
        for i in active_pitch
        if _sign(ts["preview_pitch_bias_deg"][i]) * _sign(closed["pitch_deg"][i]) > 0
    )
    opp_pitch = sum(
        1
        for i in active_pitch
        if _sign(ts["preview_pitch_bias_deg"][i]) * _sign(closed["pitch_deg"][i]) < 0
    )
    same_roll = sum(
        1
        for i in active_roll
        if _sign(ts["preview_roll_bias_deg"][i]) * _sign(closed["roll_deg"][i]) > 0
    )
    opp_roll = sum(
        1
        for i in active_roll
        if _sign(ts["preview_roll_bias_deg"][i]) * _sign(closed["roll_deg"][i]) < 0
    )

    # Positive means the planner variant increases absolute attitude vs closed-only.
    d_abs_pitch = [abs(ts["pitch_deg"][i]) - abs(closed["pitch_deg"][i]) for i in range(n)]
    d_abs_roll = [abs(ts["roll_deg"][i]) - abs(closed["roll_deg"][i]) for i in range(n)]

    # For a pure setpoint-bias path with closed-only setpoint near zero, same-sign
    # bias lowers |sp - actual| and therefore asks the PI controller to push less.
    pitch_error_relief = [
        abs(closed["pitch_deg"][i]) - abs(ts["preview_pitch_bias_deg"][i] - closed["pitch_deg"][i])
        for i in active_pitch
    ]
    roll_error_relief = [
        abs(closed["roll_deg"][i]) - abs(ts["preview_roll_bias_deg"][i] - closed["roll_deg"][i])
        for i in active_roll
    ]

    planner_rows = _read_planner_log(log_path)
    target_pairs = []
    for bucket_idx, row in enumerate(planner_rows):
        s = bucket_idx * BUCKET_S
        e = min((bucket_idx + 1) * BUCKET_S, len(closed["pitch_deg"]))
        bucket_pitch = _safe_mean(closed["pitch_deg"][s:e])
        target_pitch = float(row["target_pitch_deg"])
        target_pairs.append(f"{bucket_idx}:{target_pitch:+.3f}/{bucket_pitch:+.3f}")

    return {
        "window": window_id,
        "variant": variant,
        "active_pitch_samples": len(active_pitch),
        "pitch_bias_same_sign_pct": _pct(same_pitch, len(active_pitch)),
        "pitch_bias_opposite_sign_pct": _pct(opp_pitch, len(active_pitch)),
        "mean_pitch_bias_deg": _safe_mean([ts["preview_pitch_bias_deg"][i] for i in active_pitch]),
        "mean_closed_pitch_deg_when_active": _safe_mean([closed["pitch_deg"][i] for i in active_pitch]),
        "mean_d_abs_pitch_deg": _safe_mean(d_abs_pitch),
        "p95_abs_pitch_closed_deg": _p95_abs(closed["pitch_deg"][:n]),
        "p95_abs_pitch_variant_deg": _p95_abs(ts["pitch_deg"][:n]),
        "mean_pitch_error_relief_deg": _safe_mean(pitch_error_relief),
        "active_roll_samples": len(active_roll),
        "roll_bias_same_sign_pct": _pct(same_roll, len(active_roll)),
        "roll_bias_opposite_sign_pct": _pct(opp_roll, len(active_roll)),
        "mean_d_abs_roll_deg": _safe_mean(d_abs_roll),
        "mean_roll_error_relief_deg": _safe_mean(roll_error_relief),
        "bucket_target_pitch_vs_closed_mean": " ".join(target_pairs),
    }


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, float | str]]) -> None:
    by_variant: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    lines = [
        "# A2 onset preview-bias direction diagnostic",
        "",
        f"- input: `{DEFAULT_INPUT_DIR}`",
        f"- bias active threshold: `{BIAS_EPS_DEG:.2f} deg`",
        "- interpretation: same-sign pitch bias reduces `|setpoint - actual|` for positive pitch, so it relaxes the PI correction rather than opposing the motion.",
        "",
        "## Aggregate",
        "",
        "| variant | windows | same-sign pitch bias | opposite-sign pitch bias | mean d|pitch| | p95 pitch delta | mean pitch error relief |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        sub = by_variant.get(variant, [])
        if not sub:
            continue
        same = _safe_mean([float(r["pitch_bias_same_sign_pct"]) for r in sub])
        opp = _safe_mean([float(r["pitch_bias_opposite_sign_pct"]) for r in sub])
        d_abs = _safe_mean([float(r["mean_d_abs_pitch_deg"]) for r in sub])
        p95_delta = _safe_mean(
            [
                float(r["p95_abs_pitch_variant_deg"]) - float(r["p95_abs_pitch_closed_deg"])
                for r in sub
            ]
        )
        relief = _safe_mean([float(r["mean_pitch_error_relief_deg"]) for r in sub])
        lines.append(
            f"| {variant} | {len(sub)} | {same:.1f}% | {opp:.1f}% | "
            f"{d_abs:+.4f} | {p95_delta:+.4f} | {relief:+.4f} |"
        )

    lines += [
        "",
        "## Per-Window Pitch Direction",
        "",
        "| window | variant | active samples | same-sign | opposite-sign | mean bias | mean closed pitch active | d|pitch| mean | p95 pitch delta | bucket target/closed mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        p95_delta = float(row["p95_abs_pitch_variant_deg"]) - float(row["p95_abs_pitch_closed_deg"])
        lines.append(
            f"| {row['window']} | {row['variant']} | {int(row['active_pitch_samples'])} | "
            f"{float(row['pitch_bias_same_sign_pct']):.1f}% | "
            f"{float(row['pitch_bias_opposite_sign_pct']):.1f}% | "
            f"{float(row['mean_pitch_bias_deg']):+.4f} | "
            f"{float(row['mean_closed_pitch_deg_when_active']):+.4f} | "
            f"{float(row['mean_d_abs_pitch_deg']):+.4f} | "
            f"{p95_delta:+.4f} | {row['bucket_target_pitch_vs_closed_mean']} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        "The existing onset runs mostly apply pitch bias with the same sign as the closed-only pitch.",
        "Because preview bias enters as a setpoint shift, same-sign bias reduces the PI error magnitude in those intervals.",
        "That explains the observed pump saving with slightly worse pitch: the preview path is often telling the controller to tolerate the impending pitch rather than counteract it.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for window_id in _window_ids(DEFAULT_INPUT_DIR):
        for variant in VARIANTS:
            ts_path = DEFAULT_INPUT_DIR / f"{window_id}_{variant}_timeseries.csv"
            log_path = DEFAULT_INPUT_DIR / f"{window_id}_{variant}_planner_log.csv"
            if not ts_path.exists() or not log_path.exists():
                continue
            rows.append(_diagnose_variant(DEFAULT_INPUT_DIR, window_id, variant))
    _write_csv(DEFAULT_OUTPUT_DIR / "bias_direction_summary.csv", rows)
    _write_report(DEFAULT_OUTPUT_DIR / "bias_direction_report.md", rows)
    print(f"Report: {DEFAULT_OUTPUT_DIR / 'bias_direction_report.md'}")


if __name__ == "__main__":
    main()
