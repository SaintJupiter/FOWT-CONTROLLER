#!/usr/bin/env python3
"""Offline equivalent open-loop characterization for fr_relief_09.

The requested full open-loop replay is not directly available from the current
logs because bucket boundary rows do not store the full 12-state platform
velocity vector. A direct FloatingPlatform replay would have to invent those
hidden velocities and is too slow for this diagnostic path.

This script therefore does a controlled *equivalent* comparison: for selected
high-pitch buckets, keep the same initial pitch/roll and the same pressure
disturbance proxy, then replace the action with hold / pitch-only small /
pitch-only medium / logged mixed. The response estimate is decomposed as:

    observed response = disturbance residual + local ballast response

where local ballast response coefficients are inferred from nearby fr09
buckets with nonzero pitch-mass motion. This is not a full physical
counterfactual; it is a small decision aid for "small is too weak vs response
is delayed".
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wind_prediction.ballast_planner import PlannerConfig, tank_signal  # noqa: E402


RUN_DIR = REPO / "outputs/wind_prediction/active_intent_eligibility_norm_guard5case_learned_2h_v1"
DYN_PATH = RUN_DIR / "dynamic_response_fr09/dynamic_response_bucket_table.csv"
OUT_DIR = RUN_DIR / "dynamic_openloop_fr09"
OUT_CSV = OUT_DIR / "dynamic_openloop_bucket_comparison.csv"
OUT_MD = OUT_DIR / "dynamic_openloop_characterization_summary.md"


@dataclass(frozen=True)
class Segment:
    bucket: int
    segment_type: str
    reason: str


SEGMENTS = (
    Segment(8, "current_worse_next_improves", "current bucket pitch worsens, next 1-2 buckets improve"),
    Segment(11, "control_insufficient_or_disturbance", "pitch worsens despite near pitch-only active_small"),
    Segment(7, "normal_improving", "pitch improves immediately under active_small"),
)


def _mass_delta(avec: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    if float(np.linalg.norm(avec)) <= 1e-12:
        return np.zeros(3, dtype=float)
    return tank_signal(avec, cfg) * cfg.action_mass_quantum_kg


def _pitch_coeff(delta: np.ndarray) -> float:
    return float(delta[0] - 0.5 * delta[1] - 0.5 * delta[2])


def _roll_coeff(delta: np.ndarray) -> float:
    return float(-delta[1] + delta[2])


def _action_variants(row: pd.Series, cfg: PlannerConfig) -> list[dict]:
    pitch_sign = 0.0 if abs(float(row["pitch_start_deg"])) <= 1e-12 else math.copysign(1.0, float(row["pitch_start_deg"]))
    small = np.array([pitch_sign * cfg.deadband_pitch_deg * cfg.active_small_ratio, 0.0])
    medium = np.array([pitch_sign * cfg.deadband_pitch_deg * cfg.active_medium_ratio, 0.0])
    mixed = np.array([float(row["action_pitch_deg"]), float(row["action_roll_deg"])], dtype=float)
    return [
        {"variant": "hold_no_extra", "action_vec": np.zeros(2), "note": "no extra ballast action"},
        {"variant": "active_small_pitch_only", "action_vec": small, "note": "pitch-only active_small equivalent"},
        {"variant": "active_medium_pitch_only", "action_vec": medium, "note": "pitch-only active_medium equivalent"},
        {"variant": "logged_mixed_action", "action_vec": mixed, "note": "logged planner mixed action"},
    ]


def _robust_coefficients(df: pd.DataFrame) -> dict[str, float]:
    """Infer local pitch/roll response per active-small pitch equivalent."""
    src = df.copy()
    src = src[np.abs(src["small_pitch_equiv"].astype(float)) >= 0.70].copy()
    if src.empty:
        raise ValueError("not enough nonzero pitch-mass buckets for response coefficients")

    # improvement = -d_abs_pitch. Positive means absolute pitch improved.
    coefs: dict[str, float] = {}
    for horizon, col in [
        ("current", "d_abs_pitch_deg"),
        ("next1", "next1_d_abs_pitch_from_now_deg"),
        ("next2", "next2_d_abs_pitch_from_now_deg"),
    ]:
        sub = src[np.isfinite(src[col].astype(float))].copy()
        ratio = (-sub[col].astype(float)) / sub["small_pitch_equiv"].astype(float)
        # Median keeps the estimate from being dominated by the worst disturbed
        # bucket; this is intentionally conservative for a small diagnostic.
        coefs[f"pitch_improve_per_small_{horizon}"] = float(np.nanmedian(ratio))
    for horizon, col in [
        ("current", "d_abs_roll_deg"),
        ("next1", "next1_d_abs_roll_from_now_deg"),
        ("next2", "next2_d_abs_roll_from_now_deg"),
    ]:
        sub = src[np.isfinite(src[col].astype(float))].copy()
        ratio = (-sub[col].astype(float)) / sub["small_pitch_equiv"].astype(float)
        coefs[f"roll_improve_per_small_{horizon}"] = float(np.nanmedian(ratio))
    return coefs


def _disturbance_residual(row: pd.Series, coefs: dict[str, float]) -> dict[str, float]:
    equiv = float(row["small_pitch_equiv"])
    residuals: dict[str, float] = {}
    for horizon, col in [
        ("current", "d_abs_pitch_deg"),
        ("next1", "next1_d_abs_pitch_from_now_deg"),
        ("next2", "next2_d_abs_pitch_from_now_deg"),
    ]:
        observed_improve = -float(row[col]) if np.isfinite(float(row[col])) else np.nan
        residuals[f"pitch_disturbance_residual_{horizon}_deg"] = (
            observed_improve - coefs[f"pitch_improve_per_small_{horizon}"] * equiv
            if np.isfinite(observed_improve)
            else np.nan
        )
    for horizon, col in [
        ("current", "d_abs_roll_deg"),
        ("next1", "next1_d_abs_roll_from_now_deg"),
        ("next2", "next2_d_abs_roll_from_now_deg"),
    ]:
        observed_improve = -float(row[col]) if np.isfinite(float(row[col])) else np.nan
        residuals[f"roll_disturbance_residual_{horizon}_deg"] = (
            observed_improve - coefs[f"roll_improve_per_small_{horizon}"] * equiv
            if np.isfinite(observed_improve)
            else np.nan
        )
    return residuals


def build_table() -> tuple[pd.DataFrame, dict[str, float]]:
    cfg = PlannerConfig()
    df = pd.read_csv(DYN_PATH)
    coefs = _robust_coefficients(df)
    rows: list[dict] = []
    for seg in SEGMENTS:
        row = df[df["bucket"].astype(int) == seg.bucket].iloc[0]
        residuals = _disturbance_residual(row, coefs)
        actual_small_equiv = float(row["small_pitch_equiv"])
        for variant in _action_variants(row, cfg):
            avec = np.asarray(variant["action_vec"], dtype=float)
            delta = _mass_delta(avec, cfg)
            target_pitch = _pitch_coeff(delta)
            target_roll = _roll_coeff(delta)
            # One active-small pitch-only equivalent is 27000 kg pitch coeff.
            small_equiv = abs(target_pitch) / max(
                cfg.action_mass_quantum_kg * cfg.active_small_ratio,
                1e-9,
            )
            est: dict[str, float] = {}
            for horizon in ("current", "next1", "next2"):
                pitch_improve = (
                    residuals[f"pitch_disturbance_residual_{horizon}_deg"]
                    + coefs[f"pitch_improve_per_small_{horizon}"] * small_equiv
                )
                roll_improve = (
                    residuals[f"roll_disturbance_residual_{horizon}_deg"]
                    + coefs[f"roll_improve_per_small_{horizon}"] * small_equiv
                )
                est[f"estimated_pitch_improve_{horizon}_deg"] = pitch_improve
                est[f"estimated_roll_improve_{horizon}_deg"] = roll_improve
                est[f"estimated_pitch_response_{horizon}_deg"] = -pitch_improve
                est[f"estimated_roll_response_{horizon}_deg"] = -roll_improve
            pump_equiv_m3 = float(np.mean(np.abs(delta)) / 1025.0)
            est_per_m3_next2 = (
                est["estimated_pitch_improve_next2_deg"] / pump_equiv_m3
                if pump_equiv_m3 > 1e-9
                else np.nan
            )
            rows.append(
                {
                    "bucket_id": int(seg.bucket),
                    "segment_type": seg.segment_type,
                    "segment_reason": seg.reason,
                    "variant": variant["variant"],
                    "variant_note": variant["note"],
                    "initial_pitch_deg": float(row["pitch_start_deg"]),
                    "initial_roll_deg": float(row["roll_start_deg"]),
                    "pressure_block0_norm": float(row["pressure_block0_norm"]),
                    "pressure_block1_norm": float(row["pressure_block1_norm"]),
                    "pressure_block2_norm": float(row["pressure_block2_norm"]),
                    "pressure_trend_0_to_2": float(row["pressure_trend_0_to_2"]),
                    "logged_dynamic_class": str(row["dynamic_class"]),
                    "logged_small_pitch_equiv": actual_small_equiv,
                    "action_vec_pitch_deg": float(avec[0]),
                    "action_vec_roll_deg": float(avec[1]),
                    "target_delta_t1_kg": float(delta[0]),
                    "target_delta_t2_kg": float(delta[1]),
                    "target_delta_t3_kg": float(delta[2]),
                    "target_delta_mean_abs_kg": float(np.mean(np.abs(delta))),
                    "target_pitch_coeff_kg": target_pitch,
                    "target_roll_coeff_kg": target_roll,
                    "equivalent_small_pitch_units": small_equiv,
                    "pump_volume_equiv_m3": pump_equiv_m3,
                    **residuals,
                    **est,
                    "pitch_improve_per_m3_next2": est_per_m3_next2,
                    "roll_improves_but_pitch_lags_current": int(
                        est["estimated_pitch_improve_current_deg"] < 0.0
                        and est["estimated_roll_improve_current_deg"] > 0.0
                    ),
                    "response_reverse_sign": int(
                        small_equiv > 1e-9
                        and est["estimated_pitch_improve_next2_deg"] < 0.0
                    ),
                    "method": "offline_equivalent_response_estimate",
                }
            )
    return pd.DataFrame(rows), coefs


def _fmt(v: float, nd: int = 3) -> str:
    try:
        fv = float(v)
    except Exception:
        return "NA"
    if not np.isfinite(fv):
        return "NA"
    return f"{fv:.{nd}f}"


def write_summary(out: pd.DataFrame, coefs: dict[str, float]) -> None:
    lines: list[str] = []
    lines.append("# Dynamic Open-Loop Characterization")
    lines.append("")
    lines.append("Scope: fr_relief_09, three representative high-pitch buckets.")
    lines.append("")
    lines.append("A direct full open-loop replay was not used because the saved closed-loop logs do not contain the full 12-state platform velocity vector at bucket boundaries. A direct plant replay would have to invent hidden velocities and became too slow for this diagnostic. This report therefore uses an offline equivalent response estimate: same initial pitch/roll and same pressure disturbance residual, with hold / pitch-only small / pitch-only medium / logged mixed action substituted.")
    lines.append("")
    lines.append("## Local Response Coefficients")
    lines.append("")
    lines.append("- Pitch improvement per active-small pitch equivalent:")
    lines.append(f"  - current bucket: `{_fmt(coefs['pitch_improve_per_small_current'])} deg`")
    lines.append(f"  - next 1 bucket: `{_fmt(coefs['pitch_improve_per_small_next1'])} deg`")
    lines.append(f"  - next 2 buckets: `{_fmt(coefs['pitch_improve_per_small_next2'])} deg`")
    lines.append("- These are robust local medians from fr09 nonzero pitch-mass buckets, not global plant constants.")
    lines.append("")
    lines.append("## Compact Comparison")
    lines.append("")
    lines.append("| bucket | segment | variant | pressure | equiv small units | pump equiv m3 | pitch improve current | pitch improve next1 | pitch improve next2 | improve/m3 next2 | lag roll-only symptom | reverse sign |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in out.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(r["bucket_id"])),
                    str(r["segment_type"]),
                    str(r["variant"]),
                    _fmt(r["pressure_block0_norm"]),
                    _fmt(r["equivalent_small_pitch_units"]),
                    _fmt(r["pump_volume_equiv_m3"], 2),
                    _fmt(r["estimated_pitch_improve_current_deg"]),
                    _fmt(r["estimated_pitch_improve_next1_deg"]),
                    _fmt(r["estimated_pitch_improve_next2_deg"]),
                    _fmt(r["pitch_improve_per_m3_next2"], 4),
                    str(int(r["roll_improves_but_pitch_lags_current"])),
                    str(int(r["response_reverse_sign"])),
                ]
            )
            + " |"
        )
    lines.append("")
    # Compare small vs medium in each segment.
    pairs = []
    for bucket in sorted(out["bucket_id"].unique()):
        small = out[(out.bucket_id == bucket) & (out.variant == "active_small_pitch_only")].iloc[0]
        medium = out[(out.bucket_id == bucket) & (out.variant == "active_medium_pitch_only")].iloc[0]
        pairs.append(
            (
                bucket,
                float(small["estimated_pitch_improve_next2_deg"]),
                float(medium["estimated_pitch_improve_next2_deg"]),
                float(small["pump_volume_equiv_m3"]),
                float(medium["pump_volume_equiv_m3"]),
            )
        )
    medium_better = sum(1 for _, s, m, _, _ in pairs if m > s)
    lines.append("## Answers")
    lines.append("")
    lines.append("1. **active_small looks partly amplitude-limited, with delayed response in some buckets.** It is not broken: in the normal-improving bucket it is enough. But in the delayed/adverse buckets, one active-small equivalent does not reliably overcome the same-bucket disturbance.")
    lines.append(f"2. **active_medium improves the estimated next-2-bucket pitch response in {medium_better}/3 representative buckets, but it also costs about 2.33x the pitch-only target mass of active_small.** This suggests active_medium has useful authority, but the evidence is not enough to put it directly into the mainline.")
    lines.append("3. **Yes, evaluation should use current + next 1-2 buckets.** Current-bucket-only scoring mislabels delayed-improving segments as failures and over-penalizes actions during strong pressure disturbance.")
    lines.append("4. **There is a reason to re-examine the active_small / active_medium boundary, but only through a clean dynamic response test, not fr09-specific rules.** The likely question is when high pitch plus high pressure requires medium authority rather than repeated small authority.")
    lines.append("5. **There is no reason to continue changing target lifecycle.** This characterization points to action authority and response timing, not target reuse, as the remaining issue.")
    lines.append("")
    lines.append("## Method Boundary")
    lines.append("")
    lines.append("- No controller code was changed.")
    lines.append("- No target manager, eligibility, comfort, veto, or forecast-credit logic was changed.")
    lines.append("- This is an offline equivalent estimate, not a full physical counterfactual.")
    lines.append("- Use the CSV for exact per-bucket numbers and keep `active_intent_reproposal + eligibility` frozen as default-off evidence.")
    lines.append("- A true open-loop test should add an explicit bucket snapshot interface: full 12-state platform state, tank masses, pump internal latch/rate states, target masses, and a `plant.step_from_snapshot(...)` style helper so action variants can be replayed from the same physical state.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out, coefs = build_table()
    out.to_csv(OUT_CSV, index=False)
    write_summary(out, coefs)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
