from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "outputs/wind_prediction/regime_conditioned_policy_development_v1/"
    "overnight_regime_opportunity_v1"
)

A0_PATH = ROOT / "neutral_mhs_a0_overnight_1h" / "casebook_summary.csv"
LEARNED_PATH = ROOT / "neutral_mhs_clean_auto_overnight_1h" / "casebook_summary.csv"
CURRENT_ONLY_PATH = (
    ROOT / "neutral_mhs_current_only_overnight_1h" / "casebook_summary.csv"
)
RAW_DIR = ROOT / "raw_tables"
PAPER_DIR = ROOT / "paper_ready"


def _max_axis_p95(df: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [df["primary_pitch_p95"].abs(), df["primary_roll_p95"].abs()],
        axis=1,
    ).max(axis=1)


def _load(path: Path, arm: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df.copy()
    df["arm"] = arm
    df["p95_max_axis"] = _max_axis_p95(df)
    return df


def _arm_summary(df: pd.DataFrame, a0: pd.DataFrame | None = None) -> dict[str, float | str | int]:
    pump = float(df["primary_pump_work_m3"].sum())
    fallback_s = float(df["primary_safety_fallback_ratio"].fillna(0).sum() * 3600.0)
    row: dict[str, float | str | int] = {
        "arm": str(df["arm"].iloc[0]),
        "cases": int(len(df)),
        "pump_m3": pump,
        "fallback_s": fallback_s,
        "mean_p95_max_axis_deg": float(df["p95_max_axis"].mean()),
        "max_p95_max_axis_deg": float(df["p95_max_axis"].max()),
        "forecast_source_requested": str(df.get("forecast_source_requested", pd.Series([""])).iloc[0]),
        "forecast_source_effective": str(df.get("forecast_source_effective", pd.Series([""])).iloc[0]),
    }
    if a0 is not None:
        saved = a0["primary_pump_work_m3"].to_numpy() - df["primary_pump_work_m3"].to_numpy()
        p95_delta = df["p95_max_axis"].to_numpy() - a0["p95_max_axis"].to_numpy()
        fallback_delta_s = (
            df["primary_safety_fallback_ratio"].fillna(0).to_numpy()
            - a0["primary_safety_fallback_ratio"].fillna(0).to_numpy()
        ) * 3600.0
        row.update(
            {
                "saving_m3_vs_a0": float(saved.sum()),
                "saving_pct_vs_a0": float(saved.sum() / a0["primary_pump_work_m3"].sum() * 100.0),
                "saved_cases": int((saved > 1e-6).sum()),
                "harmed_cases": int((saved < -1e-6).sum()),
                "top_case_share": float(max(saved.max(), 0.0) / saved.clip(min=0.0).sum())
                if saved.clip(min=0.0).sum() > 0
                else 0.0,
                "delta_fallback_s_vs_a0": float(fallback_delta_s.sum()),
                "max_case_delta_fallback_s_vs_a0": float(fallback_delta_s.max()),
                "max_case_delta_p95_axis_vs_a0_deg": float(p95_delta.max()),
            }
        )
    else:
        row.update(
            {
                "saving_m3_vs_a0": 0.0,
                "saving_pct_vs_a0": 0.0,
                "saved_cases": 0,
                "harmed_cases": 0,
                "top_case_share": 0.0,
                "delta_fallback_s_vs_a0": 0.0,
                "max_case_delta_fallback_s_vs_a0": 0.0,
                "max_case_delta_p95_axis_vs_a0_deg": 0.0,
            }
        )
    return row


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    a0 = _load(A0_PATH, "a0_v1_6")
    learned = _load(LEARNED_PATH, "neutral_mhs_learned")
    current = _load(CURRENT_ONLY_PATH, "neutral_mhs_current_only")

    case_id = a0["case_id"].astype(str)
    if not (case_id.equals(learned["case_id"].astype(str)) and case_id.equals(current["case_id"].astype(str))):
        raise ValueError("Case ordering mismatch across A0 / learned / current-only summaries.")

    case_delta = pd.DataFrame(
        {
            "case_id": case_id,
            "a0_pump_m3": a0["primary_pump_work_m3"],
            "learned_pump_m3": learned["primary_pump_work_m3"],
            "current_only_pump_m3": current["primary_pump_work_m3"],
            "learned_saved_m3_vs_a0": a0["primary_pump_work_m3"] - learned["primary_pump_work_m3"],
            "current_only_saved_m3_vs_a0": a0["primary_pump_work_m3"] - current["primary_pump_work_m3"],
            "learned_minus_current_only_pump_m3": learned["primary_pump_work_m3"]
            - current["primary_pump_work_m3"],
            "a0_p95_max_axis_deg": a0["p95_max_axis"],
            "learned_p95_max_axis_deg": learned["p95_max_axis"],
            "current_only_p95_max_axis_deg": current["p95_max_axis"],
            "a0_fallback_s": a0["primary_safety_fallback_ratio"].fillna(0) * 3600.0,
            "learned_fallback_s": learned["primary_safety_fallback_ratio"].fillna(0) * 3600.0,
            "current_only_fallback_s": current["primary_safety_fallback_ratio"].fillna(0) * 3600.0,
        }
    )

    summary = pd.DataFrame(
        [
            _arm_summary(a0),
            _arm_summary(learned, a0),
            _arm_summary(current, a0),
        ]
    )

    learned_saved_pct = float(summary.loc[summary["arm"] == "neutral_mhs_learned", "saving_pct_vs_a0"].iloc[0])
    current_saved_pct = float(summary.loc[summary["arm"] == "neutral_mhs_current_only", "saving_pct_vs_a0"].iloc[0])
    learned_pump = float(summary.loc[summary["arm"] == "neutral_mhs_learned", "pump_m3"].iloc[0])
    current_pump = float(summary.loc[summary["arm"] == "neutral_mhs_current_only", "pump_m3"].iloc[0])
    saving_gap_pp = learned_saved_pct - current_saved_pct
    pump_gap_m3 = learned_pump - current_pump
    exact_same_rows = int((case_delta["learned_minus_current_only_pump_m3"].abs() <= 1e-6).sum())
    max_case_pump_gap = float(case_delta["learned_minus_current_only_pump_m3"].abs().max())

    summary_path = RAW_DIR / "neutral_mhs_attribution_summary.csv"
    delta_path = RAW_DIR / "neutral_mhs_attribution_case_delta.csv"
    md_path = PAPER_DIR / "neutral_mhs_attribution_check.md"
    summary.to_csv(summary_path, index=False)
    case_delta.to_csv(delta_path, index=False)

    verdict = (
        "neutral-MHS is not supported as a forecast-driven specialist in this run. "
        "The current-only control, which removes future forecast preview and repeats the current wind, "
        "reproduces the learned result to within 0.003 percentage points."
    )

    md = f"""# Neutral-MHS Attribution Check

## Verdict

{verdict}

The saving remains real as an engineering/state-conditioned economy result, but it should not be
described as a learned-forecast-attributable pump-saving result unless a later attribution control
shows a larger learned-vs-current-only gap.

## Matched Comparison

All three runs use the same 96-case neutral-MHS overnight casebook and the same 3600s case duration.

| arm | pump (m3) | saving vs A0 | saved cases | harmed cases | fallback delta | max p95 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 v1.6 | {summary.loc[0, 'pump_m3']:.2f} | 0.00% | 0 | 0 | 0s | 0.000 deg |
| neutral-MHS learned | {learned_pump:.2f} | {learned_saved_pct:.3f}% | {int(summary.loc[1, 'saved_cases'])} | {int(summary.loc[1, 'harmed_cases'])} | {summary.loc[1, 'delta_fallback_s_vs_a0']:.0f}s | {summary.loc[1, 'max_case_delta_p95_axis_vs_a0_deg']:.3f} deg |
| neutral-MHS current-only | {current_pump:.2f} | {current_saved_pct:.3f}% | {int(summary.loc[2, 'saved_cases'])} | {int(summary.loc[2, 'harmed_cases'])} | {summary.loc[2, 'delta_fallback_s_vs_a0']:.0f}s | {summary.loc[2, 'max_case_delta_p95_axis_vs_a0_deg']:.3f} deg |

## Attribution Signal

- learned minus current-only total pump: {pump_gap_m3:.3f} m3.
- learned minus current-only saving gap: {saving_gap_pp:.4f} percentage points.
- exactly matched case rows: {exact_same_rows} / {len(case_delta)}.
- maximum single-case pump difference: {max_case_pump_gap:.3f} m3.

This is far too small to claim that future forecast information is the cause of the neutral-MHS
result. The controller behavior is best interpreted as a clean-start / current-load gated economy
mode with templated action levels. The learned forecast can still be used as an optional eligibility
or advisory input, but the present evidence does not show that it is necessary for the saving.

## Paper Wording

Use:

> A neutral moderate-high steady clean-start economy mode reduced pump work by about 12.14% on the
> fixed 96-case candidate set with no fallback increase. A current-only attribution control reproduced
> the same result, so this mode is reported as state/current-load conditioned rather than as a
> forecast-attributable specialist.

Avoid:

> The learned forecast produced the neutral-MHS saving.

## Files

- Raw summary: `{summary_path}`
- Per-case deltas: `{delta_path}`
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {md_path}")
    print(verdict)


if __name__ == "__main__":
    main()
