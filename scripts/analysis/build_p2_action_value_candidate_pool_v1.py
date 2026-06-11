#!/usr/bin/env python3
"""Build the next P2 action-value candidate pool from existing evidence.

This is a read-only selection step. It does not run a controller and does not
change control logic. The goal is to separate the old low-gain continuous P2
result from the stronger D1/P2-like action-value opportunity evidence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs" / "wind_prediction" / "regime_conditioned_policy_development_v1"
D1 = ROOT / "dc_preserving_deadband_v1"
OUT = REPO / "outputs" / "wind_prediction" / "p2_action_value_candidate_pool_v1"
SAVING_12H_THRESHOLD_PCT = 15.0

TOP16_12H = D1 / "p2like_top16_pid_deadband_1p5_12h" / "casebook_summary.csv"
TOP16_6H = D1 / "p2like_top16_pid_deadband_1p5_6h" / "casebook_summary.csv"
SUMMARY = D1 / "d1_broad_p2_validation_summary_with_all80.csv"
PER_CASE = D1 / "d1_broad_p2_validation_per_case_with_all80.csv"

BOUNDARY = (
    ROOT
    / "clean_neutral_headroom_expansion_v1"
    / "casebooks"
    / "neutral_boundary_control_cases.csv"
)
QUIET = (
    ROOT
    / "regime_mining_v3"
    / "raw_tables"
    / "casebooks_test"
    / "quiet_low_opportunity_cases.csv"
)
REVERSAL = (
    ROOT
    / "regime_mining_v3"
    / "raw_tables"
    / "casebooks_test"
    / "direction_reversal_boundary_cases.csv"
)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.2f}")
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for rec in view.to_dict("records"):
        lines.append("| " + " | ".join(str(rec[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def _casebook_from_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[["case_id", "timestamp", "label"]].copy()
    out["source_file"] = str(path.relative_to(REPO))
    return out


def _load_control_cases(path: Path, role: str, n: int) -> pd.DataFrame:
    df = pd.read_csv(path).head(n).copy()
    out = df[["case_id", "timestamp", "label"]].copy()
    out["role"] = role
    out["source_file"] = str(path.relative_to(REPO))
    return out


def build_main_pool() -> pd.DataFrame:
    cases = _casebook_from_summary(TOP16_12H)
    per_case = pd.read_csv(PER_CASE)
    p2_12h = per_case[per_case["run"].eq("P2like top16 D1 12h")].copy()
    p2_6h = per_case[per_case["run"].eq("P2like top16 D1 6h")].copy()
    p2_12h = p2_12h.rename(
        columns={
            "case_saving_pct": "saving_pct_12h",
            "closed_pump_work_m3": "closed_pump_m3_12h",
            "primary_pump_work_m3": "primary_pump_m3_12h",
            "primary_safety_fallback_ratio": "fallback_ratio_12h",
        }
    )
    p2_6h = p2_6h.rename(
        columns={
            "case_saving_pct": "saving_pct_6h",
            "closed_pump_work_m3": "closed_pump_m3_6h",
            "primary_pump_work_m3": "primary_pump_m3_6h",
            "primary_safety_fallback_ratio": "fallback_ratio_6h",
        }
    )
    metrics = p2_12h[
        [
            "case_id",
            "saving_pct_12h",
            "closed_pump_m3_12h",
            "primary_pump_m3_12h",
            "fallback_ratio_12h",
            "d_pitch_p95",
            "d_roll_p95",
        ]
    ].merge(
        p2_6h[
            [
                "case_id",
                "saving_pct_6h",
                "closed_pump_m3_6h",
                "primary_pump_m3_6h",
                "fallback_ratio_6h",
            ]
        ],
        on="case_id",
        how="left",
    )
    out = cases.merge(metrics, on="case_id", how="left")
    out["role"] = "p2_positive_main"
    out["validation_use"] = "main evidence: >=16-case P2 action-value saving pool"
    return out


def build_guard_pool() -> pd.DataFrame:
    frames = [
        _load_control_cases(BOUNDARY, "p2_boundary_neutral_shift", 3),
        _load_control_cases(QUIET, "p2_ordinary_low_opportunity", 3),
        _load_control_cases(REVERSAL, "p2_hard_direction_reversal_negative", 2),
    ]
    guard = pd.concat(frames, ignore_index=True)
    guard["validation_use"] = "guardrail: should abstain or remain close to baseline"
    return guard


def write_readout(
    main_pool: pd.DataFrame,
    strict_main_pool: pd.DataFrame,
    below_threshold_pool: pd.DataFrame,
    guard_pool: pd.DataFrame,
) -> None:
    summary = pd.read_csv(SUMMARY)
    summary_view = summary[
        summary["run"].isin(
            [
                "P2like top16 D1 6h",
                "P2like top16 D1 12h",
                "P2like all80 D1 1h smoke",
            ]
        )
    ].copy()
    lines = [
        "# P2 Action-Value Candidate Pool v1",
        "",
        "## Purpose",
        "",
        "The old continuous low-gain P2 validation is safe but pump-neutral. This pool switches P2 work to the stronger D1/P2-like action-value evidence while keeping explicit boundary and ordinary controls.",
        "",
        f"Primary headline selection uses the user's 12h requirement directly: cases must have saving_pct_12h > {SAVING_12H_THRESHOLD_PCT:.1f}%. The full 16-case source pool is retained for audit, but the 12h-below-threshold case is not counted in the headline candidate set.",
        "",
        "## Existing Evidence",
        "",
        _md_table(
            summary_view,
            [
                "run",
                "cases",
                "saving_pct",
                "win_cases",
                "loss_cases",
                "fallback_max",
                "max_d_pitch_p95_deg",
                "max_d_roll_p95_deg",
                "latch_delta_pct",
            ],
        ),
        "",
        "Reading: the 16-case P2like D1 pool clears the user's 15% target in aggregate at both 6h and 12h. For the stricter per-case 12h gate, 15 of the 16 source positives pass.",
        "",
        "## Strict 12h Headline P2 Pool",
        "",
        _md_table(
            strict_main_pool,
            [
                "case_id",
                "saving_pct_6h",
                "saving_pct_12h",
                "fallback_ratio_12h",
                "d_pitch_p95",
                "d_roll_p95",
            ],
        ),
        "",
        "## 12h Below-Threshold Diagnostic",
        "",
        _md_table(
            below_threshold_pool,
            [
                "case_id",
                "saving_pct_6h",
                "saving_pct_12h",
                "fallback_ratio_12h",
                "d_pitch_p95",
                "d_roll_p95",
            ],
        ),
        "",
        "## Guardrail Pool",
        "",
        _md_table(guard_pool, ["case_id", "role", "timestamp"]),
        "",
        "## Recommendation",
        "",
        "- Use the 15-case strict 12h>15% P2like D1 pool as the next P2 headline evidence set, not the old 6-case low-gain continuous set.",
        "- Keep the full 16-case source pool as audit context because its aggregate 12h saving still clears the 15% target.",
        "- Keep the 8-case guardrail pool in the same validation package so the claim is not just cherry-picked positives.",
        "- For optimization, preserve the D1/deadband-style pump-saving behavior and add a deployable gate that only opens on P2like high baseline-opportunity cases.",
        "- Do not reuse the target-owning relaxed P2 profile as the main candidate; its 24h result is a rejected lifecycle-debt path.",
    ]
    (OUT / "p2_action_value_candidate_readout.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_pool = build_main_pool()
    strict_main_pool = main_pool[
        main_pool["saving_pct_12h"].gt(SAVING_12H_THRESHOLD_PCT)
    ].copy()
    below_threshold_pool = main_pool[
        ~main_pool["saving_pct_12h"].gt(SAVING_12H_THRESHOLD_PCT)
    ].copy()
    guard_pool = build_guard_pool()

    strict_candidate = pd.concat(
        [
            strict_main_pool[["case_id", "timestamp", "label"]],
            guard_pool[["case_id", "timestamp", "label"]],
        ],
        ignore_index=True,
    )
    strict_manifest = pd.concat(
        [strict_main_pool, guard_pool], ignore_index=True, sort=False
    )
    full_candidate = pd.concat(
        [
            main_pool[["case_id", "timestamp", "label"]],
            guard_pool[["case_id", "timestamp", "label"]],
        ],
        ignore_index=True,
    )
    full_manifest = pd.concat([main_pool, guard_pool], ignore_index=True, sort=False)

    _save_table(main_pool, OUT / "p2_main_positive_16case_pool.csv")
    _save_table(strict_main_pool, OUT / "p2_main_positive_12h_gt15_15case_pool.csv")
    _save_table(
        below_threshold_pool,
        OUT / "p2_main_positive_12h_below15_diagnostics.csv",
    )
    _save_table(guard_pool, OUT / "p2_guardrail_8case_pool.csv")
    _save_table(strict_manifest, OUT / "p2_action_value_candidate_manifest.csv")
    _save_table(
        strict_manifest,
        OUT / "p2_action_value_candidate_manifest_12h_gt15.csv",
    )
    _save_table(strict_candidate, OUT / "p2_action_value_23case_12h_gt15_cases.csv")
    _save_table(full_manifest, OUT / "p2_action_value_full_24case_audit_manifest.csv")
    _save_table(full_candidate, OUT / "p2_action_value_full_24case_cases.csv")
    write_readout(main_pool, strict_main_pool, below_threshold_pool, guard_pool)

    print(OUT / "p2_action_value_23case_12h_gt15_cases.csv")
    print(OUT / "p2_action_value_candidate_readout.md")


if __name__ == "__main__":
    main()
