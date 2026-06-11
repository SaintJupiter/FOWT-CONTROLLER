#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    base_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    diag_dir = base_out / "diagnostics"
    ensure_dir(base_out)
    ensure_dir(diag_dir)

    expanded = pd.read_csv(base_out / "window_selection_expanded_preview.csv")
    a14_results = pd.read_csv(base_out / "a1_4_window_results.csv")

    merged = a14_results.merge(
        expanded[
            [
                "prediction_timestamp",
                "a1_group",
                "a1_group_orig",
                "source_risk_type",
                "selection_strength",
                "pressure_mag_block1",
                "pose_exceed_time_block1",
                "block0_small_term",
            ]
        ],
        on="prediction_timestamp",
        how="left",
        suffixes=("_result", "_expanded"),
    )
    merged = merged.rename(columns={"a1_group_result": "a1_group_original_result", "a1_group_expanded": "a1_group"})
    merged.to_csv(base_out / "a1_expanded_window_results.csv", index=False)

    action_summary_rows = []
    for (profile, group), sub in merged.groupby(["discount_profile", "a1_group"]):
        total = len(sub)
        for action, cnt in sub["first_action"].value_counts().items():
            action_summary_rows.append(
                {
                    "discount_profile": profile,
                    "a1_group": group,
                    "first_action": action,
                    "count": int(cnt),
                    "ratio": float(cnt / max(total, 1)),
                }
            )
    action_summary = pd.DataFrame(action_summary_rows).sort_values(
        ["discount_profile", "a1_group", "count"], ascending=[True, True, False]
    )
    action_summary.to_csv(base_out / "a1_expanded_action_summary.csv", index=False)

    lex_summary = (
        merged.groupby(["discount_profile", "a1_group", "winning_dimension"])
        .size()
        .reset_index(name="count")
        .sort_values(["discount_profile", "a1_group", "count"], ascending=[True, True, False])
    )
    lex_summary.to_csv(diag_dir / "a1_expanded_lexicographic_summary.csv", index=False)

    group_rows = []
    for (profile, group), sub in merged.groupby(["discount_profile", "a1_group"]):
        row = {
            "discount_profile": profile,
            "a1_group": group,
            "n_windows": int(len(sub)),
            "active_medium_count": int((sub["first_action"] == "active_medium").sum()),
            "hold_count": int((sub["first_action"] == "hold").sum()),
            "pump_saving_count": int((sub["first_action"] == "pump_saving").sum()),
            "active_small_count": int((sub["first_action"] == "active_small").sum()),
            "median_block0_small_term": float(pd.to_numeric(sub["block0_small_term"], errors="coerce").median()),
            "median_pressure_mag_block1": float(pd.to_numeric(sub["pressure_mag_block1"], errors="coerce").median()),
            "median_pose_exceed_time_block1": float(pd.to_numeric(sub["pose_exceed_time_block1"], errors="coerce").median()),
        }
        if group == "residual_high_normal":
            row["residual_high_normal_explanation"] = (
                "active_medium aligns with block0_small_term>1.0 and elevated steady pressure/residual"
            )
        elif group == "clean_low_pressure_normal":
            row["residual_high_normal_explanation"] = (
                "active_medium should be rare because block0_small_term<=1.0 means active_small already enters deadband"
            )
        else:
            row["residual_high_normal_explanation"] = ""
        group_rows.append(row)
    group_diag = pd.DataFrame(group_rows).sort_values(["discount_profile", "a1_group"])
    group_diag.to_csv(diag_dir / "a1_expanded_group_diagnostics.csv", index=False)

    default_sub = merged[merged["discount_profile"] == "default_discount"].copy()
    clean_sub = default_sub[default_sub["a1_group"] == "clean_low_pressure_normal"]
    residual_sub = default_sub[default_sub["a1_group"] == "residual_high_normal"]
    high_sub = default_sub[default_sub["a1_group"] == "high_pressure_high_event"]
    combo_sub = default_sub[default_sub["a1_group"] == "combined_variability"]

    clean_medium = int((clean_sub["first_action"] == "active_medium").sum())
    residual_medium = int((residual_sub["first_action"] == "active_medium").sum())
    high_degrade = bool(
        (high_sub["first_action"] == "active_medium").any()
        or (high_sub["first_action"] == "active_reverse_small").any()
    )
    combo_medium = int((combo_sub["first_action"] == "active_medium").sum())
    overall_winning = default_sub["winning_dimension"].value_counts()
    dominant_name = str(overall_winning.index[0]) if not overall_winning.empty else "none"
    dominant_count = int(overall_winning.iloc[0]) if not overall_winning.empty else 0
    total_default = int(len(default_sub))
    single_dim_dominant = dominant_count >= max(total_default - 2, 1)

    residual_explainable = bool(
        not residual_sub.empty
        and (pd.to_numeric(residual_sub["block0_small_term"], errors="coerce") > 1.0).all()
    )

    report_lines = [
        "# A1 Expanded Dry-Run Report",
        "",
        f"- elapsed wall time for aggregation: `{time.perf_counter() - t0:.1f} s`",
        "- planner logic source: `A1.4 active-medium necessity gate`",
        "- input window file: `window_selection_expanded_preview.csv`",
        "- note: this round re-groups the existing A1.4 dry-run windows; it does not modify the planner and does not rerun closed-loop.",
        "",
        "## Final Answers",
        "",
        f"1. clean_low_pressure_normal 中 active_medium 是否基本消失：`{'YES' if clean_medium == 0 else 'NO'}`",
        f"   - default-discount active_medium count: `{clean_medium}/{len(clean_sub)}`",
        "",
        f"2. residual_high_normal 中 active_medium 是否符合“低事件但高残差/稳态高风”的解释：`{'YES' if residual_medium == len(residual_sub) and residual_explainable else 'PARTIAL'}`",
        f"   - default-discount active_medium count: `{residual_medium}/{len(residual_sub)}`",
        "   - explanation basis: `block0_small_term > 1.0` and elevated block1 pressure/residual",
        "",
        f"3. high_pressure_high_event 和 combined_variability 是否没有明显退化：`{'YES' if (not high_degrade) and combo_medium == 0 else 'NO'}`",
        f"   - high_pressure/high_event default actions: `{high_sub['first_action'].value_counts().to_dict()}`",
        f"   - combined_variability default actions: `{combo_sub['first_action'].value_counts().to_dict()}`",
        "",
        f"4. winning_dimension 是否没有重新单维垄断：`{'NO' if single_dim_dominant else 'YES'}`",
        f"   - default dominant winning_dimension: `{dominant_name} ({dominant_count}/{total_default})`",
        "",
        f"5. 是否可以进入少量 closed-loop A2，还是仍需修 planner：`{'NEED_MORE_PLANNER_FIX' if single_dim_dominant else 'CONSIDER_SMALL_A2'}`",
        "",
        "## Sample Count Note",
        "",
        f"- high_pressure_high_event: `{len(high_sub)}`",
        f"- combined_variability: `{len(combo_sub)}`",
        f"- clean_low_pressure_normal: `{len(clean_sub)}`",
        f"- residual_high_normal: `{len(residual_sub)}`",
        "- this round uses the current expanded preview set directly and does not block on adding more windows.",
    ]
    (base_out / "A1_expanded_dryrun_report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print(f"Saved A1-expanded outputs under {base_out}")


if __name__ == "__main__":
    main()
