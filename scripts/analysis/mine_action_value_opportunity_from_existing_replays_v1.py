#!/usr/bin/env python3
"""Mine action-value opportunity from existing replay outputs.

This is a read-only consolidation pass.  It does not run a controller and does
not tune thresholds from fresh outcomes.  It answers the bounded question:

Given the existing 96-case replay labels and the already-run 4hao edge scans,
where is the deployable action-value opportunity after the P2 continuous
low-gain result proved pump-neutral?
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT = Path("outputs/wind_prediction/action_value_opportunity_existing_replays_v1")
PSC = Path("outputs/wind_prediction/psc_selector_mixed_pool_v1")
LABELS = PSC / "replay_selector_labels_96case_pair.csv"
ALL_VARIANT = PSC / "psc_18pct_saving_20260602/all_variant_96case_summary.csv"
EDGE13 = PSC / "psc_4hao_25pct_overnight_v1/edge13_new_unopened_case_metrics.csv"

BASELINE_SAVED_M3 = 5999.149045747045
DENOMINATOR_M3 = 45880.13269320306


def _regime(label: str) -> str:
    if "mixed_regime=" not in label:
        return "unknown"
    return label.split("mixed_regime=", 1)[1].split("|", 1)[0].strip()


def _fmt_case_list(values: pd.Series) -> str:
    return ";".join(str(v) for v in values.tolist())


def build_replay_label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    total_safety = float(labels["safety_pump_m3"].sum())
    rows = []
    for key, desc in [
        ("rawenv_safe", "all rawenv-safe labels"),
        ("use_rawenv", "rawenv-safe and pump-saving replay labels"),
        ("rawenv_pump_saves", "pump-saving regardless of safety"),
    ]:
        selected = labels[labels[key].astype(bool)].copy()
        selected_pump = float(
            (labels["economy_pump_m3"].where(labels[key].astype(bool), labels["safety_pump_m3"])).sum()
        )
        rows.append(
            {
                "label_key": key,
                "description": desc,
                "selected_cases": int(len(selected)),
                "projected_saved_m3": total_safety - selected_pump,
                "projected_saving_pct": 100.0 * (total_safety - selected_pump) / total_safety,
                "d_fallback_s": float(selected["d_fallback_s"].sum()),
                "d_time_gt5_s": float(selected["d_time_gt5_s"].sum()),
                "case_ids": _fmt_case_list(selected["case_id"]),
            }
        )
    return pd.DataFrame(rows)


def build_regime_summary(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["mixed_regime"] = labels["label"].map(_regime)
    rows = []
    for regime, group in labels.groupby("mixed_regime", sort=True):
        selected = group[group["use_rawenv"].astype(bool)]
        safety = float(group["safety_pump_m3"].sum())
        projected = float(
            (group["economy_pump_m3"].where(group["use_rawenv"].astype(bool), group["safety_pump_m3"])).sum()
        )
        rows.append(
            {
                "mixed_regime": regime,
                "cases": int(len(group)),
                "use_rawenv_cases": int(len(selected)),
                "projected_saved_m3": safety - projected,
                "projected_saving_pct": 100.0 * (safety - projected) / safety if safety > 0 else 0.0,
                "d_fallback_s": float(selected["d_fallback_s"].sum()),
                "d_time_gt5_s": float(selected["d_time_gt5_s"].sum()),
                "case_ids": _fmt_case_list(selected["case_id"]),
            }
        )
    return pd.DataFrame(rows)


def _edge_policy(edge: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "strict":
        mask = (
            edge["safety_fail"].eq(0)
            & edge["d_fallback_s"].le(0)
            & edge["d_time_gt5_s"].le(0)
        )
    elif name == "light":
        mask = (
            edge["safety_fail"].eq(0)
            & edge["d_fallback_s"].le(0)
            & edge["d_time_gt5_s"].le(10)
            & edge["p95_axis_deg"].le(4.3)
        )
    elif name == "balanced":
        mask = (
            edge["safety_fail"].eq(0)
            & edge["d_fallback_s"].le(0)
            & edge["d_time_gt5_s"].le(35)
            & edge["p95_axis_deg"].le(4.7)
        )
    else:
        raise ValueError(name)
    return edge[mask].copy()


def build_edge_policy_summary(edge: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in ["strict", "light", "balanced"]:
        selected = _edge_policy(edge, policy)
        add = float(selected["saved_m3"].sum())
        rows.append(
            {
                "policy": policy,
                "accepted_cases": int(len(selected)),
                "additional_saved_m3": add,
                "projected_total_saved_m3": BASELINE_SAVED_M3 + add,
                "projected_total_saving_pct": 100.0 * (BASELINE_SAVED_M3 + add) / DENOMINATOR_M3,
                "d_fallback_s": float(selected["d_fallback_s"].sum()),
                "d_time_gt4_s": float(selected["d_time_gt4_s"].sum()),
                "d_time_gt5_s": float(selected["d_time_gt5_s"].sum()),
                "max_p95_axis_deg": float(selected["p95_axis_deg"].max()) if len(selected) else 0.0,
                "case_ids": _fmt_case_list(selected["case_id"]),
            }
        )
    return pd.DataFrame(rows)


def build_next_casebook(edge: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "09_dual_relief_09",
        "10_dual_relief_10",
        "18_dual_relief_18",
        "94_dual_boundary_06",
        "90_dual_boundary_02",
        "93_dual_boundary_05",
    ]
    return edge[edge["case_id"].isin(wanted)].copy().sort_values(
        ["safety_fail", "d_fallback_s", "d_time_gt5_s", "case_id"]
    )


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.2f}")
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for row in view.to_dict("records"):
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def write_readout(
    replay_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    all_variant: pd.DataFrame,
    edge_summary: pd.DataFrame,
    next_casebook: pd.DataFrame,
) -> None:
    lines = [
        "# Action-Value Opportunity from Existing Replays v1",
        "",
        "## Purpose",
        "",
        "After the continuous low-gain P2 readout showed only about 1-2% pump-volume saving, this pass moves to the broader PSC/action-value evidence already present in the repo. It is read-only: no controller rerun, no new threshold tuning.",
        "",
        "## Inputs",
        "",
        f"- 96-case replay labels: `{LABELS}`",
        f"- 96-case all-variant scan: `{ALL_VARIANT}`",
        f"- 4hao edge13 closed-loop metrics: `{EDGE13}`",
        "",
        "## 96-Case Replay-Label Reading",
        "",
        markdown_table(
            replay_summary,
            [
                "label_key",
                "selected_cases",
                "projected_saved_m3",
                "projected_saving_pct",
                "d_fallback_s",
                "d_time_gt5_s",
            ],
        ),
        "",
        "The replay-label oracle says a deployable safe/action-value selector can clear 10% in principle: `use_rawenv` projects 5960.9 m3, or 12.99%, with zero fallback increase and lower total time>5. This is not a controller claim by itself; it is the action-value target the runtime selector is trying to approximate.",
        "",
        "## Regime Location",
        "",
        markdown_table(
            regime_summary,
            [
                "mixed_regime",
                "cases",
                "use_rawenv_cases",
                "projected_saved_m3",
                "projected_saving_pct",
                "d_fallback_s",
                "d_time_gt5_s",
            ],
        ),
        "",
        "The safe replay opportunity is concentrated in neutral/headroom and transient relief/decay. Direction-reversal boundary rows look tempting by raw pump saving, but the safe label rejects them.",
        "",
        "## Existing Closed-Loop Variant Scan",
        "",
        markdown_table(
            all_variant,
            [
                "policy",
                "accepted_cases",
                "additional_saved_m3",
                "projected_total_saved_m3",
                "projected_saving_pct",
                "d_fallback_s",
                "d_time_gt5_s",
                "max_p95_axis_deg",
                "case_ids",
            ],
        ),
        "",
        "The already-run 4hao variant scan clears 10% and reaches 13.65% under strict acceptance, or 15.35% under a balanced/Pareto acceptance. It still does not reach 18%, and the balanced row spends posture debt.",
        "",
        "## Edge13 Follow-Up Policies",
        "",
        markdown_table(
            edge_summary,
            [
                "policy",
                "accepted_cases",
                "additional_saved_m3",
                "projected_total_saving_pct",
                "d_fallback_s",
                "d_time_gt4_s",
                "d_time_gt5_s",
                "max_p95_axis_deg",
                "case_ids",
            ],
        ),
        "",
        "Edge13 confirms that new safe closed-loop water is real but modest. Strict adds only case 09. A light screen adds 09/10/18 with small time>5 debt. Adding case 94 gives the largest next jump but introduces visible gt4/gt5 and p95 debt.",
        "",
        "## Next Minimal Casebook",
        "",
        markdown_table(
            next_casebook,
            [
                "case_id",
                "saved_m3",
                "saving_pct",
                "d_fallback_s",
                "d_time_gt4_s",
                "d_time_gt5_s",
                "p95_axis_deg",
                "d_p95_axis_deg",
                "safety_fail",
            ],
        ),
        "",
        "Use this as the next focused optimization set: 09/10/18 are the positive edge cases, 94 is the high-value boundary that must be made safer, and 90/93 are hard boundary negatives that should stay rejected.",
        "",
        "## Decision",
        "",
        "- The broader action-value path already clears the 10% screen in existing evidence; P2 does not.",
        "- The deployable replay-label target is about 13% on the 96-case pair pool.",
        "- Existing closed-loop 4hao evidence supports a strict mainline around 13.6% and a balanced/Pareto point around 15.3%, not 18-25%.",
        "- Next work should focus on a runtime mechanism for case 94-like boundary/relief opportunities while preserving hard rejection of 90/91/93/95-like failures.",
    ]
    (OUT / "action_value_opportunity_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS)
    labels[["rawenv_safe", "rawenv_pump_saves", "use_rawenv"]] = labels[
        ["rawenv_safe", "rawenv_pump_saves", "use_rawenv"]
    ].astype(bool)
    all_variant = pd.read_csv(ALL_VARIANT)
    all_variant = all_variant[all_variant["source_pool"].eq("96case_existing_4hao_variants")].copy()
    edge = pd.read_csv(EDGE13)

    replay_summary = build_replay_label_summary(labels)
    regime_summary = build_regime_summary(labels)
    edge_summary = build_edge_policy_summary(edge)
    next_casebook = build_next_casebook(edge)

    replay_summary.to_csv(OUT / "replay_label_policy_summary.csv", index=False)
    regime_summary.to_csv(OUT / "replay_label_regime_summary.csv", index=False)
    all_variant.to_csv(OUT / "existing_4hao_variant_summary.csv", index=False)
    edge_summary.to_csv(OUT / "edge13_policy_summary.csv", index=False)
    next_casebook.to_csv(OUT / "next_minimal_action_value_casebook.csv", index=False)
    write_readout(replay_summary, regime_summary, all_variant, edge_summary, next_casebook)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
