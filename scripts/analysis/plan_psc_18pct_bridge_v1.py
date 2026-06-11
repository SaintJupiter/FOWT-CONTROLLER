#!/usr/bin/env python3
"""Plan the shortest evidence path from current PSC savings to 18%.

This script is deliberately an analysis/coordination layer.  It does not
change controller behavior and it does not promote outcome-informed cases to a
validated claim.  It consolidates existing 96-case, fresh, and broader 6h
outputs into a gap ledger and a concrete next-run casebook.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
SRC18 = BASE / "psc_18pct_saving_20260602"
OUT = SRC18 / "bridge_plan_v1"
SOURCE_CASEBOOKS = [
    BASE / "casebooks" / "psc_mixed_24_cases.csv",
    BASE / "psc_4hao_25pct_overnight_v1" / "expanded_top20_risky_gain_cases.csv",
    BASE / "psc_4hao_broader_6h_test_only_main_v1" / "casebooks" / "positive_pool_cases.csv",
    BASE / "psc_4hao_broader_6h_test_only_main_v1" / "casebooks" / "negative_pool_cases.csv",
    BASE / "psc_4hao_broader_6h_test_only_main_v1" / "casebooks" / "background_pool_cases.csv",
]

DENOMINATOR_96CASE_M3 = 45880.13269320306
NO3_SAVED_M3 = 5999.1490457470445
TARGET_PCT = 18.0
TARGET_SAVED_M3 = DENOMINATOR_96CASE_M3 * TARGET_PCT / 100.0


def _extract(label: object, key: str) -> str:
    marker = f"{key}="
    text = str(label)
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].split("|", 1)[0].strip()


def _source_case_id(row: pd.Series) -> str:
    label = str(row.get("label", ""))
    source_case = _extract(label, "source_case_id")
    if source_case:
        return source_case
    case_id = str(row.get("case_id", ""))
    # fresh pools keep the original mixed24 ids; broader pools need a label
    # source id for traceability.
    return re.sub(r"^\d+_", "", case_id)


def _case_aliases(case_id: object, label: object = "") -> set[str]:
    raw = str(case_id or "").strip()
    aliases = {raw}
    clean = re.sub(r"^\d+_", "", raw)
    aliases.add(clean)
    source_case = _extract(label, "source_case_id")
    if source_case:
        aliases.add(source_case)
        aliases.add(re.sub(r"^\d+_", "", source_case))
    return {item for item in aliases if item}


def _source_casebook_index() -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for path in SOURCE_CASEBOOKS:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for row in df.to_dict("records"):
            timestamp = str(row.get("timestamp", "")).strip()
            label = str(row.get("label", ""))
            if not timestamp:
                continue
            case_id = str(row.get("case_id", ""))
            for alias in _case_aliases(case_id, label):
                index.setdefault(alias, (timestamp, label))
            # Also allow suffix matching of numbered casebooks, e.g.
            # 09_dual_neutral_01 -> dual_neutral_01.
            index.setdefault(re.sub(r"^\d+_", "", case_id), (timestamp, label))
    return index


def _source_timestamp(row: pd.Series, index: dict[str, tuple[str, str]]) -> str:
    for alias in _case_aliases(row.get("case_id", ""), row.get("label", "")):
        if alias in index:
            return index[alias][0]
    return str(row.get("timestamp", "")).strip()


def _load_fresh_broader() -> pd.DataFrame:
    source_index = _source_casebook_index()
    selected = pd.read_csv(SRC18 / "fresh_broader_selected_safe_sources.csv")
    selected = selected[selected["policy"].eq("balanced")].copy()
    selected["source_case_id"] = selected.apply(_source_case_id, axis=1)
    selected["source_timestamp"] = selected.apply(lambda row: _source_timestamp(row, source_index), axis=1)
    selected["is_ex_ante_positive_family"] = selected["stratum"].isin(
        ["neutral_mhs_broader", "transient_peak_future_decay", "lowrisk_stable_redundant_candidate"]
    )
    selected["is_boundary_like"] = selected["stratum"].isin(
        ["direction_reversal_boundary", "reintensification_boundary"]
    )
    selected["comfort_grade"] = "reject"
    selected.loc[
        (selected["d_fallback_s"] <= 0)
        & (selected["d_time_gt5_s"] <= 0)
        & (selected["d_time_gt4_s"] <= 0)
        & (selected["d_p95_axis_deg"] <= 1.25),
        "comfort_grade",
    ] = "main_candidate"
    selected.loc[
        (selected["comfort_grade"].eq("reject"))
        & (selected["d_fallback_s"] <= 0)
        & (selected["d_time_gt5_s"] <= 30)
        & (selected["d_p95_axis_deg"] <= 1.35)
        & (selected["p95_axis_deg"] < 3.75),
        "comfort_grade",
    ] = "debt_candidate"
    return selected


def _scenario_rows(fresh: pd.DataFrame) -> pd.DataFrame:
    summary96 = pd.read_csv(SRC18 / "all_variant_96case_summary.csv")
    strict96 = summary96[summary96["policy"].eq("strict")].iloc[0]
    balanced96 = summary96[summary96["policy"].eq("balanced")].iloc[0]

    neutral_clean = fresh[
        fresh["stratum"].eq("neutral_mhs_broader") & fresh["comfort_grade"].isin(["main_candidate", "debt_candidate"])
    ]
    relief_clean = fresh[
        fresh["stratum"].eq("transient_peak_future_decay") & fresh["comfort_grade"].isin(["main_candidate", "debt_candidate"])
    ]
    lowrisk_clean = fresh[
        fresh["stratum"].eq("lowrisk_stable_redundant_candidate") & fresh["comfort_grade"].isin(["main_candidate", "debt_candidate"])
    ]
    boundary_debt = fresh[
        fresh["is_boundary_like"] & fresh["comfort_grade"].eq("debt_candidate")
    ]

    def add_row(name: str, base_saved: float, add: pd.DataFrame, status: str, note: str) -> dict[str, object]:
        add_m3 = float(add["saved_m3"].sum()) if not add.empty else 0.0
        total = base_saved + add_m3
        return {
            "scenario": name,
            "status": status,
            "base_saved_m3": base_saved,
            "bridge_add_m3": add_m3,
            "projected_saved_m3": total,
            "projected_saving_pct": 100.0 * total / DENOMINATOR_96CASE_M3,
            "gap_to_18pct_m3": TARGET_SAVED_M3 - total,
            "bridge_cases": int(len(add)),
            "bridge_d_time_gt4_s": float(add["d_time_gt4_s"].sum()) if not add.empty else 0.0,
            "bridge_d_time_gt5_s": float(add["d_time_gt5_s"].sum()) if not add.empty else 0.0,
            "bridge_d_fallback_s": float(add["d_fallback_s"].sum()) if not add.empty else 0.0,
            "bridge_max_p95_axis_deg": float(add["p95_axis_deg"].max()) if not add.empty else 0.0,
            "bridge_max_d_p95_axis_deg": float(add["d_p95_axis_deg"].max()) if not add.empty else 0.0,
            "case_ids": ";".join(add["case_id"].astype(str).tolist()) if not add.empty else "",
            "note": note,
        }

    rows = [
        {
            "scenario": "current_no3_strict",
            "status": "current_reference",
            "base_saved_m3": NO3_SAVED_M3,
            "bridge_add_m3": 0.0,
            "projected_saved_m3": NO3_SAVED_M3,
            "projected_saving_pct": 100.0 * NO3_SAVED_M3 / DENOMINATOR_96CASE_M3,
            "gap_to_18pct_m3": TARGET_SAVED_M3 - NO3_SAVED_M3,
            "bridge_cases": 0,
            "bridge_d_time_gt4_s": 0.0,
            "bridge_d_time_gt5_s": 0.0,
            "bridge_d_fallback_s": 0.0,
            "bridge_max_p95_axis_deg": 0.0,
            "bridge_max_d_p95_axis_deg": 0.0,
            "case_ids": "",
            "note": "Current strict No.3 reference.",
        },
        {
            "scenario": "reuse_96case_strict_variants",
            "status": "insufficient",
            "base_saved_m3": NO3_SAVED_M3,
            "bridge_add_m3": float(strict96["additional_saved_m3"]),
            "projected_saved_m3": float(strict96["projected_total_saved_m3"]),
            "projected_saving_pct": float(strict96["projected_saving_pct"]),
            "gap_to_18pct_m3": float(strict96["gap_to_18pct_m3"]),
            "bridge_cases": int(strict96["accepted_cases"]),
            "bridge_d_time_gt4_s": float(strict96["d_time_gt4_s"]),
            "bridge_d_time_gt5_s": float(strict96["d_time_gt5_s"]),
            "bridge_d_fallback_s": float(strict96["d_fallback_s"]),
            "bridge_max_p95_axis_deg": float(strict96["max_p95_axis_deg"]),
            "bridge_max_d_p95_axis_deg": float(strict96["max_d_p95_axis_deg"]),
            "case_ids": str(strict96["case_ids"]),
            "note": "Safe but leaves nearly 2,000 m3 gap.",
        },
        {
            "scenario": "reuse_96case_balanced_variants",
            "status": "insufficient_with_debt",
            "base_saved_m3": NO3_SAVED_M3,
            "bridge_add_m3": float(balanced96["additional_saved_m3"]),
            "projected_saved_m3": float(balanced96["projected_total_saved_m3"]),
            "projected_saving_pct": float(balanced96["projected_saving_pct"]),
            "gap_to_18pct_m3": float(balanced96["gap_to_18pct_m3"]),
            "bridge_cases": int(balanced96["accepted_cases"]),
            "bridge_d_time_gt4_s": float(balanced96["d_time_gt4_s"]),
            "bridge_d_time_gt5_s": float(balanced96["d_time_gt5_s"]),
            "bridge_d_fallback_s": float(balanced96["d_fallback_s"]),
            "bridge_max_p95_axis_deg": float(balanced96["max_p95_axis_deg"]),
            "bridge_max_d_p95_axis_deg": float(balanced96["max_d_p95_axis_deg"]),
            "case_ids": str(balanced96["case_ids"]),
            "note": "Best already-run 96-case scan still misses 18%.",
        },
        add_row(
            "bridge_positive_families_only",
            float(balanced96["projected_total_saved_m3"]),
            pd.concat([neutral_clean, relief_clean, lowrisk_clean], ignore_index=True),
            "plausible_next_validation",
            "Adds ex-ante positive families only; must be replayed on an independent 96-case-like pool.",
        ),
        add_row(
            "bridge_positive_plus_boundary_debt",
            float(balanced96["projected_total_saved_m3"]),
            pd.concat([neutral_clean, relief_clean, lowrisk_clean, boundary_debt], ignore_index=True),
            "pareto_only",
            "Crosses 18% arithmetically but mixes boundary/reintensification debt candidates; not a main claim.",
        ),
    ]
    return pd.DataFrame(rows)


def _write_next_casebook(fresh: pd.DataFrame) -> pd.DataFrame:
    # Use only candidates that can teach the next gate.  Positive families are
    # validation targets; boundary-like debt cases are negative/pareto probes.
    work = fresh[
        fresh["comfort_grade"].isin(["main_candidate", "debt_candidate"])
        & (fresh["saved_m3"] > 0)
    ].copy()
    work["next_role"] = "positive_validation"
    work.loc[work["is_boundary_like"], "next_role"] = "boundary_debt_probe"
    work.loc[work["stratum"].eq("lowrisk_stable_redundant_candidate"), "next_role"] = "lowrisk_smoke"
    work = work.sort_values(
        ["next_role", "stratum", "saved_m3"],
        ascending=[True, True, False],
    )
    missing_ts = work["source_timestamp"].astype(str).str.strip().eq("")
    if missing_ts.any():
        missing = ";".join(work.loc[missing_ts, "case_id"].astype(str).tolist())
        raise RuntimeError(f"missing source timestamps for bridge18 cases: {missing}")

    casebook = work[["case_id", "source_timestamp", "label"]].copy()
    casebook = casebook.rename(columns={"source_timestamp": "timestamp"})
    casebook["case_id"] = [f"bridge18_{i:02d}" for i in range(1, len(casebook) + 1)]
    casebook["label"] = (
        work["label"].astype(str)
        + " | bridge18_source_pool="
        + work["source_pool"].astype(str)
        + " | bridge18_role="
        + work["next_role"].astype(str)
        + " | bridge18_saved_m3="
        + work["saved_m3"].round(3).astype(str)
    )
    casebook.to_csv(OUT / "bridge18_next_validation_cases.csv", index=False)
    work.to_csv(OUT / "bridge18_next_validation_case_metadata.csv", index=False)
    return work


def _write_report(scenarios: pd.DataFrame, fresh: pd.DataFrame, next_cases: pd.DataFrame) -> None:
    by_source = (
        fresh.groupby(["source_pool", "stratum", "comfort_grade"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            saved_m3=("saved_m3", "sum"),
            baseline_pump_m3=("pump_m3_baseline", "sum"),
            d_time_gt4_s=("d_time_gt4_s", "sum"),
            d_time_gt5_s=("d_time_gt5_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            max_p95_axis_deg=("p95_axis_deg", "max"),
            max_d_p95_axis_deg=("d_p95_axis_deg", "max"),
        )
        .reset_index()
        .sort_values("saved_m3", ascending=False)
    )
    by_source.to_csv(OUT / "bridge18_candidate_source_summary.csv", index=False)

    def md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in df[cols].to_dict("records"):
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    if col.endswith("_m3"):
                        vals.append(f"{val:.1f}")
                    elif col.endswith("_s"):
                        vals.append(f"{val:.0f}")
                    elif "pct" in col:
                        vals.append(f"{val:.2f}%")
                    elif col.endswith("_deg"):
                        vals.append(f"{val:.2f}")
                    elif "axis" in col:
                        vals.append(f"{val:.2f}")
                    elif col.endswith("_m3"):
                        vals.append(f"{val:.1f}")
                    else:
                        vals.append(f"{val:.1f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return lines

    lines = [
        "# PSC 18% Bridge Plan v1",
        "",
        "## One-line decision",
        "",
        "The current 96-case evidence cannot reach 18% by selecting among already-run variants; the shortest credible route is a new validation pass that adds clean `neutral_mhs_broader`/relief candidates first and treats boundary/reintensification candidates as Pareto probes or negative samples.",
        "",
        "## Gap ledger",
        "",
    ]
    lines.extend(
        md_table(
            scenarios,
            [
                "scenario",
                "status",
                "projected_saved_m3",
                "projected_saving_pct",
                "gap_to_18pct_m3",
                "bridge_cases",
                "bridge_d_time_gt5_s",
                "bridge_d_fallback_s",
                "bridge_max_p95_axis_deg",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Candidate source ranking",
            "",
        ]
    )
    lines.extend(
        md_table(
            by_source.head(12),
            [
                "source_pool",
                "stratum",
                "comfort_grade",
                "cases",
                "saved_m3",
                "d_time_gt4_s",
                "d_time_gt5_s",
                "d_fallback_s",
                "max_p95_axis_deg",
                "max_d_p95_axis_deg",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Execution plan",
            "",
            "1. Stop treating additional 96-case selector sweeps as the main route; best completed sweep is 15.35% and still misses by about 1218 m3.",
            "2. Run the generated `bridge18_next_validation_cases.csv` with `psc_4hao_dynamic_refresh_v1`, refresh-on, 6h duration, and keep baseline/current-only paired output.",
            "3. Main acceptance rule: positive saving, fallback delta 0, `d_time_gt5_s <= 0`, `d_time_gt4_s <= 0`, and p95-axis below 3.75 deg with d_p95 no larger than 1.25 deg.",
            "4. Boundary/reintensification rows can be used only as debt probes unless an ex-ante release rule reduces their gt4/gt5 debt; do not count them toward the main 18% claim yet.",
            "5. If positive-family validation still misses 18%, implement a real comfort-aware action interface rather than widening thresholds: shorter hold, event-risk release, and p95/time>4 debt budget.",
            "",
            "Generated next-run casebook:",
            "",
            f"`{(OUT / 'bridge18_next_validation_cases.csv').relative_to(REPO)}`",
            "",
            f"Cases in next-run casebook: {len(next_cases)}",
        ]
    )
    (OUT / "bridge18_plan_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fresh = _load_fresh_broader()
    fresh.to_csv(OUT / "bridge18_candidate_case_table.csv", index=False)
    scenarios = _scenario_rows(fresh)
    scenarios.to_csv(OUT / "bridge18_gap_scenarios.csv", index=False)
    next_cases = _write_next_casebook(fresh)
    _write_report(scenarios, fresh, next_cases)
    print(OUT / "bridge18_plan_readout.md")
    print(scenarios.to_string(index=False))


if __name__ == "__main__":
    main()
