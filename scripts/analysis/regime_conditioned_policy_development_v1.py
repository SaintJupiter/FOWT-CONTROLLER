#!/usr/bin/env python3
"""Build a regime taxonomy and strategy matrix from existing pump-saving runs.

This script is intentionally read-only with respect to controller behavior.  It
collates already generated A0/budget100/smart-release/regime-auto evidence,
assigns deployable regime labels from forecast/state features, and writes the
next strategy matrix for targeted follow-up tests.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path("outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1")
OUT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
RAW = OUT / "raw_tables"
PAPER = OUT / "paper_ready"
DIAG = OUT / "diagnostics"


@dataclass(frozen=True)
class EvidenceSource:
    name: str
    path: Path
    description: str


SOURCES = [
    EvidenceSource(
        "locked53_budget100_mechanism",
        ROOT / "budget100_final_candidate_v1/raw_tables/budget100_regime_mechanism_case_table.csv",
        "Per-case forecast shape and budget100 deltas on the locked53 mechanism table.",
    ),
    EvidenceSource(
        "locked53_opportunity_delta",
        ROOT / "budget100_final_candidate_v1/raw_tables/locked53_budget100_opportunity_case_deltas.csv",
        "Fixed-rule opportunity subset deltas under budget100.",
    ),
    EvidenceSource(
        "new24_budget100_delta",
        ROOT / "relief_decay_expansion_v1/raw_tables/relief_decay_expansion_case_delta.csv",
        "Natural relief/decay expansion deltas under budget100.",
    ),
    EvidenceSource(
        "new24_smart_release_delta",
        ROOT / "relief_decay_expansion_v1/smart_release_compare_v1/raw_tables/smart_release_per_case_delta_vs_a0.csv",
        "New24 smart-release specialist deltas versus A0.",
    ),
    EvidenceSource(
        "regime_auto_delta",
        ROOT / "regime_auto_v1/raw_tables/regime_auto_case_deltas.csv",
        "First-pass multi-regime selector case deltas.",
    ),
    EvidenceSource(
        "regime_auto_reason_counts",
        ROOT / "regime_auto_v1/raw_tables/regime_auto_reason_counts.csv",
        "First-pass multi-regime selector reason counts.",
    ),
]


def _boolish(row: pd.Series, col: str) -> bool:
    value = row.get(col, False)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _num(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_regime(row: pd.Series) -> tuple[str, str, str]:
    """Return (regime, maturity, recommended_strategy)."""

    case = str(row.get("case", "")).lower()
    label = str(row.get("label", "")).lower()
    max_peak = _num(row, "max_near_peak")
    near_drop = _num(row, "max_near_drop")
    far_drop = _num(row, "max_far_relief_drop")
    a0_pump = _num(row, "pump_m3_a0", _num(row, "a0_pump_m3"))
    p95 = _num(row, "p95_max_axis_deg_a0")
    time_gt5 = _num(row, "time_gt5_s_a0")
    saving_pct = _num(row, "pump_saving_pct")

    reversal = (
        _boolish(row, "direction_reversal_candidate")
        or _num(row, "reversal_or_signflip_rows") > 0
        or "signflip" in case
        or "signflip" in label
        or "_sf_" in case
    )
    reintensify = _num(row, "reintensification_rows") > 0 or _num(row, "h120_far_reintensify_rows") > 0
    relief_decay = (
        _boolish(row, "forecast_relief_decay_candidate")
        or _boolish(row, "fast_callback_candidate")
        or _num(row, "high_then_relief_rows") > 0
        or _num(row, "rise_then_fall_rows") > 0
        or ("relief_decay" in case and near_drop >= 0.15)
    )
    h120_far_relief_only = (
        _boolish(row, "h120_supervisory_relief_candidate")
        or _num(row, "h120_far_relief_rows") > 0
        or far_drop >= 0.25
    ) and not relief_decay
    residual_high = (
        max_peak >= 1.05
        and near_drop < 0.15
        and not relief_decay
        and not reversal
        and not reintensify
        and a0_pump >= 100.0
    )
    lowrisk_redundant = (
        _boolish(row, "is_lowrisk_label")
        or "lowrisk" in case
        or (max_peak < 0.8 and a0_pump >= 50.0)
    )
    hard_floor_dominated = (time_gt5 >= 600.0 or p95 >= 5.0) and saving_pct < 10.0

    if reversal or reintensify:
        return (
            "boundary_reversal_or_reintensification",
            "veto_boundary",
            "Do not enable aggressive pump-saving; use v1.6 hard floor and release/veto economy holds.",
        )
    if relief_decay:
        return (
            "transient_peak_future_decay",
            "mature_positive",
            "Use relief_decay_smart_release_v1 when runtime 0-60min forecast shows peak followed by decay.",
        )
    if residual_high:
        return (
            "residual_high_nonintensifying",
            "candidate_needs_isolation",
            "Candidate: residual-high economy hold with early release; isolate before any mixed deployment.",
        )
    if lowrisk_redundant:
        return (
            "lowrisk_redundant_pump",
            "candidate_sparse",
            "Candidate: very conservative redundant-pump suppression; evidence is sparse.",
        )
    if h120_far_relief_only:
        return (
            "far_relief_supervisory_only",
            "advisory_only",
            "Keep as h120 supervisory relief context; do not use for closed-loop pump saving.",
        )
    if hard_floor_dominated:
        return (
            "hard_floor_recovery_dominated",
            "no_economy_action",
            "Use v1.6 hard safety floor; no economy relaxation.",
        )
    return (
        "neutral_or_low_opportunity",
        "no_action",
        "No pump-saving specialist; keep v1.6/A0 behavior.",
    )


def load_main_table() -> pd.DataFrame:
    path = SOURCES[0].path
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    new24_delta_path = ROOT / "relief_decay_expansion_v1/raw_tables/relief_decay_expansion_case_delta.csv"
    if new24_delta_path.exists():
        new24 = pd.read_csv(new24_delta_path)
        new24 = new24.set_index("case")
        for idx, row in df.iterrows():
            case = row.get("case")
            if case not in new24.index:
                continue
            for dst, src in [
                ("pump_m3_a0", "a0_pump_m3"),
                ("pump_m3_budget100", "budget100_pump_m3"),
                ("pump_saved_m3", "pump_saved_m3"),
                ("pump_saving_pct", "pump_saving_pct"),
                ("delta_time_gt5_s", "delta_time_gt5_s"),
                ("delta_idle_gt5_s", "delta_idle_gt5_s"),
                ("delta_fallback_time_s", "delta_fallback_time_s"),
                ("delta_p95_max_axis_deg", "delta_p95_max_axis_deg"),
            ]:
                if src in new24.columns:
                    df.at[idx, dst] = new24.at[case, src]
    labels = df.apply(classify_regime, axis=1, result_type="expand")
    labels.columns = ["auto_regime", "strategy_maturity", "recommended_strategy"]
    df = pd.concat([df, labels], axis=1)

    df["a0_pump_m3"] = df.get("pump_m3_a0", df.get("a0_pump_m3", 0.0))
    df["budget100_pump_m3"] = df.get("pump_m3_budget100", df.get("budget100_pump_m3", 0.0))
    df["budget100_saving_m3"] = df.get("pump_saved_m3", 0.0)
    df["budget100_saving_pct"] = df.get("pump_saving_pct", 0.0)
    return df


def summarize_regimes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, g in df.groupby("auto_regime", dropna=False):
        a0 = g["a0_pump_m3"].sum()
        saved = g["budget100_saving_m3"].sum()
        rows.append(
            {
                "auto_regime": regime,
                "cases": len(g),
                "a0_pump_m3": a0,
                "budget100_saved_m3": saved,
                "budget100_saving_pct": 100.0 * saved / a0 if a0 else 0.0,
                "delta_time_gt5_s": g.get("delta_time_gt5_s", pd.Series(dtype=float)).sum(),
                "delta_idle_gt5_s": g.get("delta_idle_gt5_s", pd.Series(dtype=float)).sum(),
                "delta_fallback_time_s": g.get("delta_fallback_time_s", pd.Series(dtype=float)).sum(),
                "mean_delta_p95_deg": g.get("delta_p95_max_axis_deg", pd.Series(dtype=float)).mean(),
                "top_case_saved_m3": g["budget100_saving_m3"].max(),
                "top_case_share_of_regime_saved": (
                    g["budget100_saving_m3"].max() / saved if saved > 0 else 0.0
                ),
                "strategy_maturity": g["strategy_maturity"].mode().iat[0],
                "recommended_strategy": g["recommended_strategy"].mode().iat[0],
            }
        )
    return pd.DataFrame(rows).sort_values(["budget100_saved_m3", "cases"], ascending=[False, False])


def summarize_by_source_and_regime(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, regime), g in df.groupby(["source_casebook", "auto_regime"], dropna=False):
        a0 = g["a0_pump_m3"].sum()
        saved = g["budget100_saving_m3"].sum()
        rows.append(
            {
                "source_casebook": source,
                "auto_regime": regime,
                "cases": len(g),
                "a0_pump_m3": a0,
                "budget100_saved_m3": saved,
                "budget100_saving_pct": 100.0 * saved / a0 if a0 else 0.0,
                "delta_time_gt5_s": g.get("delta_time_gt5_s", pd.Series(dtype=float)).sum(),
                "delta_idle_gt5_s": g.get("delta_idle_gt5_s", pd.Series(dtype=float)).sum(),
                "delta_fallback_time_s": g.get("delta_fallback_time_s", pd.Series(dtype=float)).sum(),
                "mean_delta_p95_deg": g.get("delta_p95_max_axis_deg", pd.Series(dtype=float)).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["source_casebook", "budget100_saved_m3"], ascending=[True, False]
    )


def build_strategy_matrix() -> pd.DataFrame:
    rows = [
        {
            "regime": "transient_peak_future_decay",
            "runtime_detector": "0-60min learned forecast: high/rising near block followed by clear decay; no re-intensification or direction reversal.",
            "policy": "relief_decay_smart_release_v1",
            "status": "mature_positive",
            "expected_role": "Main paper-positive automatic specialist.",
            "next_action": "Keep fixed; use as representative automatic regime controller.",
        },
        {
            "regime": "residual_high_nonintensifying",
            "runtime_detector": "High current/near pressure without further intensification and without signflip.",
            "policy": "residual-high economy hold with early release",
            "status": "candidate_needs_isolation",
            "expected_role": "Possible supporting specialist, but mixed regime_auto failed.",
            "next_action": "Create fixed subset and test one isolated residual-high policy before any mixed selector.",
        },
        {
            "regime": "lowrisk_redundant_pump",
            "runtime_detector": "Low forecast pressure and baseline still pumps; low posture/floor risk.",
            "policy": "conservative redundant-pump suppression",
            "status": "candidate_sparse",
            "expected_role": "Small supporting result if enough cases exist.",
            "next_action": "Mine more cases first; do not make a headline.",
        },
        {
            "regime": "boundary_reversal_or_reintensification",
            "runtime_detector": "Signflip, direction reversal, far/near re-intensification, fallback-prone state.",
            "policy": "veto/release only",
            "status": "veto_boundary",
            "expected_role": "Negative boundary figure and safety explanation.",
            "next_action": "Keep out of pump-saving mode; use for boundary/catch-up case studies.",
        },
        {
            "regime": "far_relief_supervisory_only",
            "runtime_detector": "60-120min far relief without near 0-60min pump opportunity.",
            "policy": "h120 supervisory advisory",
            "status": "advisory_only",
            "expected_role": "Shows 120min forecast remains in system as context, not direct pump control.",
            "next_action": "Log/display; do not use as closed-loop pump-saving lever.",
        },
        {
            "regime": "hard_floor_recovery_dominated",
            "runtime_detector": "High posture/floor/fallback domain dominates.",
            "policy": "v1.6 hard floor",
            "status": "no_economy_action",
            "expected_role": "Safety backbone; no economy specialist.",
            "next_action": "Do not relax economy if floor/recovery/fallback is active.",
        },
        {
            "regime": "neutral_or_low_opportunity",
            "runtime_detector": "No strong pump opportunity, no deployable relief/decay signal.",
            "policy": "v1.6 baseline",
            "status": "no_action",
            "expected_role": "Keeps false positives down.",
            "next_action": "Do not pursue pump saving here.",
        },
    ]
    return pd.DataFrame(rows)


def write_casebooks(df: pd.DataFrame) -> None:
    casebook_dir = RAW / "casebooks"
    casebook_dir.mkdir(parents=True, exist_ok=True)
    cols = ["case_id", "timestamp", "label"]
    for regime, g in df.groupby("auto_regime"):
        if not set(cols).issubset(g.columns):
            continue
        out = g[cols].copy()
        out["label"] = out["label"].astype(str) + f" | auto_regime={regime}"
        out.to_csv(casebook_dir / f"{regime}_cases.csv", index=False)


def write_markdown(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    strategy: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Regime-Conditioned Policy Development v1\n")
    lines.append("This is a read-only taxonomy pass over existing pump-saving runs. It separates physical/forecast regimes from controller strategies so that later tests can be targeted instead of becoming broad threshold tuning.\n")
    lines.append("## Main Findings\n")
    lines.append("- The mature positive regime is `transient_peak_future_decay`: forecast shows a near-term peak that decays within 0-60min, so the controller should not chase the temporary peak. The current specialist is `relief_decay_smart_release_v1`.\n")
    lines.append("- Residual-high and low-risk redundant-pump regimes exist, but current mixed `regime_auto_pump_saving_v1` did not generalize on guard10/broader20. They need isolated validation, not mixed deployment.\n")
    lines.append("- Signflip/reversal/re-intensification cases are not favorable regimes; they are boundary/veto cases.\n")
    lines.append("- 60-120min h120 remains supervisory context. The closed-loop pump-saving specialist should be driven by 0-60min opportunity signals.\n")
    lines.append("## Regime Summary\n")
    lines.append(_markdown_table(summary, float_digits=3))
    lines.append("\n## Source Casebook x Regime Summary\n")
    lines.append(_markdown_table(source_summary, float_digits=3))
    lines.append("\n## Strategy Matrix\n")
    lines.append(_markdown_table(strategy, float_digits=3))
    lines.append("\n## Recommended Execution Order\n")
    lines.append("1. Lock `relief_decay_smart_release_v1` as the current mature specialist for transient-peak/future-decay cases.\n")
    lines.append("2. Run only one isolated validation for `residual_high_nonintensifying`; do not mix it into the selector until it beats baseline on its own subset.\n")
    lines.append("3. Mine additional `lowrisk_redundant_pump` cases before attempting a controller change, because current evidence is sparse.\n")
    lines.append("4. Keep reversal/re-intensification cases as veto/boundary examples, not optimization targets.\n")
    lines.append("5. Only after a candidate specialist passes its own subset should it be added to a multi-regime runtime selector.\n")
    (PAPER / "regime_taxonomy_and_strategy_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_sources() -> None:
    rows = []
    for source in SOURCES:
        rows.append(
            {
                "name": source.name,
                "path": str(source.path),
                "exists": source.path.exists(),
                "description": source.description,
            }
        )
    pd.DataFrame(rows).to_csv(DIAG / "evidence_sources.csv", index=False)


def _markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    render = df.copy()
    for col in render.columns:
        if pd.api.types.is_float_dtype(render[col]):
            render[col] = render[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
        else:
            render[col] = render[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(render.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in render.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    for d in (RAW, PAPER, DIAG):
        d.mkdir(parents=True, exist_ok=True)

    write_sources()
    df = load_main_table()
    summary = summarize_regimes(df)
    source_summary = summarize_by_source_and_regime(df)
    strategy = build_strategy_matrix()

    case_cols = [
        "case",
        "case_id",
        "timestamp",
        "label",
        "auto_regime",
        "strategy_maturity",
        "recommended_strategy",
        "a0_pump_m3",
        "budget100_pump_m3",
        "budget100_saving_m3",
        "budget100_saving_pct",
        "delta_time_gt5_s",
        "delta_idle_gt5_s",
        "delta_fallback_time_s",
        "delta_p95_max_axis_deg",
        "max_near_peak",
        "max_near_drop",
        "max_far_relief_drop",
        "forecast_relief_decay_candidate",
        "fast_callback_candidate",
        "h120_supervisory_relief_candidate",
        "direction_reversal_candidate",
        "reversal_or_signflip_rows",
        "near_high_rows",
        "high_then_relief_rows",
        "rise_then_fall_rows",
    ]
    present_cols = [c for c in case_cols if c in df.columns]
    df[present_cols].to_csv(RAW / "regime_taxonomy_case_table.csv", index=False)
    summary.to_csv(RAW / "regime_taxonomy_summary.csv", index=False)
    source_summary.to_csv(RAW / "regime_source_casebook_summary.csv", index=False)
    strategy.to_csv(RAW / "regime_strategy_matrix.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_casebooks(df)
    write_markdown(df, summary, source_summary, strategy)
    print(f"Wrote {RAW / 'regime_taxonomy_case_table.csv'}")
    print(f"Wrote {RAW / 'regime_taxonomy_summary.csv'}")
    print(f"Wrote {PAPER / 'regime_taxonomy_and_strategy_summary.md'}")


if __name__ == "__main__":
    main()
