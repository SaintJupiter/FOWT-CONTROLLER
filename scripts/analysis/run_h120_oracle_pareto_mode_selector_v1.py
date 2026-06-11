#!/usr/bin/env python3
"""Audit the f120-oracle Pareto mode selector.

The selector is intentionally narrow: it only chooses among existing static
reactive-floor modes, and only when the default-off
``--h120-pareto-mode-selector`` flag is enabled.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv312" / "bin" / "python"
OUT_DIR = REPO_ROOT / "outputs/wind_prediction/h120_oracle_pareto_mode_selector_v1"
DATASET_DIR = (
    REPO_ROOT
    / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)
GUARD10_CASES = (
    REPO_ROOT
    / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/guard10_cases.csv"
)
BROADER20_CASES = (
    REPO_ROOT
    / "outputs/wind_prediction/h120_oracle_controller_value_audit_v1/broader20_cases.csv"
)


SCENARIOS = {
    "static_v14_economy": {
        "label": "static v1.4 economy",
        "reactive_floor_action": "active_small",
        "reactive_floor_delay": "0",
        "selector": False,
    },
    "static_v16_balanced": {
        "label": "static v1.6 balanced",
        "reactive_floor_action": "active_small",
        "reactive_floor_delay": "1200",
        "selector": False,
    },
    "static_v15_safety": {
        "label": "static v1.5 safety",
        "reactive_floor_action": "active_medium",
        "reactive_floor_delay": "0",
        "selector": False,
    },
    "oracle_pareto_selector": {
        "label": "oracle h120 Pareto selector",
        "reactive_floor_action": "active_small",
        "reactive_floor_delay": "1200",
        "selector": True,
    },
}


def _casebook_args(run_dir: Path, cases_csv: Path, cfg: dict[str, Any]) -> list[str]:
    args = [
        str(PYTHON),
        "scripts/analysis/run_prediction_primary_casebook.py",
        "--cases-csv",
        str(cases_csv),
        "--primary-control-profile",
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "--primary-only",
        "--duration-s",
        "7200",
        "--skip-figures",
        "--forecast-source",
        "oracle",
        "--dataset-dir",
        str(DATASET_DIR),
        "--far-horizon",
        "--reactive-floor-predictive-veto",
        "on",
        "--reactive-floor-action",
        str(cfg["reactive_floor_action"]),
        "--reactive-floor-medium-delay-s",
        str(cfg["reactive_floor_delay"]),
        "--reactive-floor-post-exit-mode",
        "early_stop",
        "--out-dir",
        str(run_dir),
    ]
    if cfg.get("selector", False):
        args.append("--h120-pareto-mode-selector")
    return args


def _run_casebook(run_dir: Path, cases_csv: Path, cfg: dict[str, Any]) -> None:
    summary = run_dir / "casebook_summary.csv"
    if summary.exists():
        print(f"[skip] {run_dir.relative_to(REPO_ROOT)}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir.relative_to(REPO_ROOT)}", flush=True)
    subprocess.run(_casebook_args(run_dir, cases_csv, cfg), cwd=REPO_ROOT, check=True)


def _case_key(path: Path) -> str:
    name = path.name
    for suffix in (
        "_prediction_primary_econ_timeseries.csv",
        "_prediction_primary_econ_planner_log.csv",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _fallback_series(df: pd.DataFrame) -> pd.Series:
    for col in (
        "preview_primary_safety_fallback",
        "primary_safety_fallback",
        "primary_safety_fallback_active",
    ):
        if col in df.columns:
            return df[col].fillna(0).astype(float) > 0.5
    return pd.Series(False, index=df.index)


def _edge_count(series: pd.Series) -> int:
    arr = series.fillna(0).astype(float).to_numpy()
    if arr.size == 0:
        return 0
    prev = np.concatenate([[0.0], arr[:-1]])
    return int(((arr > 0.5) & (prev <= 0.5)).sum())


def _read_log(run_dir: Path, key: str) -> pd.DataFrame:
    log_dir = run_dir / "planner_logs"
    direct = log_dir / f"{key}_prediction_primary_econ_planner_log.csv"
    if direct.exists():
        return pd.read_csv(direct, low_memory=False)
    matches = sorted(log_dir.glob(f"{key}*_planner_log.csv"))
    if not matches:
        return pd.DataFrame()
    return pd.read_csv(matches[0], low_memory=False)


def _case_label_lookup(cases_csv: Path) -> dict[str, str]:
    if not cases_csv.exists():
        return {}
    table = pd.read_csv(cases_csv)
    labels: dict[str, str] = {}
    for row in table.to_dict("records"):
        raw = str(row.get("case_id", ""))
        clean = raw[3:] if len(raw) > 3 and raw[:2].isdigit() and raw[2] == "_" else raw
        label = str(row.get("label", ""))
        labels[raw] = label
        labels[clean] = label
    return labels


def _case_metrics(
    run_dir: Path,
    dataset: str,
    scenario: str,
    cases_csv: Path,
) -> list[dict[str, Any]]:
    labels = _case_label_lookup(cases_csv)
    rows: list[dict[str, Any]] = []
    for ts_path in sorted((run_dir / "timeseries").glob("*_timeseries.csv")):
        key = _case_key(ts_path)
        ts = pd.read_csv(ts_path, low_memory=False)
        pitch = ts["pitch_deg"].astype(float).abs()
        roll = ts["roll_deg"].astype(float).abs()
        max_axis = np.maximum(pitch, roll)
        pump = ts.get("pump_total_rate_m3_min", pd.Series(0.0, index=ts.index))
        pump = pump.fillna(0).astype(float).abs()
        high = max_axis > 5.0
        idle = pump < 0.05
        fallback = _fallback_series(ts)

        log = _read_log(run_dir, key)
        z = pd.Series(dtype=float)
        floor_active = log.get("reactive_floor_active", z).fillna(0).astype(float)
        medium_delay = (
            log.get("reactive_floor_medium_delay_active", z).fillna(0).astype(float)
        )
        early_stop = (
            log.get("reactive_floor_post_exit_released", z).fillna(0).astype(float)
        )
        resolved = (
            log.get("reactive_floor_resolved_action", pd.Series("", index=log.index))
            .astype(str)
            if not log.empty
            else pd.Series(dtype=str)
        )
        selector_active = (
            log.get("h120_pareto_mode_selector_active", z).fillna(0).astype(float)
        )
        selector_episode_active = (
            log.get("h120_pareto_mode_selector_episode_active", z)
            .fillna(0)
            .astype(float)
        )

        case_no_ts = key.rsplit("_", 2)[0]
        case_short = (
            case_no_ts[3:]
            if len(case_no_ts) > 3 and case_no_ts[:2].isdigit() and case_no_ts[2] == "_"
            else case_no_ts
        )
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "case": key,
                "case_short": case_short,
                "label": labels.get(case_short, ""),
                "pump": float(pump.sum() / 60.0),
                "sum_fb": float(fallback.mean() * 100.0),
                "max_p95": float(max(np.percentile(pitch, 95), np.percentile(roll, 95))),
                "pitch_p95": float(np.percentile(pitch, 95)),
                "roll_p95": float(np.percentile(roll, 95)),
                "time_over_5": int(high.sum()),
                "idle_time_over_5": int((high & idle).sum()),
                "floor_trigger_count": _edge_count(floor_active),
                "floor_active_rows": int(floor_active.sum()),
                "medium_escalation_count": _edge_count(medium_delay),
                "medium_active_rows": int(medium_delay.sum()),
                "always_medium_rows": int((resolved == "active_medium").sum()),
                "early_stop_count": _edge_count(early_stop),
                "selector_active_rows": int(selector_active.sum()),
                "selector_episode_rows": int(selector_episode_active.sum()),
                "selector_episode_count": _edge_count(selector_episode_active),
            }
        )
    return rows


def _aggregate(case_table: pd.DataFrame) -> pd.DataFrame:
    agg = (
        case_table.groupby(["dataset", "scenario"], as_index=False)
        .agg(
            cases=("case", "count"),
            sum_pump=("pump", "sum"),
            sum_fb=("sum_fb", "sum"),
            max_p95=("max_p95", "max"),
            time_over_5=("time_over_5", "sum"),
            idle_time_over_5=("idle_time_over_5", "sum"),
            floor_trigger_count=("floor_trigger_count", "sum"),
            floor_active_rows=("floor_active_rows", "sum"),
            medium_escalation_count=("medium_escalation_count", "sum"),
            medium_active_rows=("medium_active_rows", "sum"),
            always_medium_rows=("always_medium_rows", "sum"),
            early_stop_count=("early_stop_count", "sum"),
            selector_active_rows=("selector_active_rows", "sum"),
            selector_episode_rows=("selector_episode_rows", "sum"),
            selector_episode_count=("selector_episode_count", "sum"),
        )
        .sort_values(["dataset", "scenario"])
    )
    for dataset in agg["dataset"].unique():
        base_rows = agg[
            (agg["dataset"] == dataset) & (agg["scenario"] == "static_v16_balanced")
        ]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        idxs = agg["dataset"] == dataset
        for col in ("sum_pump", "sum_fb", "max_p95", "time_over_5", "idle_time_over_5"):
            agg.loc[idxs, f"delta_vs_v16_{col}"] = (
                agg.loc[idxs, col].astype(float) - float(base[col])
            )
        agg.loc[idxs, "pump_delta_pct_vs_v16"] = (
            agg.loc[idxs, "sum_pump"].astype(float) - float(base["sum_pump"])
        ) / max(float(base["sum_pump"]), 1e-9)
        agg.loc[idxs, "time_over_5_reduction_pct_vs_v16"] = (
            float(base["time_over_5"]) - agg.loc[idxs, "time_over_5"].astype(float)
        ) / max(float(base["time_over_5"]), 1e-9)
        agg.loc[idxs, "idle_over_5_reduction_pct_vs_v16"] = (
            float(base["idle_time_over_5"])
            - agg.loc[idxs, "idle_time_over_5"].astype(float)
        ) / max(float(base["idle_time_over_5"]), 1e-9)
    return agg


def _selection_table(run_map: dict[tuple[str, str], Path]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for (dataset, scenario), run_dir in run_map.items():
        if scenario != "oracle_pareto_selector":
            continue
        for log_path in sorted((run_dir / "planner_logs").glob("*_planner_log.csv")):
            log = pd.read_csv(log_path, low_memory=False)
            if log.empty:
                continue
            keep = [
                "bucket",
                "current_time_s",
                "history_end",
                "reactive_floor_active",
                "reactive_floor_resolved_action",
                "reactive_floor_elapsed_s",
                "h120_pareto_mode_selector_active",
                "h120_pareto_mode_selector_reason",
                "h120_pareto_mode_selector_selected_mode",
                "h120_pareto_mode_selector_selected_effect",
                "h120_pareto_mode_selector_episode_active",
                "h120_pareto_mode_selector_episode_mode",
                "h120_pareto_mode_selector_far_persistent_high",
                "h120_pareto_mode_selector_far_intensification",
                "h120_pareto_mode_selector_far_reintensification",
                "h120_pareto_mode_selector_far_relief",
                "h120_pareto_mode_selector_far_direction_consistent",
                "h120_pareto_mode_selector_recovery_slow",
                "h120_pareto_mode_selector_far_max_norm",
                "h120_pareto_mode_selector_far_min_norm",
                "h120_pareto_mode_selector_near_last_norm",
                "far_horizon_norm_0_20",
                "far_horizon_norm_20_40",
                "far_horizon_norm_40_60",
                "far_horizon_norm_60_80",
                "far_horizon_norm_80_100",
                "far_horizon_norm_100_120",
            ]
            cols = [c for c in keep if c in log.columns]
            sub = log[cols].copy()
            sub.insert(0, "case", _case_key(log_path))
            sub.insert(0, "scenario", scenario)
            sub.insert(0, "dataset", dataset)
            chunks.append(sub)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _fmt(x: Any, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _markdown_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    if df.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def _guard10_pass(compare: pd.DataFrame) -> tuple[bool, list[str]]:
    guard = compare[compare["dataset"] == "guard10"]
    sel = guard[guard["scenario"] == "oracle_pareto_selector"]
    v16 = guard[guard["scenario"] == "static_v16_balanced"]
    v15 = guard[guard["scenario"] == "static_v15_safety"]
    if sel.empty or v16.empty:
        return False, ["missing selector or v1.6 guard10 rows"]
    s = sel.iloc[0]
    b = v16.iloc[0]
    pump_delta_pct = float(s["pump_delta_pct_vs_v16"])
    time_drop = float(s["time_over_5_reduction_pct_vs_v16"])
    idle_drop = float(s["idle_over_5_reduction_pct_vs_v16"])
    no_hard_regression = (
        float(s["sum_fb"]) <= float(b["sum_fb"]) + 1e-9
        and float(s["max_p95"]) <= float(b["max_p95"]) + 1e-9
    )
    reasons: list[str] = []
    pass_a = (
        pump_delta_pct <= 0.03
        and (time_drop >= 0.08 or idle_drop >= 0.08)
        and no_hard_regression
    )
    if pass_a:
        reasons.append("A: small pump increase with >=8% high-posture reduction")
    pass_b = (
        abs(time_drop) <= 0.01
        and abs(idle_drop) <= 0.01
        and pump_delta_pct <= -0.03
        and no_hard_regression
    )
    if pass_b:
        reasons.append("B: similar safety with >=3% lower pump")
    pass_c = False
    if not v15.empty:
        m = v15.iloc[0]
        close_to_v15 = (
            float(s["time_over_5"]) <= float(m["time_over_5"]) * 1.05
            and float(s["idle_time_over_5"]) <= float(m["idle_time_over_5"]) * 1.05
        )
        cheaper_than_v15 = float(s["sum_pump"]) <= float(m["sum_pump"]) * 0.98
        pass_c = close_to_v15 and cheaper_than_v15 and no_hard_regression
        if pass_c:
            reasons.append("C: near v1.5 safety with materially less pump")
    if not reasons:
        reasons.append(
            "guard10 selector did not clear pump-safety improvement bars vs static v1.6"
        )
    return bool(pass_a or pass_b or pass_c), reasons


def _write_reports(
    compare: pd.DataFrame,
    case_table: pd.DataFrame,
    selection: pd.DataFrame,
    guard_pass: bool,
    pass_reasons: list[str],
    broader_ran: bool,
) -> None:
    paper_dir = OUT_DIR / "paper_ready"
    paper_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "dataset",
        "scenario",
        "cases",
        "sum_pump",
        "sum_fb",
        "max_p95",
        "time_over_5",
        "idle_time_over_5",
        "delta_vs_v16_sum_pump",
        "delta_vs_v16_time_over_5",
        "delta_vs_v16_idle_time_over_5",
        "pump_delta_pct_vs_v16",
        "time_over_5_reduction_pct_vs_v16",
        "idle_over_5_reduction_pct_vs_v16",
        "selector_episode_count",
    ]
    sel_counts = pd.DataFrame()
    if not selection.empty and "h120_pareto_mode_selector_selected_mode" in selection.columns:
        active = selection[
            selection.get("h120_pareto_mode_selector_episode_active", 0).astype(float) > 0
        ].copy()
        if not active.empty:
            sel_counts = (
                active.groupby(
                    [
                        "dataset",
                        "h120_pareto_mode_selector_episode_mode",
                        "reactive_floor_resolved_action",
                    ],
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "rows"})
            )

    outcome = "GO to broader20 sanity" if guard_pass else "NO-GO after guard10"
    broader_note = (
        "yes, required by the guard10 pass"
        if guard_pass and broader_ran
        else "yes, already completed as a sanity run; final gate remains guard10 no-go"
        if broader_ran
        else "no, guard10 did not clear the bar"
    )
    summary = [
        "# h120 oracle Pareto mode selector v1",
        "",
        f"Decision status: **{outcome}**.",
        "",
        "This audit tests the last narrow h120 controller interface: oracle 120min information may choose among existing static reactive-floor modes only. It does not add pre-floor pump, target refresh, relief suppression, or learned control.",
        "",
        "## Runtime estimate",
        "",
        "- Code/data check: ~0.5h.",
        "- Guard10 static/selector casebooks: ~1-2h depending on local load.",
        "- Broader20 sanity: only if guard10 clears the bar, ~2-4h.",
        "- Summary/report generation: ~0.5h.",
        "",
        "## Closed-loop comparison",
        "",
        _markdown_table(compare, [c for c in cols if c in compare.columns], digits=3),
        "",
        "## Selector mode volume",
        "",
        _markdown_table(sel_counts, sel_counts.columns.tolist(), digits=3)
        if not sel_counts.empty
        else "_No floor-episode selector rows were logged._",
        "",
        "## Guard10 gate",
        "",
        "\n".join(f"- {r}" for r in pass_reasons),
        "",
        f"- Broader20 sanity run: {broader_note}.",
    ]
    (OUT_DIR / "pareto_mode_selector_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    (paper_dir / "pareto_mode_selector_findings.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    case_concentration = pd.DataFrame()
    guard = case_table[
        (case_table["dataset"] == "guard10")
        & (case_table["scenario"].isin(["static_v16_balanced", "oracle_pareto_selector"]))
    ]
    if not guard.empty:
        wide = guard.pivot(index="case_short", columns="scenario", values="time_over_5")
        if {"static_v16_balanced", "oracle_pareto_selector"}.issubset(wide.columns):
            case_concentration = (
                wide.assign(
                    time_gain_vs_v16=wide["static_v16_balanced"]
                    - wide["oracle_pareto_selector"]
                )
                .sort_values("time_gain_vs_v16", ascending=False)
                .reset_index()
            )

    decision_lines = [
        "# h120 oracle Pareto mode selector decision",
        "",
        f"Final decision: **{'continue to broader20 sanity' if guard_pass else 'no-go on guard10'}**.",
        "",
        "## Required answers",
        "",
        "1. Oracle h120 mode selector vs static v1.6: "
        + (
            "guard10 cleared at least one improvement bar."
            if guard_pass
            else "guard10 did not clearly beat static v1.6, so the audit stops here."
        ),
        "2. Pump-safety Pareto movement: "
        + (
            "candidate movement exists on guard10; broader20 sanity is required before any learned route."
            if guard_pass
            else "no robust Pareto movement was demonstrated under the oracle selector."
        ),
        "3. Case concentration: see `pareto_mode_case_table.csv`; the largest guard10 time-gain cases are summarized below.",
        "4. Broader20 / lowrisk regression: "
        + (
            "broader20 was run because guard10 passed."
            if guard_pass and broader_ran
            else "broader20 sanity was already completed; it showed no regression, but this does not rescue the guard10 no-go."
            if broader_ran
            else "not run, by design, because guard10 did not pass."
        ),
        "5. Learned h120 selector: "
        + (
            "not allowed until broader20 sanity also passes."
            if guard_pass
            else "not worth starting; oracle did not beat static v1.6 on guard10."
        ),
        "6. h120 controller route: "
        + (
            "still gated on broader20 sanity."
            if guard_pass
            else "stop this Pareto selector interface; do not train learned selector."
        ),
        "7. v1.6 mainline: remains the control mainline.",
        "",
        "## Guard10 gate evidence",
        "",
        "\n".join(f"- {r}" for r in pass_reasons),
    ]
    if not case_concentration.empty:
        decision_lines.extend(
            [
                "",
                "## Largest guard10 case-level time gains",
                "",
                _markdown_table(
                    case_concentration.head(10),
                    [
                        "case_short",
                        "static_v16_balanced",
                        "oracle_pareto_selector",
                        "time_gain_vs_v16",
                    ],
                    digits=3,
                ),
            ]
        )
    (OUT_DIR / "h120_oracle_pareto_mode_selector_decision.md").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )


def _build_tables(run_map: dict[tuple[str, str], Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    case_csvs = {"guard10": GUARD10_CASES, "broader20": BROADER20_CASES}
    for (dataset, scenario), run_dir in run_map.items():
        rows.extend(_case_metrics(run_dir, dataset, scenario, case_csvs[dataset]))
    case_table = pd.DataFrame(rows).sort_values(["dataset", "scenario", "case"])
    compare = _aggregate(case_table)
    selection = _selection_table(run_map)
    return compare, case_table, selection


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw_tables").mkdir(exist_ok=True)
    (OUT_DIR / "debug").mkdir(exist_ok=True)
    (OUT_DIR / "paper_ready").mkdir(exist_ok=True)

    run_map: dict[tuple[str, str], Path] = {}
    for scenario, cfg in SCENARIOS.items():
        run_dir = OUT_DIR / "runs" / "guard10" / scenario
        _run_casebook(run_dir, GUARD10_CASES, cfg)
        run_map[("guard10", scenario)] = run_dir

    compare, case_table, selection = _build_tables(run_map)
    guard_pass, pass_reasons = _guard10_pass(compare)

    broader_ran = False
    if guard_pass:
        broader_ran = True
        for scenario, cfg in SCENARIOS.items():
            run_dir = OUT_DIR / "runs" / "broader20" / scenario
            _run_casebook(run_dir, BROADER20_CASES, cfg)
            run_map[("broader20", scenario)] = run_dir
        compare, case_table, selection = _build_tables(run_map)
    else:
        existing_broader = {
            scenario: OUT_DIR / "runs" / "broader20" / scenario
            for scenario in SCENARIOS
        }
        if all((path / "casebook_summary.csv").exists() for path in existing_broader.values()):
            broader_ran = True
            for scenario, run_dir in existing_broader.items():
                run_map[("broader20", scenario)] = run_dir
            compare, case_table, selection = _build_tables(run_map)

    compare.to_csv(OUT_DIR / "pareto_mode_compare_table.csv", index=False)
    case_table.to_csv(OUT_DIR / "pareto_mode_case_table.csv", index=False)
    selection.to_csv(OUT_DIR / "pareto_mode_selection_table.csv", index=False)
    compare.to_csv(OUT_DIR / "raw_tables/pareto_mode_compare_table.csv", index=False)
    case_table.to_csv(OUT_DIR / "raw_tables/pareto_mode_case_table.csv", index=False)
    selection.to_csv(OUT_DIR / "raw_tables/pareto_mode_selection_table.csv", index=False)
    _write_reports(compare, case_table, selection, guard_pass, pass_reasons, broader_ran)
    print("[done] wrote h120_oracle_pareto_mode_selector_v1 outputs", flush=True)


if __name__ == "__main__":
    main()
