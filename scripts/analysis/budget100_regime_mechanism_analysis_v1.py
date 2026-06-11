#!/usr/bin/env python3
"""Analyze which regimes make budget100 save pump, and why.

This is a paper-facing interpretation script.  It combines existing locked53
and natural relief/decay expansion outputs, groups cases by observable regime,
and renders representative trajectory figures for:

- clean transient-peak/future-decay saving,
- residual/high-load saving,
- low-risk redundant pump saving,
- catch-up/fallback boundary failure.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1")
FINAL = ROOT / "budget100_final_candidate_v1"
RELIEF = ROOT / "relief_decay_expansion_v1"
RAW = FINAL / "raw_tables"
PAPER = FINAL / "paper_ready"
FIG = PAPER / "figures" / "budget100_regimes"


def series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if np.isscalar(default):
            return s.fillna(default)
        return s.fillna(pd.Series(default, index=df.index))
    return pd.Series(default, index=df.index, dtype=float)


def fallback_flag(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "target_lookup_fallback",
        "preview_primary_safety_fallback",
        "fallback_reason_missing_state",
        "fallback_reason_missing_table",
        "fallback_reason_missing_columns",
        "fallback_reason_invalid_state_id",
    ]
    out = np.zeros(len(df), dtype=bool)
    for col in cols:
        if col in df.columns:
            out |= series(df, col).to_numpy() > 0.5
    return out


def load_locked() -> pd.DataFrame:
    d = pd.read_csv(RAW / "locked53_budget100_expansion_case_table.csv")
    d["source_casebook"] = "locked53"
    d["a0_pump_m3"] = d["pump_m3_a0"]
    return d


def load_expansion() -> pd.DataFrame:
    delta = pd.read_csv(RELIEF / "raw_tables" / "relief_decay_expansion_case_delta.csv")
    cases = pd.read_csv(RELIEF / "relief_decay_expansion_cases.csv")
    delta["case_id"] = delta["case"].str.replace(
        r"^\d+_", "", regex=True
    ).str.replace(r"_\d{4}-\d{2}-\d{2}_\d{6}$", "", regex=True)
    delta["label"] = "natural fixed-rule transient-peak/future-decay"
    delta["source_casebook"] = "new24_relief_decay"
    delta["forecast_relief_decay_candidate"] = True
    delta["h120_supervisory_relief_candidate"] = True
    delta["direction_reversal_candidate"] = False
    delta["opportunity_relief_decay_pump_ge150"] = delta["a0_pump_m3"] >= 150
    delta["is_lowrisk_label"] = False
    delta["timestamp"] = ""
    # Keep a copy of the original fixed-rule label when case order is available.
    if len(cases) == len(delta):
        delta["timestamp"] = cases["timestamp"]
        delta["label"] = cases["label"]
    return delta


def classify(row: pd.Series) -> str:
    if row["pump_saved_m3"] < -50 or row["delta_fallback_time_s"] > 300:
        return "boundary_catchup_or_fallback"
    if row.get("is_lowrisk_label", False) and row["pump_saved_m3"] > 20:
        return "lowrisk_redundant_pump"
    if row.get("forecast_relief_decay_candidate", False) and row["pump_saved_m3"] > 20:
        return "transient_peak_future_decay"
    if (
        (not row.get("forecast_relief_decay_candidate", False))
        and row["a0_pump_m3"] >= 150
        and row["pump_saved_m3"] > 50
    ):
        return "residual_high_or_general_economy"
    if row["pump_saved_m3"] > 20:
        return "misc_positive"
    return "neutral_or_low_opportunity"


def summarize(regimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, d in regimes.groupby("regime"):
        a0 = float(d["a0_pump_m3"].sum())
        saved = float(d["pump_saved_m3"].sum())
        rows.append(
            {
                "regime": regime,
                "cases": int(len(d)),
                "a0_pump_m3": a0,
                "pump_saved_m3": saved,
                "pump_saving_pct": 100.0 * saved / a0 if a0 > 1e-9 else 0.0,
                "delta_time_gt5_s": int(d["delta_time_gt5_s"].sum()),
                "delta_idle_gt5_s": int(d["delta_idle_gt5_s"].sum()),
                "delta_fallback_time_s": int(d["delta_fallback_time_s"].sum()),
                "mean_delta_p95_max_axis_deg": float(d["delta_p95_max_axis_deg"].mean())
                if len(d)
                else 0.0,
                "negative_saving_cases": int((d["pump_saved_m3"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("pump_saved_m3", ascending=False)


def paths_for_case(source: str, case: str) -> tuple[Path, Path] | None:
    if source == "locked53":
        a0 = (
            Path("outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1")
            / "runs"
            / "A0_learned_v16"
            / "timeseries"
            / f"{case}_prediction_primary_econ_timeseries.csv"
        )
        b = (
            ROOT
            / "locked53_7200"
            / "budget100"
            / "timeseries"
            / f"{case}_prediction_primary_econ_timeseries.csv"
        )
    else:
        a0 = RELIEF / "a0_baseline" / "timeseries" / f"{case}_prediction_primary_econ_timeseries.csv"
        b = RELIEF / "budget100" / "timeseries" / f"{case}_prediction_primary_econ_timeseries.csv"
    if a0.exists() and b.exists():
        return a0, b
    return None


def plot_case(row: pd.Series, slug: str) -> Path | None:
    paths = paths_for_case(str(row["source_casebook"]), str(row["case"]))
    if paths is None:
        return None
    a0 = pd.read_csv(paths[0], low_memory=False)
    b = pd.read_csv(paths[1], low_memory=False)
    n = min(len(a0), len(b))
    a0 = a0.iloc[:n]
    b = b.iloc[:n]
    t = series(a0, "t_s", default=np.arange(n)).to_numpy() / 60.0
    a0_pitch = series(a0, "pitch_deg").to_numpy()
    a0_roll = series(a0, "roll_deg").to_numpy()
    b_pitch = series(b, "pitch_deg").to_numpy()
    b_roll = series(b, "roll_deg").to_numpy()
    a0_axis = np.maximum(np.abs(a0_pitch), np.abs(a0_roll))
    b_axis = np.maximum(np.abs(b_pitch), np.abs(b_roll))
    a0_pump = np.abs(series(a0, "pump_total_rate_m3_min").to_numpy())
    b_pump = np.abs(series(b, "pump_total_rate_m3_min").to_numpy())
    a0_fb = fallback_flag(a0)
    b_fb = fallback_flag(b)
    cum_a0 = np.cumsum(a0_pump) / 60.0
    cum_b = np.cumsum(b_pump) / 60.0

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, a0_axis, label="A0 max-axis", lw=1.4)
    axes[0].plot(t, b_axis, label="budget100 max-axis", lw=1.4)
    axes[0].axhline(5.0, color="crimson", ls="--", lw=1, label="5 deg hard floor")
    axes[0].set_ylabel("max axis deg")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    axes[1].plot(t, a0_pump, label="A0 pump rate", lw=1.2)
    axes[1].plot(t, b_pump, label="budget100 pump rate", lw=1.2)
    axes[1].set_ylabel("m3/min")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, cum_a0, label="A0 cumulative", lw=1.4)
    axes[2].plot(t, cum_b, label="budget100 cumulative", lw=1.4)
    axes[2].set_ylabel("pump m3")
    axes[2].legend(loc="upper left", fontsize=8)

    axes[3].step(t, a0_fb.astype(int), where="post", label="A0 fallback", lw=1.1)
    axes[3].step(t, b_fb.astype(int), where="post", label="budget100 fallback", lw=1.1)
    axes[3].set_ylabel("fallback")
    axes[3].set_xlabel("time (min)")
    axes[3].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"{slug}: {row['case']} | saved {row['pump_saved_m3']:.1f} m3, "
        f"time>5 {row['delta_time_gt5_s']:+.0f}s, fallback {row['delta_fallback_time_s']:+.0f}s"
    )
    fig.tight_layout()
    out = FIG / f"{slug}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    locked = load_locked()
    expansion = load_expansion()
    common = pd.concat([locked, expansion], ignore_index=True, sort=False)
    common["regime"] = common.apply(classify, axis=1)
    common.to_csv(RAW / "budget100_regime_mechanism_case_table.csv", index=False)

    summary = summarize(common)
    summary.to_csv(RAW / "budget100_regime_mechanism_summary.csv", index=False)

    reps: list[tuple[str, pd.Series]] = []
    for regime, slug in [
        ("transient_peak_future_decay", "hero_transient_peak_decay"),
        ("residual_high_or_general_economy", "hero_residual_high_general_economy"),
        ("lowrisk_redundant_pump", "hero_lowrisk_redundant_pump"),
        ("boundary_catchup_or_fallback", "boundary_catchup_failure"),
    ]:
        d = common[common["regime"] == regime].copy()
        if d.empty:
            continue
        if regime == "boundary_catchup_or_fallback":
            row = d.sort_values(["delta_fallback_time_s", "pump_saved_m3"], ascending=[False, True]).iloc[0]
        else:
            row = d.sort_values("pump_saved_m3", ascending=False).iloc[0]
        reps.append((slug, row))
    fig_rows = []
    for slug, row in reps:
        path = plot_case(row, slug)
        if path is not None:
            fig_rows.append(
                {
                    "figure": str(path),
                    "slug": slug,
                    "case": row["case"],
                    "regime": row["regime"],
                    "pump_saved_m3": row["pump_saved_m3"],
                    "delta_time_gt5_s": row["delta_time_gt5_s"],
                    "delta_fallback_time_s": row["delta_fallback_time_s"],
                }
            )
    fig_table = pd.DataFrame(fig_rows)
    fig_table.to_csv(RAW / "budget100_regime_representative_figures.csv", index=False)

    cols = [
        "regime",
        "cases",
        "pump_saving_pct",
        "pump_saved_m3",
        "delta_time_gt5_s",
        "delta_idle_gt5_s",
        "delta_fallback_time_s",
        "mean_delta_p95_max_axis_deg",
        "negative_saving_cases",
    ]
    text = [
        "# Budget100 Regime Mechanism Analysis",
        "",
        "## What Makes The Algorithm Save Pump?",
        "",
        "The saving mechanism is not a single wind pattern. It is a reduction of non-hard-safety economy tracking: when A0 continues to chase a target or a transient load, budget100 allows the system to ride closer to the unchanged hard floor instead of spending pump immediately. This saves pump when the later trajectory does not require the skipped correction to be recovered; it fails when the skipped correction turns into catch-up/fallback.",
        "",
        "## Regime Summary",
        "",
        md_table(summary, cols),
        "",
        "## Mechanisms",
        "",
        "1. **Transient peak / future decay**: a near-term peak or high load is followed by decay. A0 pumps to chase the peak; budget100 holds economy action, and the load later relaxes. This is the cleanest and most forecast-explainable regime.",
        "",
        "2. **Residual high / general economy relaxation**: A0 still spends pump in high-load or residual-high cases even when part of that tracking is not essential for hard safety. budget100 saves pump, but the mechanism is broader economy relaxation rather than purely forecast-relief saving.",
        "",
        "3. **Low-risk redundant pump**: sparse cases where the platform is not strongly threatened but A0 still pumps. budget100 can remove these conservative corrections with little high-posture cost.",
        "",
        "4. **Boundary catch-up / fallback**: budget100 skips or delays an action that later becomes necessary. This creates catch-up pump and/or fallback. These cases define the operating boundary and should be shown honestly.",
        "",
        "## Representative Figures",
        "",
        md_table(fig_table, ["slug", "regime", "case", "pump_saved_m3", "delta_time_gt5_s", "delta_fallback_time_s", "figure"]) if not fig_table.empty else "No figures rendered.",
        "",
        "## Decision",
        "",
        "Use transient-peak/future-decay as the main paper story. Use residual-high and low-risk redundant-pump as supporting evidence that the economy layer contains broader removable pump. Use catch-up/fallback as the negative boundary explaining why budget100 must be an optional Pareto mode, not a safety-neutral automatic controller.",
        "",
    ]
    (PAPER / "budget100_regime_mechanism_analysis.md").write_text(
        "\n".join(text), encoding="utf-8"
    )

    zh = [
        "# Budget100 工况机制分析",
        "",
        "## 算法到底在什么情况下省泵？",
        "",
        "核心不是某一种单一风型，而是：A0 原控制器会继续追 target / 追临时扰动，`budget100` 放松 economy 层，让系统更靠近但不关闭 hard floor。只要后续轨迹不需要把这次少泵补回来，就会省泵；如果后续必须补回来，就会出现 catch-up 或 fallback。",
        "",
        "## 工况汇总",
        "",
        md_table(summary, cols),
        "",
        "## 具体机制",
        "",
        "1. **临时峰值 + 未来回落**：短时间风/压力上去，后面会下来。A0 会追这个峰值泵水，budget100 不追那么急，等扰动自己回落，所以省泵。这是最适合论文主线的机制。",
        "",
        "2. **高压后段 / residual high / general economy relaxation**：不一定有明确回落，但 A0 仍在做一些非 hard-safety 必需的精细追踪。budget100 砍掉这些 economy 追踪，能省泵，但这更像舒适性换泵量。",
        "",
        "3. **低风险冗余泵**：少数低风险工况里 A0 仍有保守泵动作，budget100 可以去掉。这类样本少，适合作为补充。",
        "",
        "4. **catch-up / fallback 边界**：如果少泵后后续没有自然缓解，或者方向/负载变化导致原来跳过的动作后来必须补，就会补泵甚至进入 fallback。这类不是有利工况，而是边界/失败案例。",
        "",
        "## 代表图",
        "",
        md_table(fig_table, ["slug", "regime", "case", "pump_saved_m3", "delta_time_gt5_s", "delta_fallback_time_s", "figure"]) if not fig_table.empty else "No figures rendered.",
        "",
        "## 决策",
        "",
        "论文主线建议抓“临时峰值 + 未来回落”。高压后段和低风险冗余泵作为补充说明 economy 层确实有可删泵量。catch-up/fallback 必须作为边界案例展示，用来解释为什么 budget100 是可选 Pareto 模式，而不是安全无代价自动控制器。",
        "",
    ]
    (PAPER / "budget100_regime_mechanism_analysis_zh.md").write_text(
        "\n".join(zh), encoding="utf-8"
    )
    print("Wrote regime mechanism analysis to", FINAL)


if __name__ == "__main__":
    main()
