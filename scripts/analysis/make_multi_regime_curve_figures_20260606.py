#!/usr/bin/env python3
"""Generate multi-regime curve figures with pitch and roll split.

This script is intentionally presentation-oriented:
- pitch and roll are never plotted in the same output image;
- pump rate and cumulative pump work are shown in pounds;
- each regime gets five different figure types, using varied representative cases.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = REPO_ROOT / "outputs" / "wind_prediction"
FIG_ROOT = OUT_ROOT / "multi_regime_curve_figures_20260606_py312"

KG_PER_M3 = 1025.0
LB_PER_KG = 2.20462262185
LB_PER_M3 = KG_PER_M3 * LB_PER_KG

TS_COLS = {
    "t_s",
    "wind_speed",
    "wind_dir_deg",
    "pitch_deg",
    "roll_deg",
    "pump_rate1_m3min",
    "pump_rate2_m3min",
    "pump_rate3_m3min",
    "pump_total_rate_m3_min",
    "tank1_kg",
    "tank2_kg",
    "tank3_kg",
    "target_tank1_kg",
    "target_tank2_kg",
    "target_tank3_kg",
    "preview_primary_safety_fallback",
    "preview_primary_safety_active",
    "preview_primary_action",
    "pump_latch_switch_count",
    "pump_stage_switch_count",
    "pump_fullspeed_any",
    "ctrl_status",
}

COLORS = {
    "Closed": "#1f4e79",
    "Candidate": "#c7532c",
    "Current-only": "#5f6b7a",
    "Learned": "#1f77b4",
    "PSC": "#2f6f9f",
    "wind": "#315f8c",
    "wind_dir": "#77808a",
    "tank": "#2d6a4f",
    "target": "#bc6c25",
}


@dataclass(frozen=True)
class VariantSpec:
    label: str
    run_dir: Path
    suffix: str


@dataclass(frozen=True)
class RegimeSpec:
    slug: str
    title: str
    description: str
    variants: tuple[VariantSpec, ...]
    max_cases: int = 5


REGIMES = (
    RegimeSpec(
        slug="d1_neutral_headroom",
        title="D1 neutral/headroom redundant pump",
        description="Clean neutral/headroom windows with DC-preserving economy behavior.",
        variants=(
            VariantSpec(
                "Closed",
                OUT_ROOT / "selector_positive_add40_6h_v1" / "blind_d1_engineered_add40_120case_6h",
                "closed_only",
            ),
            VariantSpec(
                "Candidate",
                OUT_ROOT / "selector_positive_add40_6h_v1" / "blind_d1_engineered_add40_120case_6h",
                "prediction_primary_econ",
            ),
        ),
    ),
    RegimeSpec(
        slug="c3_gusty_oscillation",
        title="C3 gusty/repeated-peak release",
        description="Gusty oscillatory windows with balanced release-pool candidate.",
        variants=(
            VariantSpec(
                "Closed",
                OUT_ROOT / "c3_balanced_release_pool_12h_candidate_20260605",
                "closed_only",
            ),
            VariantSpec(
                "Candidate",
                OUT_ROOT / "c3_balanced_release_pool_12h_candidate_20260605",
                "c3_balanced_release_pool_12h",
            ),
        ),
    ),
    RegimeSpec(
        slug="p2_long_boundary",
        title="P2 long-window and boundary validation",
        description="Long P2 and ramp/boundary validation cases from the 2026-06-05 3.12 run.",
        variants=(
            VariantSpec(
                "Closed",
                OUT_ROOT / "p2_relaxed_resume_guard_confirm4_4case_6h_validation_20260605",
                "closed_only",
            ),
            VariantSpec(
                "Candidate",
                OUT_ROOT / "p2_relaxed_resume_guard_confirm4_4case_6h_validation_20260605",
                "p2_relaxed_resume_guard_confirm4_4case_6h",
            ),
        ),
    ),
    RegimeSpec(
        slug="guard10_mixed_12h",
        title="Guard10 mixed 12 h learned vs current-only",
        description="Mixed guard set comparing learned gain-0.45 output against current-only baseline output.",
        variants=(
            VariantSpec(
                "Current-only",
                OUT_ROOT / "guard10_12h_current_only_baseline_20260605",
                "prediction_primary_econ",
            ),
            VariantSpec(
                "Learned",
                OUT_ROOT / "guard10_12h_production_learned_gain045_20260605",
                "prediction_primary_econ",
            ),
        ),
    ),
    RegimeSpec(
        slug="psc_selector_mixed_pool",
        title="PSC selector mixed pool",
        description="Structured chain-guard mixed pool; single PSC output shown without fabricating a comparison trace.",
        variants=(
            VariantSpec(
                "PSC",
                OUT_ROOT
                / "psc_selector_mixed_pool_v1"
                / "chain_guard_structured_validation_20260605"
                / "baseline",
                "current_forecast_adaptive",
            ),
        ),
    ),
)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d7dce2", lw=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4ad")
    ax.spines["bottom"].set_color("#9aa4ad")


def _safe_num(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _read_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "casebook_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _find_timeseries(variant: VariantSpec, case_id: str) -> Path:
    ts_dir = variant.run_dir / "timeseries"
    pattern = f"{case_id}_*_{variant.suffix}_timeseries.csv"
    matches = sorted(ts_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {ts_dir}, got {len(matches)}")
    return matches[0]


def _read_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda col: col in TS_COLS)
    if "t_s" not in df.columns:
        df["t_s"] = np.arange(len(df), dtype=float)
    df["t_min"] = pd.to_numeric(df["t_s"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 60.0
    return df


def _label_for_case(summary: pd.DataFrame, case_id: str) -> str:
    if "case_id" not in summary.columns:
        return case_id
    row = summary[summary["case_id"].astype(str).eq(case_id)]
    if row.empty or "label" not in row.columns:
        return case_id
    label = str(row.iloc[0].get("label", ""))
    return label[:150] if label else case_id


def _case_metrics_from_summary(regime: RegimeSpec) -> pd.DataFrame:
    summaries = []
    for variant in regime.variants:
        df = _read_summary(variant.run_dir).copy()
        df["variant_label"] = variant.label
        summaries.append(df)

    base = summaries[0].copy()
    out = base[["case_id", "timestamp", "label"]].copy()
    out["case_id"] = out["case_id"].astype(str)

    if len(summaries) >= 2:
        comp = summaries[0].merge(
            summaries[-1],
            on="case_id",
            how="inner",
            suffixes=("_a", "_b"),
        )
        rows = []
        for _, row in comp.iterrows():
            a_pump = _safe_num(row.get("primary_pump_work_m3_a", row.get("closed_pump_work_m3_a")))
            b_pump = _safe_num(row.get("primary_pump_work_m3_b", row.get("primary_pump_work_m3_b")))
            if math.isnan(a_pump):
                a_pump = _safe_num(row.get("closed_pump_work_m3_b"))
            if math.isnan(b_pump):
                b_pump = _safe_num(row.get("primary_pump_work_m3_b"))
            rows.append(
                {
                    "case_id": str(row["case_id"]),
                    "timestamp": row.get("timestamp_a", row.get("timestamp_b", "")),
                    "label": row.get("label_a", row.get("label_b", "")),
                    "pump_delta_m3": b_pump - a_pump if math.isfinite(a_pump) and math.isfinite(b_pump) else math.nan,
                    "pump_a_m3": a_pump,
                    "pump_b_m3": b_pump,
                    "pitch_delta": _safe_num(row.get("primary_pitch_p95_b")) - _safe_num(row.get("primary_pitch_p95_a")),
                    "roll_delta": _safe_num(row.get("primary_roll_p95_b")) - _safe_num(row.get("primary_roll_p95_a")),
                    "pitch_b": _safe_num(row.get("primary_pitch_p95_b")),
                    "roll_b": _safe_num(row.get("primary_roll_p95_b")),
                    "fallback_b": _safe_num(row.get("primary_safety_fallback_ratio_b"), 0.0),
                }
            )
        return pd.DataFrame(rows)

    out["pump_delta_m3"] = np.nan
    out["pump_a_m3"] = np.nan
    out["pump_b_m3"] = pd.to_numeric(base.get("primary_pump_work_m3", np.nan), errors="coerce")
    out["pitch_delta"] = np.nan
    out["roll_delta"] = np.nan
    out["pitch_b"] = pd.to_numeric(base.get("primary_pitch_p95", np.nan), errors="coerce")
    out["roll_b"] = pd.to_numeric(base.get("primary_roll_p95", np.nan), errors="coerce")
    out["fallback_b"] = pd.to_numeric(base.get("primary_safety_fallback_ratio", 0.0), errors="coerce")
    return out


def _pick_cases(metrics: pd.DataFrame, max_cases: int) -> list[str]:
    if metrics.empty:
        return []
    metrics = metrics.copy()
    metrics["case_id"] = metrics["case_id"].astype(str)
    picks: list[str] = []

    def add_from(df: pd.DataFrame) -> None:
        for cid in df["case_id"].astype(str).tolist():
            if cid not in picks:
                picks.append(cid)
            if len(picks) >= max_cases:
                return

    for col, ascending in [
        ("pump_delta_m3", True),
        ("pump_delta_m3", False),
        ("pitch_delta", False),
        ("roll_delta", False),
        ("pump_b_m3", False),
        ("pitch_b", False),
        ("roll_b", False),
        ("fallback_b", False),
    ]:
        if col in metrics.columns and metrics[col].notna().any():
            add_from(metrics.sort_values(col, ascending=ascending))
        if len(picks) >= max_cases:
            return picks

    add_from(metrics.sort_values("case_id"))
    return picks[:max_cases]


def _time_min(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["t_min"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=float)
    return pd.to_numeric(df[col], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)


def _dt_min(df: pd.DataFrame) -> float:
    t = _series(df, "t_s")
    if len(t) < 2:
        return 1.0 / 60.0
    dt_s = float(np.nanmedian(np.diff(t)))
    return max(dt_s, 1e-9) / 60.0


def _cum_pump_lb(df: pd.DataFrame) -> np.ndarray:
    rate_m3_min = np.abs(_series(df, "pump_total_rate_m3_min"))
    return np.cumsum(rate_m3_min * _dt_min(df) * LB_PER_M3)


def _pump_work_lb(df: pd.DataFrame) -> float:
    arr = _cum_pump_lb(df)
    return float(arr[-1]) if len(arr) else 0.0


def _total_mass_lb(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = [f"{prefix}{i}_kg" for i in (1, 2, 3)]
    if not all(col in df.columns for col in cols):
        return np.zeros(len(df), dtype=float)
    return sum(_series(df, col) for col in cols) * LB_PER_KG


def _fallback_mask(df: pd.DataFrame) -> np.ndarray:
    for col in ("preview_primary_safety_fallback", "preview_primary_safety_active", "pump_fullspeed_any"):
        if col in df.columns and np.nanmax(np.abs(_series(df, col))) > 0:
            return _series(df, col) > 0.5
    return np.zeros(len(df), dtype=bool)


def _shade_mask(ax: plt.Axes, df: pd.DataFrame, color: str = "#e76f51", alpha: float = 0.12) -> None:
    mask = _fallback_mask(df)
    if not np.any(mask):
        return
    t = _time_min(df)
    start: float | None = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = float(t[i])
        if start is not None and (not active or i == len(mask) - 1):
            end = float(t[i])
            ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
            start = None


def _plot_grid(
    regime: RegimeSpec,
    cases: list[str],
    bundles: dict[str, dict[str, pd.DataFrame]],
    summaries: dict[str, pd.DataFrame],
    filename: str,
    suptitle: str,
    plotter,
    height_per_case: float = 2.35,
) -> Path:
    out_dir = FIG_ROOT / regime.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = max(1, len(cases))
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(13.5, max(6.0, height_per_case * rows)),
        sharex=False,
        constrained_layout=True,
    )
    if rows == 1:
        axes = [axes]
    for ax, case_id in zip(axes, cases):
        case_label = _label_for_case(next(iter(summaries.values())), case_id)
        plotter(ax, case_id, bundles[case_id], case_label)
        _style_axis(ax)
    fig.suptitle(suptitle, fontsize=14, y=1.01)
    out = out_dir / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_wind(regime: RegimeSpec, cases, bundles, summaries) -> Path:
    def plotter(ax, case_id, variants, case_label):
        first = next(iter(variants.values()))
        t = _time_min(first)
        ax.plot(t, _series(first, "wind_speed"), color=COLORS["wind"], lw=1.2, label="wind speed")
        ax.set_ylabel("wind\nm/s")
        ax2 = ax.twinx()
        ax2.plot(t, _series(first, "wind_dir_deg"), color=COLORS["wind_dir"], lw=0.9, alpha=0.78, label="wind dir")
        ax2.set_ylabel("dir deg")
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax.set_title(f"{case_id} | {case_label}", fontsize=9, loc="left")
        ax.set_xlabel("time min")

    return _plot_grid(regime, cases, bundles, summaries, "01_wind_overview.png", f"{regime.title}: wind overview", plotter)


def _plot_attitude_axis(regime: RegimeSpec, cases, bundles, summaries, axis_col: str) -> Path:
    axis_name = "pitch" if axis_col == "pitch_deg" else "roll"
    thresh = 5.0

    def plotter(ax, case_id, variants, case_label):
        for label, df in variants.items():
            color = COLORS.get(label, "#333333")
            ax.plot(_time_min(df), _series(df, axis_col), color=color, lw=1.1, label=label)
            _shade_mask(ax, df, alpha=0.08)
        ax.axhline(thresh, color="#9c2f2f", lw=0.8, ls="--")
        ax.axhline(-thresh, color="#9c2f2f", lw=0.8, ls="--")
        ax.axhline(0.0, color="#687384", lw=0.7)
        ax.set_ylabel(f"{axis_name}\ndeg")
        ax.set_title(f"{case_id} | {case_label}", fontsize=9, loc="left")
        ax.legend(ncol=min(3, len(variants)), frameon=False, loc="upper right")
        ax.set_xlabel("time min")

    return _plot_grid(
        regime,
        cases,
        bundles,
        summaries,
        f"02_{axis_name}_separate.png" if axis_name == "pitch" else f"03_{axis_name}_separate.png",
        f"{regime.title}: {axis_name} only",
        plotter,
    )


def _plot_pump(regime: RegimeSpec, cases, bundles, summaries) -> Path:
    def plotter(ax, case_id, variants, case_label):
        ax2 = ax.twinx()
        for label, df in variants.items():
            color = COLORS.get(label, "#333333")
            rate_lb_min = np.abs(_series(df, "pump_total_rate_m3_min")) * LB_PER_M3
            cum_lb = _cum_pump_lb(df)
            ax.plot(_time_min(df), rate_lb_min, color=color, lw=0.8, alpha=0.78, label=f"{label} rate")
            ax2.plot(_time_min(df), cum_lb, color=color, lw=1.5, ls="--", label=f"{label} cum")
        ax.set_ylabel("pump rate\nlb/min")
        ax2.set_ylabel("cum pump\nlb")
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        totals = ", ".join(f"{label}: {_pump_work_lb(df):,.0f} lb" for label, df in variants.items())
        ax.set_title(f"{case_id} | {totals}", fontsize=9, loc="left")
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, ncol=min(4, 2 * len(variants)), frameon=False, loc="upper left")
        ax.set_xlabel("time min")

    return _plot_grid(regime, cases, bundles, summaries, "04_pump_pounds.png", f"{regime.title}: pump in pounds", plotter, 2.55)


def _plot_mass_state(regime: RegimeSpec, cases, bundles, summaries) -> Path:
    def plotter(ax, case_id, variants, case_label):
        for label, df in variants.items():
            color = COLORS.get(label, "#333333")
            mass = _total_mass_lb(df, "tank")
            target = _total_mass_lb(df, "target_tank")
            scale = 1_000_000.0
            ax.plot(_time_min(df), mass / scale, color=color, lw=1.1, label=f"{label} mass")
            ax.plot(_time_min(df), target / scale, color=color, lw=0.9, ls="--", alpha=0.72, label=f"{label} target")
            _shade_mask(ax, df, alpha=0.08)
        ax.set_ylabel("ballast\nmillion lb")
        ax.set_title(f"{case_id} | {case_label}", fontsize=9, loc="left")
        ax.legend(ncol=min(4, 2 * len(variants)), frameon=False, loc="upper right")
        ax.set_xlabel("time min")

    return _plot_grid(regime, cases, bundles, summaries, "05_ballast_mass_targets_lb.png", f"{regime.title}: ballast mass and targets", plotter)


def _write_case_metrics(regime: RegimeSpec, cases: list[str], bundles: dict[str, dict[str, pd.DataFrame]], metrics: pd.DataFrame) -> Path:
    rows = []
    for case_id in cases:
        row = {"regime": regime.slug, "case_id": case_id}
        metric_row = metrics[metrics["case_id"].astype(str).eq(case_id)]
        if not metric_row.empty:
            row["source_label"] = metric_row.iloc[0].get("label", "")
            row["timestamp"] = metric_row.iloc[0].get("timestamp", "")
        for label, df in bundles[case_id].items():
            row[f"{label}_pump_work_lb"] = _pump_work_lb(df)
            row[f"{label}_pitch_p95_abs_deg"] = float(np.nanpercentile(np.abs(_series(df, "pitch_deg")), 95))
            row[f"{label}_roll_p95_abs_deg"] = float(np.nanpercentile(np.abs(_series(df, "roll_deg")), 95))
            row[f"{label}_fallback_samples"] = int(np.sum(_fallback_mask(df)))
        rows.append(row)
    out = FIG_ROOT / regime.slug / "selected_case_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _load_bundles(regime: RegimeSpec, cases: list[str]) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]:
    summaries = {variant.label: _read_summary(variant.run_dir) for variant in regime.variants}
    bundles: dict[str, dict[str, pd.DataFrame]] = {}
    for case_id in cases:
        variants: dict[str, pd.DataFrame] = {}
        for variant in regime.variants:
            variants[variant.label] = _read_timeseries(_find_timeseries(variant, case_id))
        bundles[case_id] = variants
    return bundles, summaries


def build_regime(regime: RegimeSpec) -> list[Path]:
    metrics = _case_metrics_from_summary(regime)
    cases = _pick_cases(metrics, regime.max_cases)
    bundles, summaries = _load_bundles(regime, cases)
    paths = [
        _plot_wind(regime, cases, bundles, summaries),
        _plot_attitude_axis(regime, cases, bundles, summaries, "pitch_deg"),
        _plot_attitude_axis(regime, cases, bundles, summaries, "roll_deg"),
        _plot_pump(regime, cases, bundles, summaries),
        _plot_mass_state(regime, cases, bundles, summaries),
        _write_case_metrics(regime, cases, bundles, metrics),
    ]
    selected = pd.DataFrame({"case_id": cases})
    selected.to_csv(FIG_ROOT / regime.slug / "selected_cases.csv", index=False)
    return paths


def write_manifest(all_paths: dict[str, list[Path]]) -> Path:
    lines = [
        "# Multi-Regime Curve Figures 20260606 py312",
        "",
        "Generated with `.venv312/bin/python3.12`.",
        "",
        "Pitch and roll are separated into different PNG files for every regime.",
        "Pump rate is shown in `lb/min`; cumulative pump work and ballast mass are shown in `lb`.",
        "",
    ]
    total_png = 0
    for regime in REGIMES:
        lines.extend([f"## {regime.title}", "", regime.description, ""])
        for path in all_paths.get(regime.slug, []):
            rel = path.relative_to(REPO_ROOT)
            if path.suffix.lower() == ".png":
                total_png += 1
            lines.append(f"- `{rel}`")
        lines.append("")
    lines.append(f"Total PNG figures: `{total_png}`")
    out = FIG_ROOT / "README.md"
    out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    all_paths: dict[str, list[Path]] = {}
    for regime in REGIMES:
        print(f"building {regime.slug}...")
        paths = build_regime(regime)
        all_paths[regime.slug] = paths
    manifest = write_manifest(all_paths)
    print(manifest.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
