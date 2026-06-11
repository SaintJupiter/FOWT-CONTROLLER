#!/usr/bin/env python3
"""Plot baseline-vs-candidate attitude recovery diagnostics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ATTENTION_DEG = 3.0
HIGH_DEG = 4.0
BUCKET_S = 600.0
COL_BASE = "#465a69"
COL_CAND = "#c7532c"
COL_PITCH = "#245c8f"
COL_ROLL = "#7d4f9f"
COL_ATT_BASE = "#6b7280"
COL_ATT_CAND = "#c7532c"
COL_PUMP_BASE = "#486581"
COL_PUMP_CAND = "#b64a2d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--attention-deg", type=float, default=ATTENTION_DEG)
    parser.add_argument("--high-deg", type=float, default=HIGH_DEG)
    parser.add_argument("--bucket-s", type=float, default=BUCKET_S)
    return parser.parse_args()


def _case_id_from_timeseries(path: Path) -> str:
    name = path.name
    match = re.match(
        r"(.+?)_\d{4}-\d{2}-\d{2}_\d{6}_.+_timeseries\.csv$",
        name,
    )
    if match:
        return match.group(1)
    suffix = "_prediction_primary_econ_timeseries.csv"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    match = re.match(r"(.+)_\d{4}-\d{2}-\d{2}_\d{6}$", name)
    return match.group(1) if match else name


def _norm_case(case_id: str) -> str:
    return re.sub(r"^\d+_", "", str(case_id).strip())


def _case_sort_key(case_id: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", str(case_id))
    return (int(match.group(1)) if match else 9999, str(case_id))


def _discover(casebook_dir: Path) -> dict[str, tuple[str, Path, Path | None]]:
    ts_dir = casebook_dir / "timeseries"
    log_dir = casebook_dir / "planner_logs"
    out: dict[str, tuple[str, Path, Path | None]] = {}
    for ts_path in sorted(ts_dir.glob("*_timeseries.csv")):
        case_id = _case_id_from_timeseries(ts_path)
        log_paths = sorted(log_dir.glob(f"{case_id}_*_planner_log.csv"))
        out[_norm_case(case_id)] = (case_id, ts_path, log_paths[0] if log_paths else None)
    return out


def _num(series: pd.Series | float | int, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series([float(series)])


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return _num(df[col], default)
    return pd.Series(default, index=df.index, dtype=float)


def _load_ts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame()
    out["t_min"] = _safe_col(df, "t_s") / 60.0
    out["pitch_deg"] = _safe_col(df, "pitch_deg")
    out["roll_deg"] = _safe_col(df, "roll_deg")
    out["attitude_abs_deg"] = np.maximum(out["pitch_deg"].abs(), out["roll_deg"].abs())
    out["pump_rate_m3_min"] = _safe_col(df, "pump_total_rate_m3_min")
    dt_min = out["t_min"].diff().median()
    if not np.isfinite(dt_min) or dt_min <= 0:
        dt_min = 1.0 / 60.0
    out["pump_cum_m3"] = (out["pump_rate_m3_min"] * dt_min).cumsum()
    fallback_cols = [
        "preview_primary_safety_fallback",
        "pump_fullspeed_any",
        "preview_primary_safety_active",
    ]
    fallback = pd.Series(0.0, index=df.index, dtype=float)
    for col in fallback_cols:
        if col in df.columns:
            fallback = _safe_col(df, col)
            break
    out["fallback"] = fallback
    if "preview_primary_action" in df.columns:
        out["action"] = df["preview_primary_action"].astype(str)
    else:
        out["action"] = ""
    return out


def _load_bucket_state(log_path: Path | None, threshold: float, bucket_s: float) -> pd.DataFrame:
    if log_path is None or not log_path.exists():
        return pd.DataFrame(columns=["bucket", "t_min", "state", "action", "attitude_abs"])
    log = pd.read_csv(log_path, low_memory=False)
    if "bucket" not in log.columns:
        return pd.DataFrame(columns=["bucket", "t_min", "state", "action", "attitude_abs"])
    out = pd.DataFrame()
    out["bucket"] = _num(log["bucket"], 0.0).astype(int)
    if "current_time_s" in log.columns:
        out["t_min"] = _num(log["current_time_s"], 0.0) / 60.0
    else:
        out["t_min"] = out["bucket"] * bucket_s / 60.0
    action_col = "first_action" if "first_action" in log.columns else "planner_first_action_raw"
    out["action"] = log[action_col].astype(str) if action_col in log.columns else ""
    pitch = _safe_col(log, "current_pitch_deg")
    roll = _safe_col(log, "current_roll_deg")
    out["attitude_abs"] = np.maximum(pitch.abs(), roll.abs())
    is_high = out["attitude_abs"] > threshold
    action_l = out["action"].str.lower()
    is_hold = action_l.isin(["hold", "pump_saving", "pause", "none", "nan", ""])
    is_active = action_l.str.startswith("active") | (~is_hold & action_l.ne(""))
    out["state"] = np.select(
        [is_high & is_hold, is_high & is_active, is_high],
        ["high_hold", "high_active", "high_other"],
        default="safe_or_low",
    )
    return out


def _style(ax) -> None:
    ax.grid(True, color="#d7dce2", lw=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4ad")
    ax.spines["bottom"].set_color("#9aa4ad")


def _shade_fallback(ax, ts: pd.DataFrame, color: str) -> None:
    if ts.empty or "fallback" not in ts:
        return
    active = ts["fallback"] > 0.5
    if not active.any():
        return
    t = ts["t_min"].to_numpy()
    flags = active.to_numpy()
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = t[i]
        if start is not None and (not flag or i == len(flags) - 1):
            end = t[i] if not flag else t[i]
            ax.axvspan(start, end, color=color, alpha=0.08, lw=0)
            start = None


def _plot_thresholds(ax, threshold: float, high: float, signed: bool = False) -> None:
    if signed:
        for val, color, ls in [
            (threshold, "#a77d21", "--"),
            (-threshold, "#a77d21", "--"),
            (high, "#a73535", ":"),
            (-high, "#a73535", ":"),
        ]:
            ax.axhline(val, color=color, lw=0.9, ls=ls, alpha=0.85)
    else:
        ax.axhline(threshold, color="#a77d21", lw=0.9, ls="--", alpha=0.9, label="3 deg band")
        ax.axhline(high, color="#a73535", lw=0.9, ls=":", alpha=0.9, label="4 deg band")


def plot_case(
    case_id: str,
    base_ts_path: Path,
    cand_ts_path: Path,
    base_log_path: Path | None,
    cand_log_path: Path | None,
    metric_row: pd.Series | None,
    out_dir: Path,
    threshold: float,
    high: float,
    bucket_s: float,
) -> Path:
    base = _load_ts(base_ts_path)
    cand = _load_ts(cand_ts_path)
    base_state = _load_bucket_state(base_log_path, threshold, bucket_s)
    cand_state = _load_bucket_state(cand_log_path, threshold, bucket_s)

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14.2, 11.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.1, 1.1, 0.55]},
        constrained_layout=True,
    )
    ax_pitch, ax_roll, ax_att, ax_pump, ax_state = axes

    for ax in [ax_pitch, ax_roll, ax_att, ax_pump]:
        _shade_fallback(ax, base, COL_BASE)
        _shade_fallback(ax, cand, COL_CAND)

    ax_pitch.plot(base["t_min"], base["pitch_deg"], color=COL_BASE, lw=1.35, label="baseline pitch")
    ax_pitch.plot(cand["t_min"], cand["pitch_deg"], color=COL_CAND, lw=1.35, label="candidate pitch")
    _plot_thresholds(ax_pitch, threshold, high, signed=True)
    ax_pitch.set_ylabel("pitch (deg)")
    ax_pitch.legend(loc="upper right", ncol=2, fontsize=8)
    _style(ax_pitch)

    ax_roll.plot(base["t_min"], base["roll_deg"], color=COL_BASE, lw=1.35, label="baseline roll")
    ax_roll.plot(cand["t_min"], cand["roll_deg"], color=COL_CAND, lw=1.35, label="candidate roll")
    _plot_thresholds(ax_roll, threshold, high, signed=True)
    ax_roll.set_ylabel("roll (deg)")
    ax_roll.legend(loc="upper right", ncol=2, fontsize=8)
    _style(ax_roll)

    ax_att.plot(base["t_min"], base["attitude_abs_deg"], color=COL_ATT_BASE, lw=1.55, label="baseline max(|pitch|, |roll|)")
    ax_att.plot(cand["t_min"], cand["attitude_abs_deg"], color=COL_ATT_CAND, lw=1.55, label="candidate max(|pitch|, |roll|)")
    _plot_thresholds(ax_att, threshold, high, signed=False)
    ax_att.fill_between(
        base["t_min"],
        threshold,
        base["attitude_abs_deg"],
        where=base["attitude_abs_deg"] > threshold,
        color=COL_BASE,
        alpha=0.08,
        interpolate=True,
    )
    ax_att.fill_between(
        cand["t_min"],
        threshold,
        cand["attitude_abs_deg"],
        where=cand["attitude_abs_deg"] > threshold,
        color=COL_CAND,
        alpha=0.10,
        interpolate=True,
    )
    ax_att.set_ylabel("attitude abs (deg)")
    ax_att.legend(loc="upper right", ncol=2, fontsize=8)
    _style(ax_att)

    ax_pump.plot(base["t_min"], base["pump_rate_m3_min"], color=COL_PUMP_BASE, lw=1.05, alpha=0.7, label="baseline pump rate")
    ax_pump.plot(cand["t_min"], cand["pump_rate_m3_min"], color=COL_PUMP_CAND, lw=1.05, alpha=0.7, label="candidate pump rate")
    ax_cum = ax_pump.twinx()
    ax_cum.plot(base["t_min"], base["pump_cum_m3"], color=COL_PUMP_BASE, lw=1.7, ls="--", label="baseline cumulative")
    ax_cum.plot(cand["t_min"], cand["pump_cum_m3"], color=COL_PUMP_CAND, lw=1.7, ls="--", label="candidate cumulative")
    ax_pump.set_ylabel("pump rate (m3/min)")
    ax_cum.set_ylabel("cum. pump (m3)")
    lines, labels = ax_pump.get_legend_handles_labels()
    lines2, labels2 = ax_cum.get_legend_handles_labels()
    ax_pump.legend(lines + lines2, labels + labels2, loc="upper left", ncol=2, fontsize=8)
    _style(ax_pump)
    ax_cum.spines["top"].set_visible(False)

    state_colors = {
        "high_hold": "#d73027",
        "high_active": "#2b8cbe",
        "high_other": "#fdae61",
        "safe_or_low": "#bdbdbd",
    }
    for y, state_df, label in [(1.0, base_state, "baseline"), (0.0, cand_state, "candidate")]:
        for state, group in state_df.groupby("state"):
            ax_state.scatter(
                group["t_min"],
                np.full(len(group), y),
                s=54,
                marker="s",
                color=state_colors.get(state, "#bdbdbd"),
                edgecolor="white",
                linewidth=0.45,
                label=state if y == 1.0 else None,
            )
        ax_state.text(-0.01, y, label, transform=ax_state.get_yaxis_transform(), ha="right", va="center", fontsize=9)
    ax_state.set_ylim(-0.55, 1.55)
    ax_state.set_yticks([])
    ax_state.set_xlabel("time (min)")
    ax_state.legend(loc="upper right", ncol=4, fontsize=8)
    _style(ax_state)

    title = case_id
    if metric_row is not None:
        verdict = metric_row.get("verdict", "")
        extra = float(metric_row.get("extra_pump_m3", 0.0))
        cont_min = float(metric_row.get("max_continuous_over_3deg_reduced_s", 0.0)) / 60.0
        p95 = float(metric_row.get("attitude_abs_p95_improvement_deg", 0.0))
        fallback = float(metric_row.get("fallback_ratio_delta", 0.0)) * 100.0
        title = (
            f"{case_id} | {verdict} | extra pump {extra:+.1f} m3 | "
            f"max >3 reduced {cont_min:+.1f} min | p95 improve {p95:+.2f} deg | "
            f"fallback delta {fallback:+.1f} pp"
        )
    fig.suptitle(title, fontsize=13)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
    out_path = out_dir / f"{safe_name}_attitude_recovery_compare.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_overview(metrics: pd.DataFrame, out_dir: Path, title: str) -> Path:
    df = metrics.sort_values("case_id", key=lambda s: s.map(_case_sort_key)).copy()
    labels = df["case_id"].astype(str).tolist()
    x = np.arange(len(df))
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 10.2), constrained_layout=True)
    axes = axes.ravel()

    def bars(ax, values, ylabel, zero=True, colors=None):
        vals = np.asarray(values, dtype=float)
        if colors is None:
            colors = np.where(vals >= 0, "#2b8cbe", "#c7532c")
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
        if zero:
            ax.axhline(0, color="#374151", lw=0.9)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        _style(ax)

    bars(
        axes[0],
        df["extra_pump_m3"],
        "extra pump (m3)",
        colors=np.where(df["extra_pump_m3"] <= 0, "#2b8cbe", "#c7532c"),
    )
    bars(
        axes[1],
        df["attitude_abs_p95_improvement_deg"],
        "attitude p95 improvement (deg)",
        colors=np.where(df["attitude_abs_p95_improvement_deg"] >= 0.2, "#2b8cbe", "#bdbdbd"),
    )
    bars(
        axes[2],
        df["time_over_3deg_reduced_s"] / 60.0,
        "time >3 deg reduced (min)",
        colors=np.where(df["time_over_3deg_reduced_s"] > 0, "#2b8cbe", "#c7532c"),
    )
    bars(
        axes[3],
        df["max_continuous_over_3deg_reduced_s"] / 60.0,
        "max continuous >3 reduced (min)",
        colors=np.where(df["max_continuous_over_3deg_reduced_s"] > 0, "#2b8cbe", "#c7532c"),
    )
    bars(
        axes[4],
        df["high_hold_bucket_reduced"],
        "high-hold buckets reduced",
        colors=np.where(df["high_hold_bucket_reduced"] > 0, "#2b8cbe", "#bdbdbd"),
    )
    bars(
        axes[5],
        df["fallback_ratio_delta"] * 100.0,
        "fallback delta (pp)",
        colors=np.where(df["fallback_ratio_delta"] <= 0, "#2b8cbe", "#c7532c"),
    )
    for ax, verdicts in [(axes[0], df["verdict"])]:
        for xi, verdict in zip(x, verdicts):
            ax.text(xi, ax.get_ylim()[1], str(verdict), ha="center", va="bottom", fontsize=8, rotation=35)
    fig.suptitle(title, fontsize=14)
    out_path = out_dir / "attitude_recovery_metric_overview.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.metrics_csv)
    base_cases = _discover(baseline_dir)
    cand_cases = _discover(candidate_dir)
    metric_by_norm = {_norm_case(row["case_id"]): row for _, row in metrics.iterrows()}
    common = sorted(set(base_cases) & set(cand_cases), key=_case_sort_key)
    if not common:
        raise SystemExit("no common cases found")

    overview = plot_overview(metrics, out_dir, f"Attitude recovery metrics | {candidate_dir.name}")
    paths = [overview]
    for norm in common:
        case_id, base_ts, base_log = base_cases[norm]
        cand_case_id, cand_ts, cand_log = cand_cases[norm]
        row = metric_by_norm.get(norm)
        paths.append(
            plot_case(
                cand_case_id or case_id,
                base_ts,
                cand_ts,
                base_log,
                cand_log,
                row,
                out_dir,
                float(args.attention_deg),
                float(args.high_deg),
                float(args.bucket_s),
            )
        )

    index_path = out_dir / "figure_index.md"
    lines = ["# Attitude Recovery Figures", ""]
    for path in paths:
        lines.append(f"- `{path}`")
    index_path.write_text("\n".join(lines) + "\n")
    print(index_path)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
