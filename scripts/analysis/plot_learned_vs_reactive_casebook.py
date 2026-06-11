#!/usr/bin/env python3
"""Plot learned planner against the pure reactive_current baseline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required; run with the project .venv or install it there"
        ) from exc
    return plt


def _read_summary(folder: Path) -> list[dict[str, str]]:
    path = folder / "casebook_summary.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_case_file(folder: Path, subdir: str, case_id: str, suffix: str) -> Path | None:
    files = sorted((folder / subdir).glob(f"{case_id}_*_{suffix}.csv"))
    return files[0] if files else None


def _q(values: np.ndarray, p: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values.astype(float), p))


def _last_window(df: pd.DataFrame, seconds: float) -> pd.DataFrame:
    if df.empty or "t_s" not in df:
        return df.iloc[0:0]
    t = df["t_s"].to_numpy(dtype=float)
    start = max(0.0, float(np.nanmax(t)) - float(seconds) + 1.0)
    return df.loc[df["t_s"].astype(float) >= start]


def _pump_cum(df: pd.DataFrame) -> np.ndarray:
    if df.empty or "pump_total_rate_m3_min" not in df:
        return np.zeros(len(df), dtype=float)
    rate = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    if "t_s" in df and len(df) > 1:
        t = df["t_s"].to_numpy(dtype=float)
        dt = np.diff(t, prepend=t[0])
        positive = dt[dt > 0]
        default_dt = float(np.median(positive)) if positive.size else 1.0
        dt[dt <= 0] = default_dt
    else:
        dt = np.ones(len(rate), dtype=float)
    return np.cumsum(rate * dt / 60.0)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "pump_m3": float("nan"),
            "pitch_p95": float("nan"),
            "roll_p95": float("nan"),
            "pitch_last60_p95": float("nan"),
            "roll_last60_p95": float("nan"),
            "fallback_ratio": float("nan"),
        }
    last60 = _last_window(df, 3600.0)
    fallback = (
        df["preview_primary_safety_fallback"].to_numpy(dtype=float) > 0.5
        if "preview_primary_safety_fallback" in df
        else np.zeros(len(df), dtype=bool)
    )
    return {
        "pump_m3": float(_pump_cum(df)[-1]) if len(df) else 0.0,
        "pitch_p95": _q(np.abs(df["pitch_deg"].to_numpy(dtype=float)), 0.95),
        "roll_p95": _q(np.abs(df["roll_deg"].to_numpy(dtype=float)), 0.95),
        "pitch_last60_p95": _q(np.abs(last60["pitch_deg"].to_numpy(dtype=float)), 0.95),
        "roll_last60_p95": _q(np.abs(last60["roll_deg"].to_numpy(dtype=float)), 0.95),
        "fallback_ratio": float(np.mean(fallback)) if fallback.size else 0.0,
    }


def _draw_action_band(ax, df: pd.DataFrame, label: str) -> None:
    if df.empty or "t_s" not in df:
        return
    t = df["t_s"].to_numpy(dtype=float) / 60.0
    actions = (
        df["preview_primary_action"].astype(str).to_numpy()
        if "preview_primary_action" in df
        else np.array([""] * len(df), dtype=object)
    )
    fallback = (
        df["preview_primary_safety_fallback"].to_numpy(dtype=float) > 0.5
        if "preview_primary_safety_fallback" in df
        else np.zeros(len(df), dtype=bool)
    )
    colors = {
        "hold": "#dbe4ec",
        "pump_saving": "#f6ad65",
        "active_small": "#6aaed6",
        "active_medium": "#2868a8",
        "active_reverse_small": "#9467bd",
    }
    y0 = 0.15 if label == "reactive" else 0.58
    height = 0.30
    start = 0
    for i in range(1, len(actions) + 1):
        if i == len(actions) or actions[i] != actions[start]:
            action = str(actions[start])
            x0 = float(t[start])
            x1 = float(t[i - 1] if i - 1 < len(t) else t[-1])
            if i < len(t):
                x1 = float(t[i])
            ax.axvspan(x0, x1, ymin=y0, ymax=y0 + height, color=colors.get(action, "#eeeeee"), alpha=0.78, lw=0)
            start = i
    fb = fallback.astype(bool)
    if fb.any():
        start = None
        for i, val in enumerate(fb):
            if val and start is None:
                start = i
            if start is not None and ((not val) or i == len(fb) - 1):
                end = i if not val else i + 1
                ax.axvspan(
                    float(t[start]),
                    float(t[min(end, len(t) - 1)]),
                    ymin=y0,
                    ymax=y0 + height,
                    color="#b91c1c",
                    alpha=0.75,
                    lw=0,
                )
                start = None
    ax.text(-0.01, y0 + 0.5 * height, label, transform=ax.transAxes, ha="right", va="center", fontsize=8)


def _tank_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in ("tank1_kg", "tank2_kg", "tank3_kg") if c in df]
    if not cols:
        return np.zeros((0, 0), dtype=float), []
    data = df[cols].to_numpy(dtype=float)
    # Express each tank as percentage of its own observed range so all three are visible.
    mins = np.nanmin(data, axis=0)
    spans = np.maximum(np.nanmax(data, axis=0) - mins, 1.0)
    scaled = (data - mins) / spans
    return scaled.T, [c.replace("_kg", "") for c in cols]


def plot_case(case_id: str, learned_df: pd.DataFrame, reactive_df: pd.DataFrame,
              learned_log: pd.DataFrame | None, out_dir: Path) -> Path:
    plt = _require_pyplot()
    t_learned = learned_df["t_s"].to_numpy(dtype=float) / 60.0
    t_reactive = reactive_df["t_s"].to_numpy(dtype=float) / 60.0
    duration_min = max(float(np.nanmax(t_learned)), float(np.nanmax(t_reactive)))
    learned_cum = _pump_cum(learned_df)
    reactive_cum = _pump_cum(reactive_df)
    n = min(len(learned_cum), len(reactive_cum), len(t_learned))
    saving = reactive_cum[:n] - learned_cum[:n]

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(14.0, 14.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.72, 1.0, 1.0, 1.0, 0.82, 0.72, 0.92]},
    )
    ax = axes[0]
    ax.step(t_learned, learned_df["wind_speed"].to_numpy(dtype=float), where="post", color="#1f77b4", lw=1.3)
    ax.set_ylabel("Wind speed\n(m/s)")
    ax2 = ax.twinx()
    ax2.step(t_learned, learned_df["wind_dir_deg"].to_numpy(dtype=float), where="post", color="#6b7280", lw=1.0)
    ax2.set_ylabel("Wind dir\n(deg)")

    for ax, col, ylabel, lim in [
        (axes[1], "pitch_deg", "Pitch (deg)", 4.0),
        (axes[2], "roll_deg", "Roll (deg)", 4.0),
    ]:
        ax.plot(t_learned, learned_df[col].to_numpy(dtype=float), color="#1f77b4", lw=1.2, label="learned")
        ax.plot(t_reactive, reactive_df[col].to_numpy(dtype=float), color="#d85c27", lw=1.1, label="reactive_current")
        ax.axhline(0.0, color="#6b7280", lw=0.8, alpha=0.65)
        ax.axhline(lim, color="#b91c1c", lw=0.8, ls="--", alpha=0.55)
        ax.axhline(-lim, color="#b91c1c", lw=0.8, ls="--", alpha=0.55)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d0d7de", alpha=0.7, lw=0.7)
    axes[1].legend(loc="upper left", frameon=False, ncol=2)

    axes[3].plot(t_learned, learned_df["pump_total_rate_m3_min"].to_numpy(dtype=float), color="#1f77b4", lw=1.0, label="learned")
    axes[3].plot(t_reactive, reactive_df["pump_total_rate_m3_min"].to_numpy(dtype=float), color="#d85c27", lw=1.0, label="reactive_current")
    axes[3].set_ylabel("Pump rate\n(m3/min)")
    axes[3].grid(True, color="#d0d7de", alpha=0.7, lw=0.7)

    mat, labels = _tank_matrix(learned_df)
    if mat.size:
        axes[4].imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            extent=[float(t_learned[0]), float(t_learned[-1]), 0, len(labels)],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axes[4].set_yticks(np.arange(len(labels)) + 0.5)
        axes[4].set_yticklabels(labels)
    axes[4].set_ylabel("Learned\ntank level")

    _draw_action_band(axes[5], learned_df, "learned")
    _draw_action_band(axes[5], reactive_df, "reactive")
    axes[5].set_yticks([])
    axes[5].set_ylabel("Actions\n+ fallback")

    axes[6].plot(t_learned[:n], saving, color="#0f766e", lw=1.6)
    axes[6].axhline(0.0, color="#6b7280", lw=0.8)
    axes[6].fill_between(t_learned[:n], saving, 0.0, where=saving >= 0.0, color="#0f766e", alpha=0.18)
    axes[6].fill_between(t_learned[:n], saving, 0.0, where=saving < 0.0, color="#b91c1c", alpha=0.16)
    axes[6].set_ylabel("Cumulative\nsaving (m3)")
    axes[6].set_xlabel("Time (min)")
    axes[6].grid(True, color="#d0d7de", alpha=0.7, lw=0.7)
    axes[6].set_xlim(0.0, duration_min)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        for x in np.arange(20.0, duration_min + 1e-6, 20.0):
            ax.axvline(x, color="#9ca3af", lw=0.7, ls="--", alpha=0.35)

    lm = _metrics(learned_df)
    rm = _metrics(reactive_df)
    title = (
        f"{case_id}: learned vs reactive_current | "
        f"pump {lm['pump_m3']:.1f} vs {rm['pump_m3']:.1f} m3, "
        f"saving {rm['pump_m3'] - lm['pump_m3']:+.1f} m3"
    )
    fig.suptitle(title, fontsize=13)
    out_path = out_dir / f"{case_id}_learned_vs_reactive.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _case_ids(summary_rows: list[dict[str, str]]) -> list[str]:
    ids = [r.get("case_id", "") for r in summary_rows if r.get("case_id", "")]
    return sorted(dict.fromkeys(ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-dir", required=True)
    parser.add_argument("--reactive-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    learned_dir = (REPO_ROOT / args.learned_dir).resolve()
    reactive_dir = (REPO_ROOT / args.reactive_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    learned_summary = _read_summary(learned_dir)
    reactive_summary = _read_summary(reactive_dir)
    case_ids = sorted(set(_case_ids(learned_summary)) & set(_case_ids(reactive_summary)))
    rows: list[dict[str, Any]] = []
    figures: list[Path] = []
    for case_id in case_ids:
        learned_ts = _find_case_file(learned_dir, "timeseries", case_id, "prediction_primary_econ_timeseries")
        reactive_ts = _find_case_file(reactive_dir, "timeseries", case_id, "prediction_primary_econ_timeseries")
        learned_log_path = _find_case_file(learned_dir, "planner_logs", case_id, "prediction_primary_econ_planner_log")
        if learned_ts is None or reactive_ts is None:
            continue
        learned_df = pd.read_csv(learned_ts)
        reactive_df = pd.read_csv(reactive_ts)
        learned_log = pd.read_csv(learned_log_path) if learned_log_path is not None else None
        fig_path = plot_case(case_id, learned_df, reactive_df, learned_log, out_dir)
        figures.append(fig_path)
        lm = _metrics(learned_df)
        rm = _metrics(reactive_df)
        rows.append({
            "case_id": case_id,
            "learned_pump_m3": lm["pump_m3"],
            "reactive_pump_m3": rm["pump_m3"],
            "saving_m3": rm["pump_m3"] - lm["pump_m3"],
            "learned_pitch_last60_p95": lm["pitch_last60_p95"],
            "reactive_pitch_last60_p95": rm["pitch_last60_p95"],
            "learned_roll_last60_p95": lm["roll_last60_p95"],
            "reactive_roll_last60_p95": rm["roll_last60_p95"],
            "learned_fallback_ratio": lm["fallback_ratio"],
            "reactive_fallback_ratio": rm["fallback_ratio"],
            "figure": str(fig_path.relative_to(REPO_ROOT)),
        })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "learned_vs_reactive_summary.csv", index=False)
    print(f"wrote {out_dir / 'learned_vs_reactive_summary.csv'}")
    for fig in figures:
        print(f"wrote {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
