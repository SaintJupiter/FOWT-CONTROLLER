#!/usr/bin/env python3
"""Final four-panel rolling-control process figure for case 06_sel6h_38.

This figure deliberately separates the evidence roles:

* panel (a) shows the prediction-assisted algorithm's executed rolling action
  and the paired target-update amplitudes;
* panel (b) shows measured wind only (no forecast-accuracy claim);
* panels (c) and (d) compare the prediction-assisted and posture-feedback
  algorithms over the same paired window.

All numerical data are loaded by the audited frozen-result workflow reused
from ``make_06_sel6h_38_abcd_high_attitude_20260720.py``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = (
    ROOT
    / "scripts/paper_figures/make_06_sel6h_38_abcd_high_attitude_20260720.py"
)
OUT = (
    ROOT
    / "outputs/paper_figures/20260720_chapter3_3_38_rolling_process_final"
)
FIG_DIR = OUT / "figures"
DATA_DIR = OUT / "source_data"
QA_DIR = OUT / "qa"
MM_PER_INCH = 25.4


def load_source_module():
    spec = importlib.util.spec_from_file_location(
        "case38_high_attitude_source_20260720",
        SOURCE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source = load_source_module()
base = source.base
CASE = source.CASE

# Keep the algorithm encoding consistent across the whole manuscript:
# prediction-assisted algorithm = black solid line;
# traditional posture-feedback algorithm = gray dashed line.
PRED_LINESTYLE = "-"
POSTURE_LINESTYLE = (0, (4.0, 2.0))


def configure_style() -> None:
    source.configure_style()
    plt.rcParams.update(
        {
            "legend.fontsize": 6.5,
            "legend.borderpad": 0.22,
            "legend.labelspacing": 0.20,
            "legend.handlelength": 2.0,
            "legend.handletextpad": 0.35,
        }
    )


def style_axis(
    ax: plt.Axes,
    *,
    show_x: bool,
    event_line: bool = True,
) -> None:
    ax.set_xlim(base.REL_START, base.REL_END)
    ax.set_xticks(
        base.REL_TICKS,
        [str(int(v + source.TIME_OFFSET_MIN)) for v in base.REL_TICKS],
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(base.FRAME)
        spine.set_linewidth(0.58)
    if event_line:
        ax.axvline(
            0.0,
            color=base.MID,
            linewidth=0.50,
            linestyle=(0, (2.0, 1.8)),
            zorder=1,
        )
    if show_x:
        ax.set_xlabel("片段时间 / min", labelpad=1.0)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def box_legend(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_visible(True)
    frame.set_facecolor("white")
    frame.set_edgecolor(base.FRAME)
    frame.set_linewidth(0.45)
    frame.set_boxstyle("square", pad=0.14)


def plot_action_and_target(
    ax_action: plt.Axes,
    ax_update: plt.Axes,
    data: dict,
) -> None:
    planner = data["planner"]
    updates = data["updates"]
    ax_action.plot(
        planner.relative_time_min,
        planner.action_level,
        color=base.BLACK,
        linewidth=0.84,
        linestyle=PRED_LINESTYLE,
        marker="s",
        markersize=2.20,
        markerfacecolor=base.BLACK,
        markeredgecolor=base.BLACK,
        markeredgewidth=0.28,
        zorder=4,
    )
    style_axis(ax_action, show_x=False)
    ax_action.set_ylim(-0.35, 4.35)
    action_labels = [
        "暂缓更新" if label == "停止执行" else label
        for label in base.ACTION_LABELS
    ]
    ax_action.set_yticks(base.ACTION_TICKS, action_labels)
    ax_action.set_ylabel("预测辅助执行策略", labelpad=3.0)

    ax_update.plot(
        updates.relative_time_min,
        updates.prediction_target_update_m3,
        color=base.BLACK,
        linewidth=0.84,
        linestyle=PRED_LINESTYLE,
        marker="s",
        markersize=2.05,
        markerfacecolor=base.BLACK,
        markeredgewidth=0.28,
        label="预测辅助算法",
    )
    ax_update.plot(
        updates.relative_time_min,
        updates.posture_target_update_m3,
        color=base.MID,
        linewidth=0.84,
        linestyle=POSTURE_LINESTYLE,
        marker="o",
        markersize=2.15,
        markerfacecolor="white",
        markeredgecolor=base.MID,
        markeredgewidth=0.48,
        label="传统姿态反馈算法",
    )
    style_axis(ax_update, show_x=True)
    update_max = float(
        max(
            updates.prediction_target_update_m3.max(),
            updates.posture_target_update_m3.max(),
        )
    )
    ax_update.set_ylim(-0.03, update_max * 1.48)
    ax_update.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_update.set_ylabel("目标更新幅值 / m³", labelpad=3.0)
    ax_update.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        borderaxespad=0.0,
    )
    box_legend(ax_update)


def plot_measured_wind(
    ax_speed: plt.Axes,
    ax_direction: plt.Axes,
    data: dict,
) -> None:
    measured = data["measured"]

    def plot_display_curve(ax: plt.Axes, values) -> None:
        """Draw a shape-preserving display curve through audited samples."""
        x = measured.relative_time_min.to_numpy(float)
        y = np.asarray(values, dtype=float)
        dense_x = np.linspace(x[0], x[-1], (len(x) - 1) * 40 + 1)
        dense_y = PchipInterpolator(x, y)(dense_x)
        ax.plot(
            dense_x,
            dense_y,
            color=base.BLACK,
            linewidth=0.88,
            zorder=2,
        )
        ax.plot(
            x,
            y,
            linestyle="none",
            color=base.BLACK,
            marker="o",
            markersize=2.10,
            markerfacecolor="white",
            markeredgecolor=base.BLACK,
            markeredgewidth=0.46,
            zorder=3,
        )

    plot_display_curve(
        ax_speed,
        measured.measured_wind_speed_m_s,
    )
    style_axis(ax_speed, show_x=False)
    ax_speed.set_ylabel("风速 / (m/s)")
    ax_speed.yaxis.set_major_locator(MaxNLocator(nbins=4))

    plot_display_curve(
        ax_direction,
        measured.measured_wind_direction_deg,
    )
    style_axis(ax_direction, show_x=True)
    ax_direction.set_ylabel("风向 / °")
    ax_direction.yaxis.set_major_locator(MaxNLocator(nbins=4))


def plot_cumulative(ax: plt.Axes, data: dict) -> None:
    pred = data["pred_cumulative"].copy()
    posture = data["posture_cumulative"].copy()
    pred0 = float(
        pred.loc[
            pred.relative_time_min.ge(0.0),
            "window_cumulative_pump_m3",
        ].iloc[0]
    )
    posture0 = float(
        posture.loc[
            posture.relative_time_min.ge(0.0),
            "window_cumulative_pump_m3",
        ].iloc[0]
    )
    pred = pred[pred.relative_time_min.ge(0.0)].copy()
    posture = posture[posture.relative_time_min.ge(0.0)].copy()
    pred["post_decision_cumulative_m3"] = (
        pred.window_cumulative_pump_m3 - pred0
    )
    posture["post_decision_cumulative_m3"] = (
        posture.window_cumulative_pump_m3 - posture0
    )
    ax.plot(
        pred.relative_time_min,
        pred.post_decision_cumulative_m3,
        color=base.BLACK,
        linewidth=0.92,
        linestyle=PRED_LINESTYLE,
        label="预测辅助算法",
    )
    ax.plot(
        posture.relative_time_min,
        posture.post_decision_cumulative_m3,
        color=base.MID,
        linewidth=0.92,
        linestyle=POSTURE_LINESTYLE,
        label="传统姿态反馈算法",
    )
    style_axis(ax, show_x=True)
    ax.set_xlim(0.0, base.REL_END)
    ax.set_xticks(
        [0.0, 20.0, 40.0, 60.0],
        ["20", "40", "60", "80"],
    )
    cumulative_max = float(
        max(
            pred.post_decision_cumulative_m3.max(),
            posture.post_decision_cumulative_m3.max(),
        )
    )
    ax.set_ylim(-1.0, cumulative_max * 1.30)
    ax.set_ylabel("决策后累计泵量 / m³")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        borderaxespad=0.0,
    )
    box_legend(ax)


def plot_attitude(
    ax_pitch: plt.Axes,
    ax_roll: plt.Axes,
    data: dict,
) -> None:
    pred = data["pred_attitude"]
    posture = data["posture_attitude"]
    bound = max(3.5, base.shared_attitude_limit(data))
    for ax, variable, ylabel in [
        (ax_pitch, "pitch_deg", "俯仰角 / °"),
        (ax_roll, "roll_deg", "横摇角 / °"),
    ]:
        ax.plot(
            pred.relative_time_min,
            pred[variable],
            color=base.BLACK,
            linewidth=0.86,
            linestyle=PRED_LINESTYLE,
            label="预测辅助算法",
        )
        ax.plot(
            posture.relative_time_min,
            posture[variable],
            color=base.MID,
            linewidth=0.86,
            linestyle=POSTURE_LINESTYLE,
            label="传统姿态反馈算法",
        )
        ax.axhline(0.0, color=base.LIGHT, linewidth=0.42, zorder=0)
        ax.set_ylim(-bound, bound)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    style_axis(ax_pitch, show_x=False)
    style_axis(ax_roll, show_x=True)
    ax_roll.set_xlabel("时间 / min", labelpad=1.0)
    ax_pitch.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        borderaxespad=0.0,
    )
    box_legend(ax_pitch)


def make_composite(data: dict) -> Path:
    fig = plt.figure(figsize=(170.0 / MM_PER_INCH, 114.0 / MM_PER_INCH))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.973,
        bottom=0.108,
        top=0.950,
        hspace=0.43,
        wspace=0.31,
    )
    a_grid = outer[0, 0].subgridspec(
        2, 1, height_ratios=[1.05, 0.80], hspace=0.10
    )
    ax_a1 = fig.add_subplot(a_grid[0, 0])
    ax_a2 = fig.add_subplot(a_grid[1, 0], sharex=ax_a1)
    b_grid = outer[0, 1].subgridspec(2, 1, hspace=0.10)
    ax_b1 = fig.add_subplot(b_grid[0, 0])
    ax_b2 = fig.add_subplot(b_grid[1, 0], sharex=ax_b1)
    ax_c = fig.add_subplot(outer[1, 0])
    d_grid = outer[1, 1].subgridspec(2, 1, hspace=0.10)
    ax_d1 = fig.add_subplot(d_grid[0, 0])
    ax_d2 = fig.add_subplot(d_grid[1, 0], sharex=ax_d1)

    plot_action_and_target(ax_a1, ax_a2, data)
    plot_measured_wind(ax_b1, ax_b2, data)
    plot_cumulative(ax_c, data)
    plot_attitude(ax_d1, ax_d2, data)

    captions = [
        (ax_a2, "(a) 滚动执行策略与目标更新"),
        (ax_b2, "(b) 后续实测风况"),
        (ax_c, "(c) 决策后累计泵量"),
        (ax_d2, "(d) 俯仰角与横摇角响应"),
    ]
    for ax, text in captions:
        box = ax.get_position()
        fig.text(
            (box.x0 + box.x1) / 2.0,
            box.y0 - 0.064,
            text,
            ha="center",
            va="top",
            fontsize=7.5,
        )

    stem = FIG_DIR / f"{CASE.short_name}_rolling_process_final"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    return stem


def main() -> None:
    configure_style()
    for directory in [FIG_DIR, DATA_DIR, QA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    replay = base.Fino1ReplayDataset(base.DATASET_DIR, split="test")
    adapter = base.ForecastModelAdapter(
        base.MODEL_DIR,
        base.DATASET_DIR,
        device="cpu",
    )
    base.style_axis = style_axis
    data = base.load_case(CASE, replay, adapter)
    source.save_source_data(data)
    stem = make_composite(data)

    audit = data["audit"]
    pred = data["pred_cumulative"]
    posture = data["posture_cumulative"]
    pred_at_decision = float(
        pred.loc[
            pred.relative_time_min.ge(0.0),
            "window_cumulative_pump_m3",
        ].iloc[0]
    )
    posture_at_decision = float(
        posture.loc[
            posture.relative_time_min.ge(0.0),
            "window_cumulative_pump_m3",
        ].iloc[0]
    )
    pred_60 = float(
        pred.window_cumulative_pump_m3.iloc[-1] - pred_at_decision
    )
    posture_60 = float(
        posture.window_cumulative_pump_m3.iloc[-1] - posture_at_decision
    )
    audit["post_decision_pump_60min"] = {
        "prediction_m3": pred_60,
        "posture_m3": posture_60,
        "difference_m3": posture_60 - pred_60,
        "reduction_percent": (posture_60 - pred_60) / posture_60 * 100.0,
    }
    audit["figure_claim_boundary"] = {
        "panel_b": "measured wind only; no forecast-accuracy claim",
        "panel_b_display": (
            "Open circles are the audited 10 min samples. The connecting "
            "curve is shape-preserving PCHIP interpolation for display only; "
            "it passes through every sample and is not used in calculations."
        ),
        "action_trace": "each point is a rolling replan at that time",
        "attitude": (
            "local P95 comparison only; the case does not demonstrate "
            "reduced duration above 2 degrees"
        ),
        "population_inference": "not permitted from this representative case",
    }
    (QA_DIR / f"{CASE.short_name}_rolling_process_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(stem)


if __name__ == "__main__":
    main()
