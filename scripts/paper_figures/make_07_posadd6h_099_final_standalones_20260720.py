#!/usr/bin/env python3
"""Render case 07_posadd6h_099 with the latest approved case-38 style."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
LATEST_STYLE_SCRIPT = (
    ROOT
    / "scripts/paper_figures/"
    "make_06_sel6h_38_rolling_process_final_20260720.py"
)
OUT = (
    ROOT
    / "outputs/paper_figures/20260720_07_posadd6h_099_rolling_process_final"
)
FIG_DIR = OUT / "standalone_figures"
QA_DIR = OUT / "qa"
MM_PER_INCH = 25.4


def load_latest_style_module():
    spec = importlib.util.spec_from_file_location(
        "case38_latest_style_for_case99_20260720",
        LATEST_STYLE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {LATEST_STYLE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


latest = load_latest_style_module()
base = latest.base
base.ACTION_LABELS = [
    "反向执行",
    "暂缓更新",
    "减弱执行",
    "常规执行",
    "加强执行",
]

CASE = base.CaseSpec(
    short_name="07_posadd6h_099_t180",
    case_name="07_posadd6h_099_2023-03-07_000000",
    batch_task="batch_2_paired/task_3",
    event_min=180.0,
    recommendation="大姿态暂缓更新候选",
    evidence_note=(
        "事件时预测辅助算法主导姿态约2.27°，既有目标已完成且冻结预测"
        "未显示风况继续增强，滚动策略由常规执行转为暂缓更新。"
    ),
)


def export(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def panel_caption(fig: plt.Figure, text: str, *, x: float) -> None:
    fig.text(x, 0.020, text, ha="center", va="bottom", fontsize=7.5)


def make_panel_a(data: dict) -> None:
    fig = plt.figure(figsize=(82.0 / MM_PER_INCH, 58.0 / MM_PER_INCH))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.05, 0.80],
        left=0.235,
        right=0.985,
        bottom=0.235,
        top=0.985,
        hspace=0.10,
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
    latest.plot_action_and_target(ax1, ax2, data)
    panel_caption(fig, "(a) 滚动执行策略与目标更新", x=0.61)
    export(fig, FIG_DIR / "07_posadd6h_099_panel_a_strategy_target")


def make_panel_b(data: dict) -> None:
    fig = plt.figure(figsize=(82.0 / MM_PER_INCH, 58.0 / MM_PER_INCH))
    grid = fig.add_gridspec(
        2,
        1,
        left=0.205,
        right=0.985,
        bottom=0.235,
        top=0.985,
        hspace=0.10,
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
    latest.plot_measured_wind(ax1, ax2, data)
    panel_caption(fig, "(b) 后续实测风况", x=0.595)
    export(fig, FIG_DIR / "07_posadd6h_099_panel_b_measured_wind")


def make_panel_c(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(82.0 / MM_PER_INCH, 58.0 / MM_PER_INCH))
    fig.subplots_adjust(
        left=0.205,
        right=0.985,
        bottom=0.235,
        top=0.985,
    )
    latest.plot_cumulative(ax, data)
    panel_caption(fig, "(c) 决策后累计泵量", x=0.595)
    export(fig, FIG_DIR / "07_posadd6h_099_panel_c_cumulative_pump")


def make_panel_d(data: dict) -> None:
    fig = plt.figure(figsize=(82.0 / MM_PER_INCH, 58.0 / MM_PER_INCH))
    grid = fig.add_gridspec(
        2,
        1,
        left=0.205,
        right=0.985,
        bottom=0.235,
        top=0.985,
        hspace=0.10,
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
    latest.plot_attitude(ax1, ax2, data)
    panel_caption(fig, "(d) 俯仰角与横摇角响应", x=0.595)
    export(fig, FIG_DIR / "07_posadd6h_099_panel_d_attitude")


def main() -> None:
    latest.configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    replay = base.Fino1ReplayDataset(base.DATASET_DIR, split="test")
    adapter = base.ForecastModelAdapter(
        base.MODEL_DIR,
        base.DATASET_DIR,
        device="cpu",
    )
    base.style_axis = latest.style_axis
    data = base.load_case(CASE, replay, adapter)
    make_panel_a(data)
    make_panel_b(data)
    make_panel_c(data)
    make_panel_d(data)
    (QA_DIR / "wind_display_method.json").write_text(
        json.dumps(
            {
                "samples": "audited measured values at 10 min intervals",
                "markers": "open circles show every original sample",
                "connecting_curve": (
                    "shape-preserving PCHIP interpolation for display only"
                ),
                "integrity": (
                    "the curve passes through every sample, introduces no "
                    "overshoot, and is not used in any calculation"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(FIG_DIR)


if __name__ == "__main__":
    main()
