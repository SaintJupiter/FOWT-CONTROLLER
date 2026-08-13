from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fowt_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fowt_xdg_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "paper_figures_current" / "results"


def pick_font() -> str:
    preferred = [
        "Songti SC",
        "STSong",
        "SimSun",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Hiragino Sans GB",
        "Arial Unicode MS",
        "SimHei",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    return "DejaVu Sans"


FONT = pick_font()
mpl.rcParams.update(
    {
        "font.family": ["Times New Roman", FONT, "DejaVu Sans", "sans-serif"],
        "font.sans-serif": ["Times New Roman", FONT, "DejaVu Sans", "sans-serif"],
        "font.serif": ["Times New Roman", FONT, "DejaVu Serif", "serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,
        "font.size": 7.4,
        "axes.labelsize": 7.8,
        "axes.titlesize": 9.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.2,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "action": "#0077BB",
    "boundary": "#EE7733",
    "background": "#999999",
    "saving": "#009988",
    "baseline": "#6B7280",
    "supervised": "#0077BB",
    "warning": "#CC3311",
    "neutral": "#374151",
    "grid": "#E5E7EB",
}

RUN_ROOT = ROOT / "outputs" / "wind_prediction" / "selector_mixed_6h_limit20_v1"
FINAL_PER_CASE = RUN_ROOT / "selector_gated_summary_101case_6h" / "selector_gated_per_case.csv"
FINAL_EXTENDED_PER_CASE = RUN_ROOT / "selector_gated_summary_101case_6h" / "selector_gated_extended_per_case.csv"
CASEBOOK_SUMMARY = RUN_ROOT / "blind_d1_engineered_101case_6h" / "casebook_summary.csv"
TS_ROOT = RUN_ROOT / "blind_d1_engineered_101case_6h" / "timeseries"
V6_TABLES = (
    ROOT
    / "outputs"
    / "paper_figures_current"
    / "overnight_evidence"
    / "control_validation_redesign"
    / "v6_final_polished"
    / "tables"
)
V7_OUT = ROOT / "outputs" / "paper_figures_current" / "validation_expansion_v7"
V7_FIGURES = V7_OUT / "figures"
V7_TABLES = V7_OUT / "tables"

ROLE_CN = {
    "positive_allow": "具有调节空间的窗口",
    "negative_abstain": "风险边界窗口",
    "background_abstain": "低扰动背景窗口",
}

V7_COLORS = {
    "baseline": "#74808D",
    "predictive": "#4B97B8",
    "predictive_dark": "#155E83",
    "benefit": "#7EA692",
    "benefit_soft": "#DCEBE4",
    "penalty": "#B98272",
    "penalty_soft": "#EFE0DA",
    "neutral": "#1F2933",
    "muted": "#5F6F82",
    "grid": "#E8EDF2",
    "axis": "#B9C3CF",
    "light": "#D7DEE7",
}


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def save_publication(fig: plt.Figure, stem: str, dpi: int = 600) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=dpi)
    fig.savefig(OUT / f"{stem}.tiff", dpi=dpi)
    plt.close(fig)


def mixed_summary_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "预测可行动窗口",
                "short": "可行动",
                "n": 50,
                "closed": 54024.40,
                "supervised": 35465.69,
                "saving_pct": 34.35,
                "dt5": 26,
                "dt75": 4,
                "dt10": 0,
                "fallback": 0,
                "color": COLORS["action"],
            },
            {
                "name": "预警边界窗口",
                "short": "边界",
                "n": 31,
                "closed": 28452.32,
                "supervised": 28452.32,
                "saving_pct": 0.0,
                "dt5": 0,
                "dt75": 0,
                "dt10": 0,
                "fallback": 0,
                "color": COLORS["boundary"],
            },
            {
                "name": "低机会背景窗口",
                "short": "背景",
                "n": 20,
                "closed": 67.20,
                "supervised": 67.20,
                "saving_pct": 0.0,
                "dt5": 0,
                "dt75": 0,
                "dt10": 0,
                "fallback": 0,
                "color": COLORS["background"],
            },
        ]
    )


def annotate_segment(ax, left, width, y, label, value_label, min_width=3.5):
    if width >= min_width:
        ax.text(
            left + width / 2,
            y,
            f"{label}\n{value_label}",
            ha="center",
            va="center",
            color="white" if label != "背景" else "#111827",
            fontsize=8,
            linespacing=1.2,
        )
    else:
        ax.text(
            left + width + 1.0,
            y,
            f"{label} {value_label}",
            ha="left",
            va="center",
            color="#111827",
            fontsize=8,
        )


def figure_mixed_contribution() -> None:
    df = mixed_summary_data()
    total_n = float(df["n"].sum())
    total_closed = float(df["closed"].sum())
    total_supervised = float(df["supervised"].sum())
    total_saved = total_closed - total_supervised
    total_saving_pct = total_saved / total_closed * 100.0

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.45], width_ratios=[1.05, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # A: window counts and baseline opportunity share.
    for y, values, title in [
        (1.0, df["n"].to_numpy(dtype=float) / total_n * 100.0, "窗口数量占比"),
        (0.0, df["closed"].to_numpy(dtype=float) / total_closed * 100.0, "原闭环泵耗占比"),
    ]:
        left = 0.0
        for (_, row), width in zip(df.iterrows(), values):
            ax_a.barh(y, width, left=left, color=row["color"], height=0.34)
            value_label = f"{row['n']}个" if y == 1.0 else f"{width:.1f}%"
            annotate_segment(ax_a, left, width, y, row["short"], value_label)
            left += width
        ax_a.text(-3, y, title, ha="right", va="center", fontsize=8.5, color="#111827")
    ax_a.set_xlim(-22, 103)
    ax_a.set_ylim(-0.55, 1.55)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_xlabel("占比（%）")
    ax_a.set_yticks([])
    ax_a.set_title("A  样本构成与泵耗机会", loc="left", fontweight="bold")
    ax_a.grid(axis="x", color=COLORS["grid"], linewidth=0.8)

    # B: concise comparison of total pump work.
    labels = ["原闭环", "监督策略"]
    vals = [total_closed, total_supervised]
    bars = ax_b.bar(labels, vals, color=[COLORS["baseline"], COLORS["supervised"]], width=0.55)
    for bar, val in zip(bars, vals):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            val + total_closed * 0.015,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_b.annotate(
        f"降低 {total_saving_pct:.2f}%\n节省 {total_saved:,.0f} 立方米",
        xy=(1, total_supervised),
        xytext=(0.28, total_closed * 0.58),
        arrowprops=dict(arrowstyle="->", color=COLORS["saving"], lw=1.2),
        color=COLORS["saving"],
        ha="left",
        va="center",
        fontsize=8.5,
    )
    ax_b.set_ylabel("累计泵耗（立方米）")
    ax_b.set_ylim(0, total_closed * 1.18)
    ax_b.set_title("B  总体泵耗变化", loc="left", fontweight="bold")
    ax_b.grid(axis="y", color=COLORS["grid"], linewidth=0.8)

    # C: waterfall contribution.
    x = np.arange(4)
    starts = [0, 0, 0, 0]
    heights = [total_closed, -total_saved, 0, total_supervised]
    colors = [COLORS["baseline"], COLORS["saving"], "#D1D5DB", COLORS["supervised"]]
    labels = ["原闭环总泵耗", "可行动窗口节省", "边界/背景节省", "监督策略泵耗"]
    current = total_closed
    ax_c.bar(0, total_closed, color=colors[0], width=0.58)
    ax_c.bar(1, -total_saved, bottom=current, color=colors[1], width=0.58)
    ax_c.bar(2, 0.001, bottom=current - total_saved, color=colors[2], width=0.58)
    ax_c.bar(3, total_supervised, color=colors[3], width=0.58)
    ax_c.plot([0.29, 0.71], [total_closed, total_closed], color="#9CA3AF", lw=0.8)
    ax_c.plot([1.29, 1.71], [total_supervised, total_supervised], color="#9CA3AF", lw=0.8)
    ax_c.text(0, total_closed + 2500, f"{total_closed:,.0f}", ha="center", fontsize=8)
    ax_c.text(1, total_closed - total_saved / 2, f"-{total_saved:,.0f}", ha="center", va="center", color="white", fontsize=8)
    ax_c.text(2, total_supervised + 1600, "0", ha="center", fontsize=8)
    ax_c.text(3, total_supervised + 2500, f"{total_supervised:,.0f}", ha="center", fontsize=8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=18, ha="right")
    ax_c.set_ylabel("累计泵耗（立方米）")
    ax_c.set_ylim(0, total_closed * 1.18)
    ax_c.set_title("C  泵耗降低贡献分解", loc="left", fontweight="bold")
    ax_c.grid(axis="y", color=COLORS["grid"], linewidth=0.8)

    # D: attitude/safety ledger.
    ledger = pd.DataFrame(
        {
            "metric": ["T>5°", "T>7.5°", "T>10°", "安全回退"],
            "value": [26, 4, 0, 0],
            "unit": ["秒", "秒", "秒", "个"],
        }
    )
    y = np.arange(len(ledger))[::-1]
    ax_d.axvline(0, color="#111827", lw=0.9)
    ax_d.hlines(y, 0, ledger["value"], color="#CBD5E1", lw=2)
    ax_d.scatter(
        ledger["value"],
        y,
        s=52,
        color=[COLORS["warning"], COLORS["boundary"], COLORS["neutral"], COLORS["neutral"]],
        zorder=3,
    )
    for yi, (_, row) in zip(y, ledger.iterrows()):
        ax_d.text(row["value"] + 1.5, yi, f"{row['value']:+.0f} {row['unit']}" if row["metric"] != "安全回退" else f"{row['value']:.0f} 个", va="center", fontsize=8)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(ledger["metric"])
    ax_d.set_xlabel("相对原闭环变化")
    ax_d.set_xlim(-5, 38)
    ax_d.set_title("D  姿态与安全代价账本", loc="left", fontweight="bold")
    ax_d.grid(axis="x", color=COLORS["grid"], linewidth=0.8)

    save(fig, "fig_mixed_contribution_advanced")


def read_mixed_per_case() -> pd.DataFrame:
    path = ROOT / "outputs/wind_prediction/selector_mixed_6h_limit20_v1/selector_gated_summary_101case_6h/selector_gated_per_case.csv"
    df = pd.read_csv(path)
    label_map = {
        "positive_allow": "预测可行动",
        "negative_abstain": "预警边界",
        "background_abstain": "低机会背景",
    }
    df["role_cn"] = df["validation_role"].map(label_map).fillna(df["validation_role"])
    df["saved_m3"] = df["closed_pump_m3"] - df["gated_pump_m3"]
    df["d_time_over_10_s"] = 0.0
    return df


def jitter_positions(groups: pd.Series, order: list[str], seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = {g: i for i, g in enumerate(order)}
    return np.array([base[g] + rng.uniform(-0.18, 0.18) for g in groups])


def figure_case_distribution() -> None:
    df = read_mixed_per_case()
    order = ["预测可行动", "预警边界", "低机会背景"]
    colors = {"预测可行动": COLORS["action"], "预警边界": COLORS["boundary"], "低机会背景": COLORS["background"]}
    x = jitter_positions(df["role_cn"], order)
    sizes = 24 + 110 * np.sqrt(np.clip(df["closed_pump_m3"], 0, None) / max(df["closed_pump_m3"].max(), 1))
    edge = np.where(df["gated_d_time_over_7p5_s"] > 0, COLORS["warning"], "#FFFFFF")

    fig, ax = plt.subplots(figsize=(6.3, 3.45), constrained_layout=True)
    # Soft distribution envelope for each group, then individual cases.
    data_by_group = [df.loc[df["role_cn"] == g, "gated_saving_pct"].to_numpy() for g in order]
    bp = ax.boxplot(
        data_by_group,
        positions=np.arange(len(order)),
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#111827", linewidth=1.0),
        boxprops=dict(linewidth=0.8, color="#6B7280"),
        whiskerprops=dict(linewidth=0.8, color="#6B7280"),
        capprops=dict(linewidth=0.8, color="#6B7280"),
    )
    for patch, role in zip(bp["boxes"], order):
        patch.set_facecolor(colors[role])
        patch.set_alpha(0.16)
    for role in order:
        m = df["role_cn"] == role
        ax.scatter(
            x[m],
            df.loc[m, "gated_saving_pct"],
            s=np.minimum(sizes[m] * 0.72, 120),
            color=colors[role],
            alpha=0.62,
            edgecolor=edge[m],
            linewidth=0.9,
            label=role,
        )
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{g}\n(n={int((df['role_cn']==g).sum())})" for g in order])
    ax.set_ylabel("单窗口泵耗降低比例（%）")
    ax.set_title("101个6小时窗口的逐案泵耗降低分布", loc="left", fontweight="bold", fontsize=10)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylim(-5, max(42, df["gated_saving_pct"].max() + 5))
    ax.text(
        0.02,
        0.95,
        "箱体表示组内分布；点面积表示原闭环泵耗规模；红色描边表示 T>7.5° 有增加",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#374151",
    )
    save(fig, "fig_101case_pump_saving_distribution")


def figure_positive_tradeoff() -> None:
    path = ROOT / "outputs/wind_prediction/selector_positive_add40_6h_v1/combined_positive_summary_170case_6h/positive_combined_by_stratum.csv"
    df = pd.read_csv(path)
    name_map = {
        "p2_neutral_headroom": "姿态余量型",
        "c3_gusty_oscillatory": "阵风振荡型",
        "w1_stable_direction_event": "稳定风向事件型",
    }
    df["name"] = df["selector_stratum"].map(name_map)
    df = df.set_index("name").loc[["姿态余量型", "阵风振荡型", "稳定风向事件型"]].reset_index()
    colors = [COLORS["action"], "#8B5CF6", "#009988"]

    fig = plt.figure(figsize=(7.0, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    y = np.arange(len(df))[::-1]
    ax1.barh(y, df["pump_saving_pct"], color=colors, height=0.48)
    for yi, (_, row) in zip(y, df.iterrows()):
        ax1.text(row["pump_saving_pct"] + 0.8, yi, f"{row['pump_saving_pct']:.2f}%\n{row['pump_saved_m3']:,.0f} 立方米", va="center", fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["name"])
    ax1.set_xlabel("泵耗降低比例（%）")
    ax1.set_title("A  分层泵耗收益", loc="left", fontweight="bold")
    ax1.set_xlim(0, 40)
    ax1.grid(axis="x", color=COLORS["grid"], linewidth=0.8)

    # Trade-off plot: saving vs severe exposure change.
    sizes = 80 + 240 * df["pump_saved_m3"] / df["pump_saved_m3"].max()
    ax2.axhline(0, color="#111827", lw=0.8)
    ax2.scatter(df["pump_saving_pct"], df["d_time_over_7p5_s"], s=sizes, color=colors, alpha=0.8, edgecolor="white", linewidth=1.0)
    for _, row in df.iterrows():
        ax2.text(row["pump_saving_pct"] + 0.55, row["d_time_over_7p5_s"] + 0.15, row["name"], fontsize=8, va="center")
    ax2.set_xlabel("泵耗降低比例（%）")
    ax2.set_ylabel("T>7.5° 增加（秒）")
    ax2.set_title("B  收益与较严重姿态压力", loc="left", fontweight="bold")
    ax2.set_xlim(18, 37)
    ax2.set_ylim(-2, max(14, df["d_time_over_7p5_s"].max() + 3))
    ax2.grid(color=COLORS["grid"], linewidth=0.8)
    ax2.text(
        0.02,
        0.95,
        "点面积表示节省泵耗量",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#374151",
    )
    save(fig, "fig_positive_regime_tradeoff")


def figure_12h_boundary() -> None:
    closed = 79061.46
    supervised = 62791.70
    saved = closed - supervised
    pct = saved / closed * 100

    fig = plt.figure(figsize=(6.8, 3.55), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.bar([0, 1], [closed, supervised], color=[COLORS["baseline"], COLORS["supervised"]], width=0.56)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["原闭环", "监督策略"])
    ax1.set_ylabel("累计泵耗（立方米）")
    ax1.set_title("A  40个12小时窗口泵耗", loc="left", fontweight="bold")
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    for x, val in zip([0, 1], [closed, supervised]):
        ax1.text(x, val + closed * 0.018, f"{val:,.0f}", ha="center", fontsize=8)
    ax1.annotate(
        f"降低 {pct:.2f}%",
        xy=(1, supervised),
        xytext=(0.32, closed * 0.70),
        arrowprops=dict(arrowstyle="->", color=COLORS["saving"], lw=1.1),
        color=COLORS["saving"],
        fontsize=8.5,
    )

    metrics = ["T>5°", "T>7.5°", "T>10°"]
    vals = [-31035, -404, 4]
    y = np.arange(len(metrics))[::-1]
    ax2.axvline(0, color="#111827", lw=0.9)
    ax2.barh(y, vals, height=0.28, color=[COLORS["saving"], COLORS["saving"], COLORS["boundary"]], alpha=0.72)
    ax2.scatter(vals, y, s=58, color=[COLORS["saving"], COLORS["saving"], COLORS["boundary"]], zorder=3)
    for yi, val in zip(y, vals):
        if val < -1000:
            offset, ha = 900, "left"
        elif val < 0:
            offset, ha = -1100, "right"
        else:
            offset, ha = 420, "left"
        ax2.text(val + offset, yi, f"{val:+,} 秒", va="center", ha=ha, fontsize=8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(metrics)
    ax2.set_xlabel("相对原闭环变化（秒）")
    ax2.set_title("B  姿态暴露变化", loc="left", fontweight="bold")
    ax2.set_xlim(-33500, 2500)
    ax2.set_xticks([-30000, -20000, -10000, 0])
    ax2.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    save(fig, "fig_12h_robustness_boundary")


def figure_positive_actionable_attitude_lines() -> None:
    root = ROOT / "outputs/wind_prediction/selector_mixed_6h_limit20_v1/blind_d1_engineered_101case_6h"
    per = pd.read_csv(
        ROOT
        / "outputs/wind_prediction/selector_mixed_6h_limit20_v1/selector_gated_summary_101case_6h/selector_gated_per_case.csv"
    )
    per = per.loc[per["validation_role"] == "positive_allow"].copy()

    rows = []
    for _, row in per.iterrows():
        cid = row["case_id"]
        closed_file = next((root / "timeseries").glob(f"{cid}_*_closed_only_timeseries.csv"))
        supervised_file = next((root / "timeseries").glob(f"{cid}_*_prediction_primary_econ_timeseries.csv"))
        vals = {}
        for key, path in [("原闭环", closed_file), ("监督策略", supervised_file)]:
            ts = pd.read_csv(path, usecols=["pitch_deg", "roll_deg"])
            max_axis = np.maximum(ts["pitch_deg"].abs().to_numpy(), ts["roll_deg"].abs().to_numpy())
            vals[key] = float(np.nanmean(max_axis))
        rows.append(
            {
                "case_id": cid,
                "closed_mean": vals["原闭环"],
                "supervised_mean": vals["监督策略"],
                "delta": vals["监督策略"] - vals["原闭环"],
            }
        )

    df = pd.DataFrame(rows).sort_values("closed_mean").reset_index(drop=True)
    x = np.arange(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(6.4, 3.55), constrained_layout=True)
    ax.plot(
        x,
        df["closed_mean"],
        color=COLORS["baseline"],
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        label="原闭环",
    )
    ax.plot(
        x,
        df["supervised_mean"],
        color=COLORS["supervised"],
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        label="监督策略",
    )
    ax.fill_between(
        x,
        df["closed_mean"],
        df["supervised_mean"],
        where=df["supervised_mean"] >= df["closed_mean"],
        color=COLORS["supervised"],
        alpha=0.12,
        linewidth=0,
    )
    ax.axhline(df["closed_mean"].mean(), color=COLORS["baseline"], linewidth=0.9, linestyle="--", alpha=0.8)
    ax.axhline(df["supervised_mean"].mean(), color=COLORS["supervised"], linewidth=0.9, linestyle="--", alpha=0.8)
    ax.text(
        1,
        df["supervised_mean"].max() * 0.96,
        f"均值：原闭环 {df['closed_mean'].mean():.2f}°，监督策略 {df['supervised_mean'].mean():.2f}°",
        ha="left",
        va="top",
        fontsize=8,
        color="#374151",
    )
    ax.set_xlabel("预测可行动窗口序号（按原闭环平均姿态角排序）")
    ax.set_ylabel("平均姿态倾角（°）")
    ax.set_title("预测可行动窗口平均姿态倾角对比", loc="left", fontweight="bold", fontsize=10)
    ax.set_xlim(1, len(df))
    ax.set_ylim(0, max(df["closed_mean"].max(), df["supervised_mean"].max()) * 1.10)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    save(fig, "fig_positive_actionable_attitude_lines")


def figure_positive_actionable_pump_lines() -> None:
    per = pd.read_csv(
        ROOT
        / "outputs/wind_prediction/selector_mixed_6h_limit20_v1/selector_gated_summary_101case_6h/selector_gated_per_case.csv"
    )
    per["saved_m3"] = per["closed_pump_m3"] - per["gated_pump_m3"]
    df = per.loc[(per["validation_role"] == "positive_allow") & (per["saved_m3"] > 1e-6)].copy()
    df = df.sort_values("saved_m3", ascending=False)
    df = df.reset_index(drop=True)
    df["cumulative_saved_m3"] = df["saved_m3"].cumsum()
    x = np.arange(1, len(df) + 1)

    total_closed = float(df["closed_pump_m3"].sum())
    total_saved = float(df["saved_m3"].sum())
    saving_pct = total_saved / total_closed * 100.0
    mean_saved = float(df["saved_m3"].mean())
    if len(df) != 50:
        raise ValueError(f"Expected 50 actionable windows, found {len(df)}")
    if not np.isclose(total_saved, 18558.71, atol=0.03):
        raise ValueError(f"Unexpected actionable saved volume: {total_saved:.4f} m3")
    if not np.isclose(saving_pct, 34.35, atol=0.01):
        raise ValueError(f"Unexpected actionable saving percentage: {saving_pct:.4f}%")

    blue = "#0F4D92"
    blue_soft = "#B4C0E4"
    teal = "#42949E"
    neutral_light = "#CFCECE"
    neutral_grid = "#E8E8E8"
    neutral_mid = "#767676"
    neutral_dark = "#4D4D4D"
    fig_width = 183 / 25.4
    fig_height = 94 / 25.4
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [0.86, 1.32], "hspace": 0.13},
    )
    fig.subplots_adjust(left=0.088, right=0.878, top=0.835, bottom=0.18)
    fig.text(
        0.088,
        0.975,
        "具有调节空间窗口的累计泵量降低量",
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color=neutral_dark,
    )
    fig.text(
        0.088,
        0.928,
        "仅展示50个实际产生累计泵量降低的窗口；横轴按单窗口降低量降序排列。",
        ha="left",
        va="top",
        fontsize=6.6,
        color=neutral_mid,
    )

    ax1.axhline(0, color=neutral_light, linewidth=0.58, linestyle=(0, (4, 2)))
    ax1.vlines(x, 0, df["saved_m3"], color=blue_soft, linewidth=0.46, alpha=0.54)
    ax1.plot(
        x,
        df["saved_m3"],
        color=blue,
        linewidth=0.86,
        marker="o",
        markersize=2.15,
        markerfacecolor="white",
        markeredgecolor=blue,
        markeredgewidth=0.58,
        zorder=3,
    )
    ax1.axhline(mean_saved, color=neutral_mid, linewidth=0.62, linestyle=(0, (3, 2)))
    ax1.text(
        49.4,
        mean_saved + 62.0,
        f"均值 {mean_saved:.2f} m$^3$",
        ha="right",
        va="center",
        fontsize=6.2,
        color=neutral_dark,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.94},
    )
    ax1.text(-0.078, 1.045, "a", transform=ax1.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=neutral_dark)
    ax1.text(-0.036, 1.045, "单窗口累计泵量降低量", transform=ax1.transAxes, ha="left", va="bottom", fontsize=7.2, color=neutral_dark)
    ax1.set_ylabel("降低量 (m$^3$)", fontsize=6.8, labelpad=2)
    ax1.set_ylim(-35, float(df["saved_m3"].max()) * 1.12)
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

    cumulative_1e4 = df["cumulative_saved_m3"].to_numpy(dtype=float) / 1e4
    total_1e4 = total_saved / 1e4
    ax2.plot(
        x,
        cumulative_1e4,
        color=teal,
        linewidth=1.14,
        marker="o",
        markersize=2.0,
        markerfacecolor="white",
        markeredgecolor=teal,
        markeredgewidth=0.55,
        zorder=3,
    )
    ax2.fill_between(x, 0, cumulative_1e4, color=teal, alpha=0.11, linewidth=0)
    ax2.axhline(total_1e4, color=neutral_light, linewidth=0.58, linestyle=(0, (4, 2)))
    ax2.scatter([50], [total_1e4], s=17, color=teal, zorder=4)
    ax2.text(
        50.9,
        total_1e4,
        f"{total_saved:,.2f} m$^3$\n{saving_pct:.2f}%",
        ha="left",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=teal,
        linespacing=1.15,
        clip_on=False,
    )
    ax2.text(-0.078, 1.035, "b", transform=ax2.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=neutral_dark)
    ax2.text(-0.036, 1.035, "累计泵量降低量", transform=ax2.transAxes, ha="left", va="bottom", fontsize=7.2, color=neutral_dark)
    ax2.set_ylabel("累计降低量 ($10^4$ m$^3$)", fontsize=6.8, labelpad=2)
    ax2.set_xlabel("具有调节空间窗口序号", fontsize=6.8, labelpad=2)
    ax2.set_ylim(-0.05, total_1e4 * 1.13)
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1f}"))
    ax2.set_xlim(1, 55)
    ax2.set_xticks([1, 10, 20, 30, 40, 50])

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(axis="y", color=neutral_grid, linewidth=0.45)
        ax.grid(axis="x", visible=False)
        ax.spines["left"].set_color(neutral_light)
        ax.spines["bottom"].set_color(neutral_light)
        ax.spines["left"].set_linewidth(0.62)
        ax.spines["bottom"].set_linewidth(0.62)
        ax.tick_params(axis="both", colors=neutral_mid, labelsize=6.4, length=2.2, width=0.62, pad=1.5)
        ax.yaxis.label.set_color(neutral_dark)
        ax.xaxis.label.set_color(neutral_dark)

    fig.text(
        0.088,
        0.052,
        "注：风险边界窗口和低扰动背景窗口见表11；其最终保持无预测闭环反馈策略输出，未纳入本图横轴。",
        ha="left",
        va="center",
        fontsize=6.2,
        color=neutral_mid,
    )
    save_publication(fig, "fig_positive_actionable_pump_lines")


def figure_positive_latch_switch_comparison() -> None:
    per = pd.read_csv(
        ROOT
        / "outputs/wind_prediction/selector_mixed_6h_limit20_v1/selector_gated_summary_101case_6h/selector_gated_per_case.csv"
    )
    df = per.loc[per["validation_role"] == "positive_allow"].copy()
    name_map = {
        "p2_neutral_headroom": "姿态余量型",
        "c3_gusty_oscillatory": "阵风振荡型",
        "w1_stable_direction_event": "稳定风向事件型",
    }
    order = ["姿态余量型", "阵风振荡型", "稳定风向事件型"]
    df["stratum_cn"] = df["selector_stratum"].map(name_map)
    df["switch_reduction"] = df["closed_latch_switches"] - df["gated_latch_switches"]

    grouped = (
        df.groupby("stratum_cn")
        .agg(
            cases=("case_id", "count"),
            closed=("closed_latch_switches", "sum"),
            supervised=("gated_latch_switches", "sum"),
            reduction=("switch_reduction", "sum"),
        )
        .reindex(order)
        .reset_index()
    )
    grouped["reduction_pct"] = grouped["reduction"] / grouped["closed"] * 100.0
    total_closed = float(df["closed_latch_switches"].sum())
    total_supervised = float(df["gated_latch_switches"].sum())
    total_reduction = total_closed - total_supervised
    total_pct = total_reduction / total_closed * 100.0

    fig = plt.figure(figsize=(7.2, 3.85), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    y = np.arange(len(grouped))[::-1]
    ax1.barh(y + 0.16, grouped["closed"], height=0.28, color="#CBD5E1", label="原闭环")
    ax1.barh(y - 0.16, grouped["supervised"], height=0.28, color=COLORS["supervised"], label="监督策略")
    for yi, (_, row) in zip(y, grouped.iterrows()):
        ax1.text(
            row["supervised"] + 340,
            yi - 0.16,
            f"-{row['reduction_pct']:.1f}%",
            va="center",
            fontsize=8,
            color=COLORS["saving"],
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r.stratum_cn}\n(n={int(r.cases)})" for _, r in grouped.iterrows()])
    ax1.set_xlabel("启停切换次数（次）")
    ax1.set_title(
        f"A  分工况启停切换次数\n合计减少 {total_reduction:,.0f} 次，降幅 {total_pct:.2f}%",
        loc="left",
        fontweight="bold",
    )
    ax1.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax1.legend(frameon=False, loc="upper right")

    colors = {"姿态余量型": COLORS["action"], "阵风振荡型": "#8B5CF6", "稳定风向事件型": COLORS["saving"]}
    rng = np.random.default_rng(9)
    for i, role in enumerate(order):
        sub = df.loc[df["stratum_cn"] == role, "switch_reduction"].to_numpy(dtype=float)
        xpos = np.full(len(sub), i) + rng.uniform(-0.12, 0.12, len(sub))
        ax2.scatter(xpos, sub, s=28, color=colors[role], alpha=0.70, edgecolor="white", linewidth=0.45)
        ax2.hlines(np.median(sub), i - 0.22, i + 0.22, color="#111827", linewidth=1.2)
    ax2.axhline(0, color="#111827", linewidth=0.85)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=15, ha="right")
    ax2.set_ylabel("单窗口减少次数（次）")
    ax2.set_title("B  逐窗口减少量分布", loc="left", fontweight="bold")
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax2.text(
        0.02,
        0.95,
        "点为单窗口；横线为中位数",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#374151",
    )
    save(fig, "fig_positive_latch_switch_comparison")


def figure_selected_attitude_curves() -> None:
    root = ROOT / "outputs/wind_prediction/selector_mixed_6h_limit20_v1/blind_d1_engineered_101case_6h"
    per = pd.read_csv(
        ROOT
        / "outputs/wind_prediction/selector_mixed_6h_limit20_v1/selector_gated_summary_101case_6h/selector_gated_per_case.csv"
    ).set_index("case_id")
    cases = [
        ("01_sel6h_01", "姿态余量型"),
        ("21_sel6h_21", "阵风振荡型"),
        ("48_sel6h_48", "稳定风向事件型"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 5.6), sharex=True, constrained_layout=True)
    for ax, (cid, label) in zip(axes, cases):
        r = per.loc[cid]
        closed_file = next((root / "timeseries").glob(f"{cid}_*_closed_only_timeseries.csv"))
        supervised_file = next((root / "timeseries").glob(f"{cid}_*_prediction_primary_econ_timeseries.csv"))

        data = {}
        for key, path in [("原闭环", closed_file), ("监督策略", supervised_file)]:
            ts = pd.read_csv(path, usecols=["t_s", "pitch_deg", "roll_deg"])
            t_h = ts["t_s"].to_numpy(dtype=float) / 3600.0
            max_axis = np.maximum(ts["pitch_deg"].abs().to_numpy(), ts["roll_deg"].abs().to_numpy())
            # The validation metrics use raw 1 Hz signals; this display curve is smoothed only for readability.
            max_axis = (
                pd.Series(max_axis)
                .rolling(window=60, center=True, min_periods=1)
                .mean()
                .to_numpy(dtype=float)
            )
            data[key] = (t_h, max_axis)

        ax.plot(data["原闭环"][0], data["原闭环"][1], color=COLORS["baseline"], linewidth=1.0, label="原闭环")
        ax.plot(data["监督策略"][0], data["监督策略"][1], color=COLORS["supervised"], linewidth=1.0, label="监督策略")
        for yline, lw in [(5, 0.8), (7.5, 0.75), (10, 0.75)]:
            ax.axhline(yline, color="#9CA3AF", linestyle="--", linewidth=lw, alpha=0.8)
        ax.set_ylabel("姿态角（°）")
        ax.set_ylim(0, max(11.0, max(data["原闭环"][1].max(), data["监督策略"][1].max()) * 1.08))
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        saving_pct = float(r["gated_saving_pct"])
        d5 = float(r["gated_d_time_over_5_s"])
        d75 = float(r["gated_d_time_over_7p5_s"])
        ax.text(
            0.01,
            0.93,
            f"{label}  泵耗降低 {saving_pct:.2f}%  T>5° {d5:+.0f}秒  T>7.5° {d75:+.0f}秒",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#111827",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        )
        if ax is axes[0]:
            ax.legend(frameon=False, loc="upper right", ncols=2)
    axes[-1].set_xlabel("时间（小时）")
    axes[0].set_title("典型预测可行动窗口的姿态响应曲线（60秒滑动平均）", loc="left", fontweight="bold", fontsize=10)
    save(fig, "fig_selected_positive_attitude_curves")


def figure_compact_table6_only() -> None:
    """A compact single-column alternative for the mixed summary."""
    df = mixed_summary_data()
    total_closed = df["closed"].sum()
    total_supervised = df["supervised"].sum()
    total_saved = total_closed - total_supervised

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    y = np.arange(len(df))[::-1]
    ax.barh(y + 0.14, df["closed"], height=0.24, color="#CBD5E1", label="原闭环")
    ax.barh(y - 0.14, df["supervised"], height=0.24, color=COLORS["supervised"], label="监督策略")
    for yi, (_, row) in zip(y, df.iterrows()):
        if row["saving_pct"] > 0:
            ax.text(row["supervised"] + 1200, yi - 0.14, f"-{row['saving_pct']:.2f}%", va="center", fontsize=8, color=COLORS["saving"])
        else:
            ax.text(row["closed"] + 1200, yi, "0%", va="center", fontsize=8, color="#374151")
    ax.set_yticks(y)
    ax.set_yticklabels(df["short"])
    ax.set_xlabel("累计泵耗（立方米）")
    ax.set_title("混合工况泵耗对比", loc="left", fontweight="bold")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.03, 0.05, f"总降低 22.48%，节省 {total_saved:,.0f} 立方米", transform=ax.transAxes, fontsize=8, color=COLORS["saving"])
    save(fig, "fig_table6_compact_pump_comparison")


def v7_save(fig: plt.Figure, stem: str) -> None:
    V7_FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(V7_FIGURES / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def v7_style_axis(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.set_facecolor("white")
    ax.spines["left"].set_color(V7_COLORS["axis"])
    ax.spines["bottom"].set_color(V7_COLORS["axis"])
    ax.spines["left"].set_linewidth(0.92)
    ax.spines["bottom"].set_linewidth(0.92)
    ax.tick_params(axis="both", colors=V7_COLORS["muted"], width=0.88, length=3.2, pad=2.4, labelsize=9.8)
    ax.xaxis.label.set_color(V7_COLORS["neutral"])
    ax.yaxis.label.set_color(V7_COLORS["neutral"])
    ax.xaxis.label.set_size(11.2)
    ax.yaxis.label.set_size(11.2)
    if ygrid:
        ax.grid(axis="y", color=V7_COLORS["grid"], linewidth=0.72)
    if xgrid:
        ax.grid(axis="x", color=V7_COLORS["grid"], linewidth=0.72)


def v7_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=V7_COLORS["neutral"],
    )


def v7_case_paths(case_id: str) -> tuple[Path, Path]:
    closed = sorted(TS_ROOT.glob(f"{case_id}_*_closed_only_timeseries.csv"))
    predictive = sorted(TS_ROOT.glob(f"{case_id}_*_prediction_primary_econ_timeseries.csv"))
    if len(closed) != 1 or len(predictive) != 1:
        raise FileNotFoundError(f"Expected one closed and one predictive timeseries for {case_id}")
    return closed[0], predictive[0]


def v7_load_final_per_case() -> pd.DataFrame:
    df = pd.read_csv(FINAL_PER_CASE)
    df["窗口序号"] = np.arange(1, len(df) + 1)
    df["窗口类别"] = df["validation_role"].map(ROLE_CN)
    df["累计泵量降低_m3"] = df["closed_pump_m3"] - df["gated_pump_m3"]
    df["水泵启停频次减少"] = df["closed_latch_switches"] - df["gated_latch_switches"]
    return df


def v7_load_extended_per_case() -> pd.DataFrame:
    df = pd.read_csv(FINAL_EXTENDED_PER_CASE)
    df["窗口序号"] = np.arange(1, len(df) + 1)
    df["窗口类别"] = df["validation_role"].map(ROLE_CN)
    return df


def v7_load_actuator_source() -> pd.DataFrame:
    df = pd.read_csv(V6_TABLES / "fig_actuator_burden_benefit_final_source.csv")
    df["动作时长减少率_pct"] = (
        df["动作时长减少_min"] / df["无预测闭环反馈策略动作时长_min"] * 100.0
    )
    df["水泵启停频次减少率_pct"] = (
        df["启停频次减少"] / df["无预测闭环反馈策略启停频次"] * 100.0
    )
    df["累计目标水量变化减少率_pct"] = (
        df["累计目标水量变化减少_t"] / df["无预测闭环反馈策略累计目标水量变化_t"] * 100.0
    )
    return df


def v7_validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per = v7_load_final_per_case()
    ext = v7_load_extended_per_case()
    actuator = v7_load_actuator_source()

    role_counts = per["validation_role"].value_counts().to_dict()
    expected_counts = {"positive_allow": 50, "negative_abstain": 31, "background_abstain": 20}
    if role_counts != expected_counts:
        raise ValueError(f"Unexpected validation-role counts: {role_counts}")

    active = per.loc[per["validation_role"] == "positive_allow"]
    total_saved = float(per["累计泵量降低_m3"].sum())
    active_saved = float(active["累计泵量降低_m3"].sum())
    total_saving_pct = total_saved / float(per["closed_pump_m3"].sum()) * 100.0
    active_saving_pct = active_saved / float(active["closed_pump_m3"].sum()) * 100.0
    total_switch_pct = (
        (per["closed_latch_switches"].sum() - per["gated_latch_switches"].sum())
        / per["closed_latch_switches"].sum()
        * 100.0
    )
    active_switch_pct = (
        (active["closed_latch_switches"].sum() - active["gated_latch_switches"].sum())
        / active["closed_latch_switches"].sum()
        * 100.0
    )
    checks = {
        "具有调节空间窗口累计泵量降低率": (active_saving_pct, 34.35, 0.02),
        "综合验证窗口累计泵量降低率": (total_saving_pct, 22.48, 0.02),
        "具有调节空间窗口水泵启停频次降低率": (active_switch_pct, 39.93, 0.02),
        "综合验证窗口水泵启停频次降低率": (total_switch_pct, 26.95, 0.02),
        "T>5累计超阈时间变化": (float(ext["gated_d_time_over_5_s"].sum()), 26.0, 0.1),
        "T>7.5累计超阈时间变化": (float(ext["gated_d_time_over_7p5_s"].sum()), 4.0, 0.1),
        "T>10累计超阈时间变化": (float(ext["gated_d_time_over_10_s"].sum()), 0.0, 0.1),
    }
    for name, (value, expected, atol) in checks.items():
        if not np.isclose(value, expected, atol=atol):
            raise ValueError(f"{name} mismatch: {value:.4f}, expected {expected:.4f}")

    if int((actuator["动作时长减少_min"] > 0).sum()) != 50:
        raise ValueError("Expected 50/50 windows with reduced action duration")

    return per, ext, actuator


def v7_strip_points(
    ax: plt.Axes,
    values: pd.Series,
    *,
    title: str,
    xlabel: str,
    positive_label: str,
    xlim: tuple[float, float] | None = None,
    seed: int = 0,
) -> None:
    vals = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 0.035, len(vals))
    colors = np.where(vals >= 0, V7_COLORS["predictive"], V7_COLORS["penalty"])
    ax.axvline(0, color=V7_COLORS["muted"], linewidth=0.82, linestyle=(0, (3, 2)), zorder=1)
    ax.scatter(vals, y, s=22, color=colors, alpha=0.72, edgecolor="white", linewidth=0.42, zorder=3)
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    ax.hlines(0.16, q25, q75, color=V7_COLORS["predictive_dark"], linewidth=3.2, alpha=0.34, zorder=2)
    ax.scatter([med], [0.16], marker="D", s=28, color=V7_COLORS["neutral"], edgecolor="white", linewidth=0.4, zorder=4)
    ax.set_ylim(-0.16, 0.31)
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=8.3, fontweight="bold", color=V7_COLORS["neutral"], pad=3.0)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.text(
        0.99,
        0.80,
        f"{positive_label}\n中位数 {med:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.1,
        color=V7_COLORS["muted"],
    )
    v7_style_axis(ax, xgrid=True, ygrid=False)


def figure_v7_a_actuator_burden_distribution(actuator: pd.DataFrame) -> None:
    source = actuator[
        [
            "窗口序号",
            "时间",
            "窗口类别",
            "累计泵量降低率_pct",
            "动作时长减少率_pct",
            "启停频次减少",
            "水泵启停频次减少率_pct",
            "累计目标水量变化减少_t",
            "累计目标水量变化减少率_pct",
        ]
    ].copy()
    V7_TABLES.mkdir(parents=True, exist_ok=True)
    source.to_csv(V7_TABLES / "fig_a_actuator_burden_distribution_source.csv", index=False)

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 5.45), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.91, bottom=0.10, hspace=0.78)
    fig.suptitle(
        "具有调节空间窗口的执行器负担降低分布",
        x=0.12,
        y=0.98,
        ha="left",
        fontsize=10.2,
        fontweight="bold",
        color=V7_COLORS["neutral"],
    )
    specs = [
        (
            "a",
            "累计泵量降低",
            "累计泵量降低率_pct",
            "累计泵量降低率（%）",
            f"{int((source['累计泵量降低率_pct'] > 0).sum())}/50 降低",
            (0, 56),
        ),
        (
            "b",
            "动作时长减少",
            "动作时长减少率_pct",
            "动作时长减少率（%）",
            f"{int((source['动作时长减少率_pct'] > 0).sum())}/50 减少",
            (0, 78),
        ),
        (
            "c",
            "水泵启停频次减少",
            "启停频次减少",
            "水泵启停频次减少（次）",
            f"{int((source['启停频次减少'] > 0).sum())}/50 减少",
            (-210, 850),
        ),
        (
            "d",
            "累计目标水量变化减少",
            "累计目标水量变化减少率_pct",
            "累计目标水量变化减少率（%）",
            f"{int((source['累计目标水量变化减少率_pct'] > 0).sum())}/50 减少",
            (-36, 46),
        ),
    ]
    for i, (ax, (label, title, col, xlabel, positive_label, xlim)) in enumerate(zip(axes, specs, strict=True)):
        v7_panel_label(ax, label)
        v7_strip_points(
            ax,
            source[col],
            title=title,
            xlabel=xlabel,
            positive_label=positive_label,
            xlim=xlim,
            seed=11 + i,
        )

    v7_save(fig, "fig_a_actuator_burden_distribution")


def figure_v7_b_pump_duration_relation(actuator: pd.DataFrame) -> None:
    source = actuator[
        [
            "窗口序号",
            "时间",
            "窗口类别",
            "累计泵量降低率_pct",
            "动作时长减少率_pct",
            "启停频次减少",
            "水泵启停频次减少率_pct",
        ]
    ].copy()
    source.to_csv(V7_TABLES / "fig_b_pump_duration_switch_relation_source.csv", index=False)

    x = source["累计泵量降低率_pct"].to_numpy(dtype=float)
    y = source["动作时长减少率_pct"].to_numpy(dtype=float)
    switches = source["启停频次减少"].to_numpy(dtype=float)
    switch_vmin, switch_vmax = -210, 850
    zero_position = (0 - switch_vmin) / (switch_vmax - switch_vmin)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "switch_reduction",
        [
            (0.0, V7_COLORS["penalty"]),
            (zero_position, "#E8EEF2"),
            (1.0, V7_COLORS["predictive_dark"]),
        ],
    )
    norm = mpl.colors.Normalize(vmin=switch_vmin, vmax=switch_vmax)

    fig, ax = plt.subplots(figsize=(7.35, 4.05), constrained_layout=True)
    sc = ax.scatter(
        x,
        y,
        c=switches,
        cmap=cmap,
        norm=norm,
        s=52,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.62,
        zorder=3,
    )
    coef = np.polyfit(x, y, deg=1)
    xx = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(xx, coef[0] * xx + coef[1], color=V7_COLORS["neutral"], linewidth=1.12, linestyle=(0, (4, 2)))
    ax.axhline(0, color=V7_COLORS["muted"], linewidth=0.92, linestyle=(0, (3, 2)))
    ax.set_xlabel("累计泵量降低率（%）")
    ax.set_ylabel("动作时长减少率（%）")
    ax.set_xlim(8, 54)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.86)
    cbar.set_ticks([-200, -100, 0, 200, 400, 600, 800])
    cbar.set_label("水泵启停频次减少（次）", fontsize=11.2)
    cbar.ax.tick_params(labelsize=9.8, length=2.8)
    v7_style_axis(ax, xgrid=False, ygrid=True)
    v7_save(fig, "fig_b_pump_duration_switch_relation")


def v7_read_timeseries(path: Path, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=cols)


def v7_build_timeseries_products(per: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive = per.loc[per["validation_role"] == "positive_allow"].copy().reset_index(drop=True)
    positive["调节空间窗口序号"] = np.arange(1, len(positive) + 1)
    if len(positive) != 50:
        raise ValueError(f"Expected 50 windows with regulation room, found {len(positive)}")
    positive_window_no = dict(zip(positive["case_id"], positive["调节空间窗口序号"], strict=True))

    duty_closed_sum: np.ndarray | None = None
    duty_predictive_sum: np.ndarray | None = None
    time_s: np.ndarray | None = None
    cumulative_rows: list[dict[str, float | int]] = []
    cumulative_matrix: list[np.ndarray] = []
    threshold2_records: list[dict[str, float | int | str]] = []
    sample_idx: np.ndarray | None = None

    read_cols = ["t_s", "pump_total_rate_m3_min", "pitch_deg", "roll_deg"]
    posture_cols = ["t_s", "pitch_deg", "roll_deg"]

    for row in per.itertuples(index=False):
        closed_path, predictive_path = v7_case_paths(row.case_id)
        is_positive = row.validation_role == "positive_allow"
        closed = v7_read_timeseries(closed_path, read_cols if is_positive else posture_cols)
        t = closed["t_s"].to_numpy(dtype=float)
        dt_s = float(np.median(np.diff(t)))
        closed_max_axis = np.maximum(
            np.abs(closed["pitch_deg"].to_numpy(dtype=float)),
            np.abs(closed["roll_deg"].to_numpy(dtype=float)),
        )
        closed_over_2_s = float(np.sum(closed_max_axis > 2.0) * dt_s)

        if is_positive:
            predictive = v7_read_timeseries(predictive_path, read_cols)
            predictive_max_axis = np.maximum(
                np.abs(predictive["pitch_deg"].to_numpy(dtype=float)),
                np.abs(predictive["roll_deg"].to_numpy(dtype=float)),
            )
            predictive_over_2_s = float(np.sum(predictive_max_axis > 2.0) * dt_s)

            closed_rate = closed["pump_total_rate_m3_min"].to_numpy(dtype=float)
            predictive_rate = predictive["pump_total_rate_m3_min"].to_numpy(dtype=float)
            if time_s is None:
                time_s = t
                duty_closed_sum = np.zeros_like(t, dtype=float)
                duty_predictive_sum = np.zeros_like(t, dtype=float)
                sample_idx = np.r_[np.arange(59, len(t), 60), len(t) - 1]
                sample_idx = np.unique(sample_idx)
            elif len(t) != len(time_s):
                raise ValueError(f"Unexpected timeseries length for {row.case_id}: {len(t)}")

            duty_closed_sum += (closed_rate > 1e-9).astype(float)
            duty_predictive_sum += (predictive_rate > 1e-9).astype(float)

            dt_min = dt_s / 60.0
            cumulative_extra = np.cumsum((closed_rate - predictive_rate) * dt_min)
            sampled = cumulative_extra[sample_idx]
            cumulative_matrix.append(sampled)
            window_no = int(positive_window_no[row.case_id])
            for idx, value in zip(sample_idx, sampled, strict=True):
                cumulative_rows.append(
                    {
                        "调节空间窗口序号": window_no,
                        "时间_min": float(t[idx] / 60.0),
                        "无预测闭环反馈策略多出的累计泵量_m3": float(value),
                    }
                )
        else:
            predictive_over_2_s = closed_over_2_s

        threshold2_records.append(
            {
                "窗口序号": int(row.窗口序号),
                "时间": row.timestamp,
                "窗口类别": row.窗口类别,
                "阈值_deg": 2.0,
                "无预测闭环反馈策略姿态超阈时间_s": closed_over_2_s,
                "预测辅助闭环调节策略姿态超阈时间_s": predictive_over_2_s,
                "姿态超阈时间变化_s": predictive_over_2_s - closed_over_2_s,
            }
        )

    if time_s is None or duty_closed_sum is None or duty_predictive_sum is None or sample_idx is None:
        raise ValueError("No timeseries products were built")

    duty_raw = pd.DataFrame(
        {
            "时间_min": time_s / 60.0,
            "minute": np.floor(time_s / 60.0).astype(int),
            "无预测闭环反馈策略水泵动作占比_pct": duty_closed_sum / len(positive) * 100.0,
            "预测辅助闭环调节策略水泵动作占比_pct": duty_predictive_sum / len(positive) * 100.0,
        }
    )
    duty = (
        duty_raw.groupby("minute", as_index=False)
        .agg(
            时间_min=("时间_min", "mean"),
            无预测闭环反馈策略水泵动作占比_pct=("无预测闭环反馈策略水泵动作占比_pct", "mean"),
            预测辅助闭环调节策略水泵动作占比_pct=("预测辅助闭环调节策略水泵动作占比_pct", "mean"),
        )
        .drop(columns=["minute"])
    )
    duty["动作占比降低_pct"] = (
        duty["无预测闭环反馈策略水泵动作占比_pct"] - duty["预测辅助闭环调节策略水泵动作占比_pct"]
    )

    cumulative_long = pd.DataFrame(cumulative_rows)
    cumulative_array = np.vstack(cumulative_matrix)
    cumulative_summary = pd.DataFrame(
        {
            "时间_min": time_s[sample_idx] / 60.0,
            "q25_m3": np.percentile(cumulative_array, 25, axis=0),
            "中位数_m3": np.percentile(cumulative_array, 50, axis=0),
            "q75_m3": np.percentile(cumulative_array, 75, axis=0),
            "均值_m3": np.mean(cumulative_array, axis=0),
        }
    )
    threshold2 = pd.DataFrame(threshold2_records)
    return duty, cumulative_long, cumulative_summary, threshold2


def figure_v7_c_actuator_duty_cycle(duty: pd.DataFrame) -> None:
    duty.to_csv(V7_TABLES / "fig_c_actuator_duty_cycle_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.35, 3.75), constrained_layout=True)
    x = duty["时间_min"].to_numpy(dtype=float) / 60.0
    closed = duty["无预测闭环反馈策略水泵动作占比_pct"].to_numpy(dtype=float)
    predictive = duty["预测辅助闭环调节策略水泵动作占比_pct"].to_numpy(dtype=float)
    ax.plot(x, closed, color="#9A7F63", linewidth=0.96, label="无预测闭环反馈策略")
    ax.plot(x, predictive, color="#2E7D68", linewidth=1.08, label="预测辅助闭环调节策略")
    ax.set_xlabel("窗口内时间（h）")
    ax.set_ylabel("水泵动作状态占比（%）")
    ax.set_title("具有调节空间窗口内水泵动作占比随时间变化", loc="left", fontsize=13.0, fontweight="bold")
    ax.legend(loc="upper right", ncols=2, frameon=False, fontsize=10.4, handlelength=2.6)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    v7_style_axis(ax, xgrid=False, ygrid=True)
    v7_save(fig, "fig_c_actuator_duty_cycle")


def figure_v7_d_cumulative_benefit_process(
    cumulative_long: pd.DataFrame, cumulative_summary: pd.DataFrame
) -> None:
    cumulative_long.to_csv(V7_TABLES / "fig_d_cumulative_pump_benefit_process_source.csv", index=False)
    cumulative_summary.to_csv(V7_TABLES / "fig_d_cumulative_pump_benefit_process_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.35, 3.95), constrained_layout=True)
    original_blue = "#2D7FB8"
    original_blue_dark = "#0F4D92"
    for _, sub in cumulative_long.groupby("调节空间窗口序号"):
        ax.plot(
            sub["时间_min"].to_numpy(dtype=float) / 60.0,
            sub["无预测闭环反馈策略多出的累计泵量_m3"].to_numpy(dtype=float),
            color=original_blue,
            linewidth=0.38,
            alpha=0.15,
            zorder=1,
        )
    x = cumulative_summary["时间_min"].to_numpy(dtype=float) / 60.0
    q25 = cumulative_summary["q25_m3"].to_numpy(dtype=float)
    med = cumulative_summary["中位数_m3"].to_numpy(dtype=float)
    q75 = cumulative_summary["q75_m3"].to_numpy(dtype=float)
    ax.fill_between(x, q25, q75, color=original_blue, alpha=0.17, linewidth=0, label="IQR")
    ax.plot(x, med, color=original_blue_dark, linewidth=1.55, label="中位数", zorder=4)
    ax.axhline(0, color=V7_COLORS["muted"], linewidth=0.92, linestyle=(0, (3, 2)))
    ax.set_xlabel("窗口内时间（h）")
    ax.set_ylabel("累计泵量降低量（m³）")
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=13.0,
        handlelength=2.7,
    )
    ax.set_xlim(0, 6)
    ax.set_ylim(-100, 1050)
    ax.set_yticks([-100, 0, 200, 400, 600, 800, 1000])
    v7_style_axis(ax, xgrid=False, ygrid=True)
    v7_save(fig, "fig_d_cumulative_pump_benefit_process")


def v7_threshold_profile_source(ext: pd.DataFrame, threshold2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for row in ext.itertuples(index=False):
        for threshold, col_key in [(3.0, "3"), (4.0, "4"), (5.0, "5"), (7.5, "7p5"), (10.0, "10")]:
            rows.append(
                {
                    "窗口序号": int(row.窗口序号),
                    "时间": row.timestamp,
                    "窗口类别": row.窗口类别,
                    "阈值_deg": threshold,
                    "无预测闭环反馈策略姿态超阈时间_s": float(getattr(row, f"closed_time_over_{col_key}_s")),
                    "预测辅助闭环调节策略姿态超阈时间_s": float(getattr(row, f"gated_time_over_{col_key}_s")),
                    "姿态超阈时间变化_s": float(getattr(row, f"gated_d_time_over_{col_key}_s")),
                }
            )
    df = pd.concat([threshold2, pd.DataFrame(rows)], ignore_index=True)
    order = {2.0: 0, 3.0: 1, 4.0: 2, 5.0: 3, 7.5: 4, 10.0: 5}
    df["_order"] = df["阈值_deg"].map(order)
    df = df.sort_values(["_order", "窗口序号"]).drop(columns=["_order"]).reset_index(drop=True)
    return df


def figure_v7_e_attitude_threshold_risk_curve(threshold_source: pd.DataFrame) -> None:
    threshold_source.to_csv(V7_TABLES / "fig_e_attitude_threshold_risk_curve_source.csv", index=False)
    total_seconds = 101 * 6 * 3600
    summary = (
        threshold_source.groupby("阈值_deg", as_index=False)
        .agg(
            无预测闭环反馈策略姿态超阈时间_s=("无预测闭环反馈策略姿态超阈时间_s", "sum"),
            预测辅助闭环调节策略姿态超阈时间_s=("预测辅助闭环调节策略姿态超阈时间_s", "sum"),
            姿态超阈时间变化_s=("姿态超阈时间变化_s", "sum"),
        )
        .sort_values("阈值_deg")
    )
    summary["姿态超阈时间变化占总验证时长_pct"] = summary["姿态超阈时间变化_s"] / total_seconds * 100.0
    summary["相对无预测闭环反馈策略变化_pct"] = (
        summary["预测辅助闭环调节策略姿态超阈时间_s"]
        / summary["无预测闭环反馈策略姿态超阈时间_s"]
        - 1.0
    ) * 100.0
    summary.to_csv(V7_TABLES / "fig_e_attitude_threshold_risk_curve_summary.csv", index=False)

    labels = [f"{v:g}°" for v in summary["阈值_deg"]]
    x = np.arange(len(summary))
    y = summary["姿态超阈时间变化占总验证时长_pct"].to_numpy(dtype=float)
    delta_s = summary["姿态超阈时间变化_s"].to_numpy(dtype=float)
    colors = [V7_COLORS["benefit"] if val < 0 else V7_COLORS["penalty"] if val > 0 else V7_COLORS["neutral"] for val in delta_s]

    fig, ax = plt.subplots(figsize=(6.6, 3.45), constrained_layout=True)
    ax.axhline(0, color=V7_COLORS["muted"], linewidth=0.78, linestyle=(0, (3, 2)))
    ax.vlines(x, 0, y, color=colors, linewidth=2.0, alpha=0.55)
    ax.scatter(x, y, s=48, color=colors, edgecolor="white", linewidth=0.55, zorder=3)
    for xi, yi, ds in zip(x, y, delta_s, strict=True):
        ax.text(
            xi,
            yi + (0.010 if yi >= 0 else -0.010),
            f"{ds:+.0f} s",
            ha="center",
            va="bottom" if yi >= 0 else "top",
            fontsize=7.1,
            color=V7_COLORS["neutral"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("姿态超阈时间变化占总验证时长（%）")
    ax.set_xlabel("姿态阈值")
    ax.set_title("综合验证窗口的姿态超阈风险剖面", loc="left", fontsize=10.0, fontweight="bold")
    ax.text(
        0.98,
        0.92,
        "阈值越接近风险边界，增量越小；10°为0",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=V7_COLORS["neutral"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.8},
    )
    ymax = max(0.02, float(np.nanmax(np.abs(y))) * 1.30)
    ax.set_ylim(-ymax * 0.12, ymax)
    v7_style_axis(ax, xgrid=False, ygrid=True)
    v7_save(fig, "fig_e_attitude_threshold_risk_curve")


def figure_v7_f_attitude_severity_decomposition(ext: pd.DataFrame) -> None:
    active = ext.loc[ext["validation_role"] == "positive_allow"].copy().reset_index(drop=True)
    active["调节空间窗口序号"] = np.arange(1, len(active) + 1)
    source = pd.DataFrame(
        {
            "调节空间窗口序号": active["调节空间窗口序号"],
            "时间": active["timestamp"],
            "窗口类别": active["窗口类别"],
            "累计泵量降低率_pct": active["gated_saving_pct"],
            "T大于5度累计超阈时间变化_s": active["gated_d_time_over_5_s"],
            "T大于5度最长连续超阈时间变化_s": active["gated_d_max_cont_over_5_s"],
            "T大于5度超阈面积变化_deg_s": active["gated_d_area_over_5_deg_s"],
        }
    )
    source.to_csv(V7_TABLES / "fig_f_attitude_severity_decomposition_source.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.18, wspace=0.34)
    fig.suptitle(
        "T>5°姿态事件严重度分解",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
        color=V7_COLORS["neutral"],
    )
    specs = [
        ("a", "累计超阈时间", "T大于5度累计超阈时间变化_s", "变化（s）", (-20, 45)),
        ("b", "最长连续超阈时间", "T大于5度最长连续超阈时间变化_s", "变化（s）", (-20, 45)),
        ("c", "超阈面积", "T大于5度超阈面积变化_deg_s", "变化（deg·s）", (-20, 45)),
    ]
    for i, (ax, (label, title, col, xlabel, xlim)) in enumerate(zip(axes, specs, strict=True)):
        vals = source[col]
        v7_panel_label(ax, label)
        v7_strip_points(
            ax,
            vals,
            title=title,
            xlabel=xlabel,
            positive_label=f"{int((vals > 0).sum())}/50 增加",
            xlim=xlim,
            seed=41 + i,
        )
    v7_save(fig, "fig_f_attitude_severity_decomposition")


def v7_write_readme(
    per: pd.DataFrame,
    ext: pd.DataFrame,
    actuator: pd.DataFrame,
    duty: pd.DataFrame,
    threshold_source: pd.DataFrame,
) -> None:
    V7_OUT.mkdir(parents=True, exist_ok=True)
    active = per.loc[per["validation_role"] == "positive_allow"]
    active_saved = float(active["累计泵量降低_m3"].sum())
    active_saving_pct = active_saved / float(active["closed_pump_m3"].sum()) * 100.0
    active_switch_pct = (
        (active["closed_latch_switches"].sum() - active["gated_latch_switches"].sum())
        / active["closed_latch_switches"].sum()
        * 100.0
    )
    total_saving_pct = float(per["累计泵量降低_m3"].sum() / per["closed_pump_m3"].sum() * 100.0)
    total_switch_pct = (
        (per["closed_latch_switches"].sum() - per["gated_latch_switches"].sum())
        / per["closed_latch_switches"].sum()
        * 100.0
    )
    action_duration_reduced = int((actuator["动作时长减少_min"] > 0).sum())
    target_reduced = int((actuator["累计目标水量变化减少_t"] > 0).sum())
    duty_closed = float(duty["无预测闭环反馈策略水泵动作占比_pct"].mean())
    duty_predictive = float(duty["预测辅助闭环调节策略水泵动作占比_pct"].mean())
    d5 = float(ext["gated_d_time_over_5_s"].sum())
    d75 = float(ext["gated_d_time_over_7p5_s"].sum())
    d10 = float(ext["gated_d_time_over_10_s"].sum())

    threshold_summary = (
        threshold_source.groupby("阈值_deg")["姿态超阈时间变化_s"].sum().reset_index().sort_values("阈值_deg")
    )
    threshold_line = "，".join(
        f"{row['阈值_deg']:g}°: {row['姿态超阈时间变化_s']:+.0f} s"
        for _, row in threshold_summary.iterrows()
    )

    readme = f"""# validation_expansion_v7

本包围绕一个结论筛图：在具有调节空间的窗口内，预测辅助闭环调节策略相对无预测闭环反馈策略降低累计泵量和执行器负担，同时不放大高阈值姿态风险。

## 统计口径

- 101个综合验证窗口：50个具有调节空间的窗口、31个风险边界窗口、20个低扰动背景窗口。
- 具有调节空间的窗口：累计泵量降低 {active_saving_pct:.2f}%，水泵启停频次降低 {active_switch_pct:.2f}%。
- 综合验证窗口总计：累计泵量降低 {total_saving_pct:.2f}%，水泵启停频次降低 {total_switch_pct:.2f}%。
- 最终判别口径下姿态变化：θ_d>5°累计时间 {d5:+.0f} s，θ_d>7.5°累计时间 {d75:+.0f} s，θ_d>10°累计时间 {d10:.0f}。
- 时序图 C/D 使用1 Hz水泵状态和流量积分；图 E 的2°阈值由1 Hz姿态时序补算，其余阈值来自最终逐案扩展统计。
- 本包 A-F 不展示风速/风向时序。若后续增加风速/风向曲线，展示曲线必须做滑动平均；平滑仅用于显示，不参与任何统计计算。

## 候选图与筛选意见

| 图 | 文件 | source csv | 核心结论 | 建议 |
|---|---|---|---|---|
| A | `figures/fig_a_actuator_burden_distribution.*` | `tables/fig_a_actuator_burden_distribution_source.csv` | 累计泵量与动作时长均为50/50降低，水泵启停频次46/50减少，累计目标水量变化38/50减少。 | 信息密度较高，可作为备选或附录。 |
| B | `figures/fig_b_pump_duration_switch_relation.*` | `tables/fig_b_pump_duration_switch_relation_source.csv` | 累计泵量降低率与动作时长减少率呈正相关，且所有点位于动作时长减少区间。 | 建议正文。用于说明累计泵量降低并非以更长动作时长为代价。 |
| C | `figures/fig_c_actuator_duty_cycle.*` | `tables/fig_c_actuator_duty_cycle_source.csv` | 50个窗口汇总后，水泵动作占比从 {duty_closed:.1f}% 降至 {duty_predictive:.1f}%，不是单窗口故事。 | 建议正文或正文备选。用于展示执行器动作占比的时间过程。 |
| D | `figures/fig_d_cumulative_pump_benefit_process.*` | `tables/fig_d_cumulative_pump_benefit_process_source.csv`; `tables/fig_d_cumulative_pump_benefit_process_summary.csv` | 多数窗口的累计收益随时间形成稳定正值，粗线中位数和IQR显示收益不是少数离群窗口造成。 | 建议正文。它补足“收益何时形成”的动态证据。 |
| E | `figures/fig_e_attitude_threshold_risk_curve.*` | `tables/fig_e_attitude_threshold_risk_curve_source.csv`; `tables/fig_e_attitude_threshold_risk_curve_summary.csv` | 阈值剖面为 {threshold_line}，越接近风险边界增量越小，10°为0。 | 建议正文。它直接服务“不放大姿态风险”的主张。 |
| F | `figures/fig_f_attitude_severity_decomposition.*` | `tables/fig_f_attitude_severity_decomposition_source.csv` | T>5°的累计时间、最长连续时间和超阈面积多数窗口为0附近变化，未形成普遍持续性事件尾部。 | 建议附录。正文可引用其结论，除非审稿人要求事件严重度分解。 |

## 弃用判断

本轮 A-F 均有可辨识证据。若正文需要增加执行器与过程证据，优先采用 B、C、D；A 信息密度较高，可作为备选或附录。不要把这些图表述为真实能耗、功率、疲劳载荷或结构损伤改善。

## 主代理需审查的风险点

- 图 E 的2°阈值由1 Hz姿态时序补算，不来自最终逐案扩展表；请确认正文是否需要2°这个低阈值。
- 图 C/D 使用水泵动作状态和流量积分，不等价于真实泵功率或电能消耗。
- 图 F 中 θ_d>5° 最长连续超阈时间变化存在单窗口正向尾部，正文措辞应写“没有形成普遍持续性高姿态风险”，不要写“所有事件都缩短”。
- 若主文只能放一张执行器关系图，优先B；若需要展示时间过程，再补C或D。
"""
    (V7_OUT / "README.md").write_text(readme, encoding="utf-8")


def make_validation_expansion_v7() -> None:
    V7_FIGURES.mkdir(parents=True, exist_ok=True)
    V7_TABLES.mkdir(parents=True, exist_ok=True)
    per, ext, actuator = v7_validate_inputs()

    figure_v7_a_actuator_burden_distribution(actuator)
    figure_v7_b_pump_duration_relation(actuator)

    duty, cumulative_long, cumulative_summary, threshold2 = v7_build_timeseries_products(per)
    threshold_source = v7_threshold_profile_source(ext, threshold2)
    figure_v7_c_actuator_duty_cycle(duty)
    figure_v7_d_cumulative_benefit_process(cumulative_long, cumulative_summary)
    figure_v7_e_attitude_threshold_risk_curve(threshold_source)
    figure_v7_f_attitude_severity_decomposition(ext)
    v7_write_readme(per, ext, actuator, duty, threshold_source)
    print(f"saved validation expansion figures to {V7_OUT}")


def main() -> None:
    if os.environ.get("FOWT_RESULT_FIGURE_SET") == "v7":
        make_validation_expansion_v7()
        return

    figure_mixed_contribution()
    figure_case_distribution()
    figure_positive_tradeoff()
    figure_12h_boundary()
    figure_positive_actionable_attitude_lines()
    figure_positive_actionable_pump_lines()
    figure_positive_latch_switch_comparison()
    figure_selected_attitude_curves()
    figure_compact_table6_only()
    make_validation_expansion_v7()
    print(f"saved figures to {OUT}")
    for p in sorted(OUT.glob("fig_*.png")):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
