from __future__ import annotations

from pathlib import Path

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
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "STSong",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
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
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
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


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
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
    df = per.loc[per["validation_role"] == "positive_allow"].copy()
    df["saved_m3"] = df["closed_pump_m3"] - df["gated_pump_m3"]
    df = df.sort_values("closed_pump_m3").reset_index(drop=True)
    x = np.arange(1, len(df) + 1)

    closed_mean = float(df["closed_pump_m3"].mean())
    supervised_mean = float(df["gated_pump_m3"].mean())
    total_saved = float(df["saved_m3"].sum())
    saving_pct = total_saved / float(df["closed_pump_m3"].sum()) * 100.0

    fig, ax = plt.subplots(figsize=(6.4, 3.55), constrained_layout=True)
    ax.plot(
        x,
        df["closed_pump_m3"],
        color=COLORS["baseline"],
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        label="原闭环",
    )
    ax.plot(
        x,
        df["gated_pump_m3"],
        color=COLORS["supervised"],
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        label="监督策略",
    )
    ax.fill_between(
        x,
        df["gated_pump_m3"].to_numpy(dtype=float),
        df["closed_pump_m3"].to_numpy(dtype=float),
        where=df["closed_pump_m3"].to_numpy(dtype=float) >= df["gated_pump_m3"].to_numpy(dtype=float),
        color=COLORS["saving"],
        alpha=0.14,
        linewidth=0,
        label="节省量",
    )
    ax.axhline(closed_mean, color=COLORS["baseline"], linewidth=0.9, linestyle="--", alpha=0.8)
    ax.axhline(supervised_mean, color=COLORS["supervised"], linewidth=0.9, linestyle="--", alpha=0.8)
    ax.text(
        1,
        max(df["closed_pump_m3"].max(), df["gated_pump_m3"].max()) * 0.96,
        f"总降低 {saving_pct:.2f}%，节省 {total_saved:,.0f} 立方米",
        ha="left",
        va="top",
        fontsize=8,
        color=COLORS["saving"],
    )
    ax.text(
        1,
        max(df["closed_pump_m3"].max(), df["gated_pump_m3"].max()) * 0.87,
        f"均值：原闭环 {closed_mean:.0f}，监督策略 {supervised_mean:.0f} 立方米",
        ha="left",
        va="top",
        fontsize=8,
        color="#374151",
    )
    ax.set_xlabel("预测可行动窗口序号（按原闭环水泵负载排序）")
    ax.set_ylabel("累计水泵负载（立方米）")
    ax.set_title("预测可行动窗口累计水泵负载对比", loc="left", fontweight="bold", fontsize=10)
    ax.set_xlim(1, len(df))
    ax.set_ylim(0, max(df["closed_pump_m3"].max(), df["gated_pump_m3"].max()) * 1.10)
    ax.legend(frameon=False, loc="upper left", ncols=3)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    save(fig, "fig_positive_actionable_pump_lines")


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


def main() -> None:
    figure_mixed_contribution()
    figure_case_distribution()
    figure_positive_tradeoff()
    figure_12h_boundary()
    figure_positive_actionable_attitude_lines()
    figure_positive_actionable_pump_lines()
    figure_positive_latch_switch_comparison()
    figure_selected_attitude_curves()
    figure_compact_table6_only()
    print(f"saved figures to {OUT}")
    for p in sorted(OUT.glob("fig_*.png")):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
