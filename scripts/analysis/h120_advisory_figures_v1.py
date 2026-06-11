"""Generate paper/report figures for the h120 supervisory advisory (read-only).

Reads only the canonical pipeline outputs and writes PNGs into paper_ready/figures/.
Does not touch the controller, the model, or the advisory numbers.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path("outputs/wind_prediction/h120_supervisory_advisory_v1")
FIG = B / "paper_ready" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
contract = json.loads((B / "far_event_advisory_contract.json").read_text())
TAU_ON = contract["tau_on"]
ST = contract["severity_thresholds"]


def fig_pr_frontier():
    val = pd.read_csv(B / "raw_tables" / "validation_pr_frontier.csv")
    test = pd.read_csv(B / "raw_tables" / "test_pr_frontier.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    for df, name, c in [(val, "validation", "tab:blue"), (test, "test (frozen)", "tab:orange")]:
        d = df[df["fires"] > 0].sort_values("recall")
        ax.plot(d["recall"], d["precision"], "-", color=c, label=name, lw=1.8)
    # locked operating point on test
    op = test.iloc[(test["tau"] - TAU_ON).abs().argmin()]
    ax.scatter([op["recall"]], [op["precision"]], color="black", zorder=5,
               label=f"locked op (tau={TAU_ON:.2f})")
    ax.annotate(f"P={op['precision']:.2f}\nR={op['recall']:.2f}",
                (op["recall"], op["precision"]), textcoords="offset points", xytext=(-58, -6), fontsize=8)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Far-event advisory: precision-recall frontier")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig1_pr_frontier.png", dpi=160); plt.close(fig)


def fig_calibration():
    cal = pd.read_csv(B / "raw_tables" / "calibration_bins.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(cal["mean_score"], cal["event_rate"], "o-", color="tab:green", lw=1.8)
    for key, lab, c in [("watch_enter", "watch", "gold"), ("tau_on", "alarm (tau_on)", "darkorange"),
                        ("high_confidence", "high-conf", "red")]:
        ax.axvline(ST[key], ls="--", color=c, lw=1.2, label=f"{lab}={ST[key]:.2f}")
    ax.set_xlabel("far-event score (mean per bin)")
    ax.set_ylabel("empirical event rate (calibrated confidence)")
    ax.set_title("Calibration / reliability of the far-event score")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig2_calibration.png", dpi=160); plt.close(fig)


def fig_per_regime():
    sc = pd.read_csv(B / "supervisory_scenario_table.csv")
    sc = sc.sort_values("count", ascending=False)
    x = np.arange(len(sc)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w, sc["precision"], w, label="precision", color="tab:blue")
    ax.bar(x, sc["recall"], w, label="recall", color="tab:green")
    ax.bar(x + w, sc["fpr"], w, label="FPR", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels(sc["scenario"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Advisory performance by programmatic regime (test)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    for i, n in enumerate(sc["count"]):
        ax.annotate(f"n={int(n)}\nbase={sc['base_rate'].iloc[i]:.2f}", (i, 1.0),
                    ha="center", va="top", fontsize=6, color="dimgray")
    fig.tight_layout(); fig.savefig(FIG / "fig3_per_regime.png", dpi=160); plt.close(fig)


def main():
    fig_pr_frontier()
    fig_calibration()
    fig_per_regime()
    print("figures written:")
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.relative_to(B))


if __name__ == "__main__":
    main()
