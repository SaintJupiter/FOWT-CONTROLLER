#!/usr/bin/env python3
"""Rank existing matched runs by safety-tail cost of blind economy actions.

This is a read-only evidence audit.  It scans existing result CSVs that already
record matched closed-only and primary timeseries paths, recomputes posture
exposure metrics from the traces, and writes the strongest examples where a
blind relaxed/economy action increases high-posture exposure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/saintyoung/Desktop/FOWT-CONTROLLER-main")
SEARCH_ROOT = ROOT / "outputs/wind_prediction/regime_conditioned_policy_development_v1"
OUT = ROOT / "FIXED_PUMP_SAVING_RESULTS_20260530/W1_prediction_necessity_audit"


def _resolve(pathish: str) -> Path:
    p = Path(str(pathish))
    return p if p.is_absolute() else ROOT / p


def _axis(df: pd.DataFrame) -> pd.Series:
    return df[["pitch_deg", "roll_deg"]].abs().max(axis=1)


def _dt(df: pd.DataFrame) -> float:
    t = df["t_s"].to_numpy(dtype=float)
    if len(t) < 2:
        return 1.0
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 1.0


def _pump(df: pd.DataFrame) -> float:
    if "pump_total_rate_m3_min" not in df.columns:
        return float("nan")
    return float((df["pump_total_rate_m3_min"].astype(float) * _dt(df) / 60.0).sum())


def _fallback_seconds(df: pd.DataFrame) -> float:
    for col in (
        "preview_primary_safety_fallback",
        "primary_safety_fallback",
        "preview_primary_safety_hard_active",
    ):
        if col in df.columns:
            return float((df[col].fillna(0).astype(float) > 0).sum() * _dt(df))
    return 0.0


def _metrics(closed: pd.DataFrame, primary: pd.DataFrame) -> dict[str, float]:
    ca = _axis(closed)
    pa = _axis(primary)
    dt = _dt(closed)
    out: dict[str, float] = {}
    for threshold in (3.0, 4.0, 5.0):
        key = str(int(threshold))
        out[f"closed_time_gt{key}_s"] = float((ca > threshold).sum() * dt)
        out[f"primary_time_gt{key}_s"] = float((pa > threshold).sum() * dt)
        out[f"d_time_gt{key}_s"] = out[f"primary_time_gt{key}_s"] - out[f"closed_time_gt{key}_s"]
        out[f"closed_area_gt{key}_deg_s"] = float(np.clip(ca - threshold, 0, None).sum() * dt)
        out[f"primary_area_gt{key}_deg_s"] = float(np.clip(pa - threshold, 0, None).sum() * dt)
        out[f"d_area_gt{key}_deg_s"] = out[f"primary_area_gt{key}_deg_s"] - out[f"closed_area_gt{key}_deg_s"]
    out["closed_axis_p95"] = float(np.percentile(ca, 95))
    out["primary_axis_p95"] = float(np.percentile(pa, 95))
    out["d_axis_p95"] = out["primary_axis_p95"] - out["closed_axis_p95"]
    out["closed_axis_max"] = float(ca.max())
    out["primary_axis_max"] = float(pa.max())
    out["d_axis_max"] = out["primary_axis_max"] - out["closed_axis_max"]
    cp = _pump(closed)
    pp = _pump(primary)
    out["closed_pump_m3"] = cp
    out["primary_pump_m3"] = pp
    out["saving_pct"] = 100.0 * (cp - pp) / cp if np.isfinite(cp) and cp > 0 else 0.0
    out["primary_fallback_s"] = _fallback_seconds(primary)
    return out


def _read_candidate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for csv_path in SEARCH_ROOT.rglob("*.csv"):
        name = csv_path.name
        if name not in {"casebook_summary.csv", "hard_gate_case_audit.csv"}:
            continue
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception:
            continue
        if not {"closed_timeseries", "primary_timeseries"}.issubset(header.columns):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        for _, row in df.iterrows():
            closed_path = _resolve(str(row.get("closed_timeseries", "")))
            primary_path = _resolve(str(row.get("primary_timeseries", "")))
            if not closed_path.exists() or not primary_path.exists():
                continue
            label = str(row.get("label", ""))
            case_id = str(row.get("case_id", closed_path.stem))
            rows.append(
                {
                    "source_csv": str(csv_path.relative_to(ROOT)),
                    "case_id": case_id,
                    "timestamp": row.get("timestamp", ""),
                    "label": label,
                    "closed_timeseries": str(closed_path.relative_to(ROOT)),
                    "primary_timeseries": str(primary_path.relative_to(ROOT)),
                }
            )
    return rows


def _plot_case(row: pd.Series, out_path: Path) -> None:
    closed = pd.read_csv(ROOT / row["closed_timeseries"])
    primary = pd.read_csv(ROOT / row["primary_timeseries"])
    t = closed["t_s"].to_numpy(dtype=float) / 3600.0
    ca = _axis(closed)
    pa = _axis(primary)
    cp = np.cumsum(closed["pump_total_rate_m3_min"].astype(float).to_numpy() * _dt(closed) / 60.0)
    pp = np.cumsum(primary["pump_total_rate_m3_min"].astype(float).to_numpy() * _dt(primary) / 60.0)

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.4), sharex=True)
    fig.suptitle(f"{row['case_id']} | blind economy safety-tail example", fontsize=13, fontweight="bold")

    axes[0].plot(t, closed["pitch_deg"], color="#1f77b4", lw=1.4, label="closed pitch")
    axes[0].plot(t, primary["pitch_deg"], color="#d62728", lw=1.4, label="blind relaxed pitch")
    axes[0].axhline(5, color="#444", lw=0.8, ls="--")
    axes[0].axhline(-5, color="#444", lw=0.8, ls="--")
    axes[0].set_ylabel("pitch (deg)")
    axes[0].legend(ncol=2, loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, closed["roll_deg"], color="#1f77b4", lw=1.4, label="closed roll")
    axes[1].plot(t, primary["roll_deg"], color="#d62728", lw=1.4, label="blind relaxed roll")
    axes[1].axhline(5, color="#444", lw=0.8, ls="--")
    axes[1].axhline(-5, color="#444", lw=0.8, ls="--")
    axes[1].set_ylabel("roll (deg)")
    axes[1].legend(ncol=2, loc="upper right")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, cp, color="#1f77b4", lw=1.8, label="closed cumulative pump")
    axes[2].plot(t, pp, color="#d62728", lw=1.8, label="blind relaxed cumulative pump")
    axes[2].set_ylabel("pump (m3)")
    axes[2].set_xlabel("time (h)")
    axes[2].legend(ncol=2, loc="upper left")
    axes[2].grid(True, alpha=0.25)

    subtitle = (
        f"saving {row['saving_pct']:.1f}% | "
        f"dT>5 {row['d_time_gt5_s']:.0f}s | "
        f"dArea>4 {row['d_area_gt4_deg_s']:.0f} deg*s | "
        f"dP95 {row['d_axis_p95']:+.2f} deg"
    )
    axes[0].text(0.01, 1.04, subtitle, transform=axes[0].transAxes, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = _read_candidate_rows()
    scored: list[dict[str, object]] = []
    for row in candidates:
        try:
            closed = pd.read_csv(ROOT / str(row["closed_timeseries"]))
            primary = pd.read_csv(ROOT / str(row["primary_timeseries"]))
            if not {"pitch_deg", "roll_deg", "t_s"}.issubset(closed.columns) or not {
                "pitch_deg",
                "roll_deg",
                "t_s",
            }.issubset(primary.columns):
                continue
            scored.append({**row, **_metrics(closed, primary)})
        except Exception as exc:
            scored.append({**row, "error": repr(exc)})

    df = pd.DataFrame([r for r in scored if "error" not in r])
    if df.empty:
        raise SystemExit("No matched timeseries rows found.")
    df["tail_score"] = (
        df["d_time_gt5_s"].clip(lower=0) * 10.0
        + df["d_area_gt4_deg_s"].clip(lower=0)
        + df["primary_fallback_s"].clip(lower=0) * 5.0
    )
    df = df.sort_values(["tail_score", "d_time_gt5_s", "d_area_gt4_deg_s"], ascending=False)
    out_csv = OUT / "safety_tail_hunt_v1.csv"
    df.to_csv(out_csv, index=False)

    strong = df.head(20).copy()
    strong.to_csv(OUT / "safety_tail_hunt_top20_v1.csv", index=False)
    for i, (_, row) in enumerate(strong.head(4).iterrows(), start=1):
        safe_case = str(row["case_id"]).replace("/", "_").replace(" ", "_")
        _plot_case(row, OUT / f"F_safety_tail_hunt_{i}_{safe_case}.png")

    w1_boundary = df[df["source_csv"].str.contains("d1_boundary_reversal_reintensification_6h_v1", regex=False)]
    note = [
        "# Safety Tail Hunt v1",
        "",
        "Read-only scan over existing matched closed-only vs primary traces.",
        "",
        f"- matched rows scanned: {len(df)}",
        f"- top safety-tail CSV: `{out_csv.relative_to(ROOT)}`",
        "",
        "## Strongest existing blind-relaxation tail examples",
        "",
        "| rank | source | case | saving % | dT>5 s | dArea>4 deg*s | dP95 deg | fallback s |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(strong.head(10).iterrows(), start=1):
        note.append(
            f"| {rank} | `{Path(row['source_csv']).parent.name}` | {row['case_id']} | "
            f"{row['saving_pct']:.1f} | {row['d_time_gt5_s']:.0f} | "
            f"{row['d_area_gt4_deg_s']:.0f} | {row['d_axis_p95']:+.2f} | "
            f"{row['primary_fallback_s']:.0f} |"
        )
    note += [
        "",
        "## W1 boundary subset",
        "",
        "This is the cleaner prediction-necessity subset: prediction-gated deployment would abstain here, while blind deadband is the counterfactual.",
    ]
    if not w1_boundary.empty:
        note += [
            f"- cases: {len(w1_boundary)}",
            f"- aggregate dT>5: {w1_boundary['d_time_gt5_s'].sum():.0f}s",
            f"- worst dT>5: {w1_boundary['d_time_gt5_s'].max():.0f}s",
            f"- aggregate dArea>4: {w1_boundary['d_area_gt4_deg_s'].sum():.1f} deg*s",
        ]
    note += [
        "",
        "Interpretation: the very largest tail cases often come from older unsafe/freezing mechanisms and should be used as cautionary context, not as W1 evidence. The W1 boundary subset is smaller but cleaner for proving prediction-gated abstention.",
        "",
    ]
    (OUT / "safety_tail_hunt_v1.md").write_text("\n".join(note), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {OUT / 'safety_tail_hunt_v1.md'}")


if __name__ == "__main__":
    main()
