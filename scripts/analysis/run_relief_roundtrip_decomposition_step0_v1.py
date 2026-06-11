#!/usr/bin/env python3
"""Step 0 read-only round-trip decomposition for relief-economy pump saving.

The aim is to decide whether the next lever should be partial target capping
(large pump-up-then-unwind cycles) or relief-aware deadband widening (small
high-frequency chatter). It reads existing A0/A1 runs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1"
RUN_A0 = BASE / "runs/A0_learned_v16"
RUN_A1 = BASE / "runs/A1_learned_near_envelope_medium_axis_guard"
MANIFEST = BASE / "locked_casebook_manifest.csv"
DELTA_TABLE = BASE / "learned_a1_per_case_delta_table.csv"
OUT = REPO / "outputs/wind_prediction/relief_roundtrip_decomposition_step0_v1"

M3_PER_KG = 1.0 / 1000.0
LARGE_CYCLE_M3 = 50.0
CHATTER_CYCLE_M3 = 10.0
SHORT_INTERVAL_S = 1800.0


@dataclass
class Segment:
    case: str
    tank: str
    direction: int
    start_s: float
    end_s: float
    motion_m3: float
    safe_fraction: float
    relief_adjacent_fraction: float
    addressable_fraction: float
    floor_fraction: float


def _dirs() -> dict[str, Path]:
    dirs = {"out": OUT, "raw": OUT / "raw_tables", "paper": OUT / "paper_ready", "debug": OUT / "debug"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _case_from_path(path: Path) -> str:
    suffix = "_prediction_primary_econ_timeseries.csv"
    return path.name[: -len(suffix)]


def _read_case(run: Path, case: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_path = run / "timeseries" / f"{case}_prediction_primary_econ_timeseries.csv"
    log_path = run / "planner_logs" / f"{case}_prediction_primary_econ_planner_log.csv"
    ts_cols = [
        "t_s",
        "pitch_deg",
        "roll_deg",
        "pump_total_rate_m3_min",
        "tank1_kg",
        "tank2_kg",
        "tank3_kg",
    ]
    log_cols = [
        "current_time_s",
        "raw_pressure_block0_norm",
        "raw_pressure_block1_norm",
        "raw_pressure_block2_norm",
        "reactive_floor_active",
        "relief_envelope_active",
        "relief_envelope_reason",
    ]
    ts_avail = pd.read_csv(ts_path, nrows=1).columns
    log_avail = pd.read_csv(log_path, nrows=1).columns
    ts = pd.read_csv(ts_path, usecols=[c for c in ts_cols if c in ts_avail])
    log = pd.read_csv(log_path, usecols=[c for c in log_cols if c in log_avail])
    return ts, log


def _log_flags_at_seconds(ts: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    t = pd.to_numeric(log["current_time_s"], errors="coerce").to_numpy(dtype=float)
    b0 = pd.to_numeric(log.get("raw_pressure_block0_norm", pd.Series(np.nan, index=log.index)), errors="coerce").to_numpy(dtype=float)
    b1 = pd.to_numeric(log.get("raw_pressure_block1_norm", pd.Series(np.nan, index=log.index)), errors="coerce").to_numpy(dtype=float)
    b2 = pd.to_numeric(log.get("raw_pressure_block2_norm", pd.Series(np.nan, index=log.index)), errors="coerce").to_numpy(dtype=float)
    floor = pd.to_numeric(log.get("reactive_floor_active", pd.Series(0, index=log.index)), errors="coerce").fillna(0).to_numpy(dtype=float)
    if len(t) == 0:
        out = ts[["t_s"]].copy()
        out["near_relief"] = False
        out["floor_active"] = False
        return out
    # Hold each 600s planner row until the next one.
    idx = np.searchsorted(t, ts["t_s"].to_numpy(dtype=float), side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)
    near_max = np.nanmax(np.vstack([b0[idx], b1[idx], b2[idx]]), axis=0)
    near_relief = (b0[idx] <= 0.65) & (b1[idx] <= 0.75) & (b2[idx] <= 0.85) & (near_max <= 0.90)
    out = ts[["t_s", "pitch_deg", "roll_deg"]].copy()
    out["near_relief"] = near_relief
    out["floor_active"] = floor[idx] > 0.5
    out["max_axis"] = np.maximum(out["pitch_deg"].abs(), out["roll_deg"].abs())
    out["safe_posture"] = out["max_axis"] < 4.65
    out["addressable"] = out["near_relief"] & out["safe_posture"] & (~out["floor_active"])
    return out


def _segments_for_tank(case: str, tank: str, ts: pd.DataFrame, flags: pd.DataFrame) -> list[Segment]:
    x = pd.to_numeric(ts[tank], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(ts["t_s"], errors="coerce").to_numpy(dtype=float)
    dx = np.diff(x, prepend=x[0])
    # Ignore tiny numerical drift.
    sign = np.where(dx > 1e-6, 1, np.where(dx < -1e-6, -1, 0))
    segments: list[Segment] = []
    active_sign = 0
    start_idx = 0
    motion = 0.0

    def close(end_idx: int) -> None:
        nonlocal active_sign, start_idx, motion
        if active_sign == 0 or motion <= 1e-9 or end_idx <= start_idx:
            return
        f = flags.iloc[start_idx : end_idx + 1]
        segments.append(
            Segment(
                case=case,
                tank=tank,
                direction=int(active_sign),
                start_s=float(t[start_idx]),
                end_s=float(t[end_idx]),
                motion_m3=float(motion * M3_PER_KG),
                safe_fraction=float(f["safe_posture"].mean()) if len(f) else 0.0,
                relief_adjacent_fraction=float(f["near_relief"].mean()) if len(f) else 0.0,
                addressable_fraction=float(f["addressable"].mean()) if len(f) else 0.0,
                floor_fraction=float(f["floor_active"].mean()) if len(f) else 0.0,
            )
        )

    for i in range(1, len(sign)):
        s = int(sign[i])
        if s == 0:
            continue
        if active_sign == 0:
            active_sign = s
            start_idx = i
            motion = abs(dx[i])
            continue
        if s == active_sign:
            motion += abs(dx[i])
            continue
        close(i - 1)
        active_sign = s
        start_idx = i
        motion = abs(dx[i])
    close(len(sign) - 1)
    return segments


def _case_segments(case: str) -> tuple[pd.DataFrame, dict[str, float]]:
    ts, log = _read_case(RUN_A0, case)
    flags = _log_flags_at_seconds(ts, log)
    segs: list[Segment] = []
    for tank in ("tank1_kg", "tank2_kg", "tank3_kg"):
        if tank in ts.columns:
            segs.extend(_segments_for_tank(case, tank, ts, flags))
    rows = [s.__dict__ for s in segs]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "case",
                "tank",
                "direction",
                "start_s",
                "end_s",
                "motion_m3",
                "safe_fraction",
                "relief_adjacent_fraction",
                "addressable_fraction",
                "floor_fraction",
            ]
        )
    pump_total = float(ts["pump_total_rate_m3_min"].abs().sum() / 60.0)
    addressable_pump = float(
        (ts["pump_total_rate_m3_min"].abs().to_numpy(dtype=float) * flags["addressable"].to_numpy(dtype=bool)).sum() / 60.0
    )
    meta = {
        "A0_pump_m3": pump_total,
        "A0_addressable_pump_m3": addressable_pump,
        "A0_addressable_seconds": float(flags["addressable"].sum()),
        "A0_near_relief_seconds": float(flags["near_relief"].sum()),
        "A0_safe_posture_seconds": float(flags["safe_posture"].sum()),
    }
    return df, meta


def _cycle_pairs(seg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if seg.empty:
        return pd.DataFrame()
    for (case, tank), g in seg.groupby(["case", "tank"]):
        g = g.sort_values("start_s").reset_index(drop=True)
        for i in range(len(g) - 1):
            a = g.iloc[i]
            b = g.iloc[i + 1]
            if int(a["direction"]) == int(b["direction"]):
                continue
            interval = float(b["start_s"] - a["end_s"])
            reversible = min(float(a["motion_m3"]), float(b["motion_m3"]))
            addressable = max(float(a["addressable_fraction"]), float(b["addressable_fraction"]))
            relief = max(float(a["relief_adjacent_fraction"]), float(b["relief_adjacent_fraction"]))
            safe = max(float(a["safe_fraction"]), float(b["safe_fraction"]))
            floor = max(float(a["floor_fraction"]), float(b["floor_fraction"]))
            if reversible >= LARGE_CYCLE_M3:
                kind = "large_cycle"
            elif reversible <= CHATTER_CYCLE_M3 and interval <= SHORT_INTERVAL_S:
                kind = "chatter"
            else:
                kind = "mid_cycle"
            rows.append(
                {
                    "case": case,
                    "tank": tank,
                    "first_start_s": float(a["start_s"]),
                    "second_end_s": float(b["end_s"]),
                    "interval_s": interval,
                    "reversible_m3": reversible,
                    "kind": kind,
                    "safe_fraction": safe,
                    "relief_adjacent_fraction": relief,
                    "addressable_fraction": addressable,
                    "floor_fraction": floor,
                    "addressable_proxy": int((addressable >= 0.25) and (floor < 0.10)),
                }
            )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    d = df if max_rows is None else df.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    dirs = _dirs()
    manifest = pd.read_csv(MANIFEST)
    delta = pd.read_csv(DELTA_TABLE)
    delta = delta[["case", "A1_minus_A0_pump_m3", "A1_minus_A0_time5_s", "A1_minus_A0_idle5_s"]]

    all_segments = []
    meta_rows = []
    for path in sorted((RUN_A0 / "timeseries").glob("*_prediction_primary_econ_timeseries.csv")):
        case = _case_from_path(path)
        seg, meta = _case_segments(case)
        all_segments.append(seg)
        meta["case"] = case
        meta_rows.append(meta)
    segments = pd.concat(all_segments, ignore_index=True) if all_segments else pd.DataFrame()
    pairs = _cycle_pairs(segments)
    meta = pd.DataFrame(meta_rows)

    case_info = manifest.copy()
    case_info["case"] = case_info["locked_index"].astype(str).str.zfill(2) + "_" + case_info["case_id"].astype(str) + "_" + case_info["timestamp"].astype(str).str.replace("-", "", regex=False).str.replace(":", "", regex=False).str.replace(" ", "_", regex=False)
    # File case ids include label-derived names, so use a loose match fallback from actual meta.
    source_map = {}
    for _, m in manifest.iterrows():
        locked = str(m["locked_index"]).zfill(2)
        source_map[locked] = {
            "case_source": m.get("case_source", ""),
            "source_groups": m.get("source_groups", ""),
            "label": m.get("label", ""),
        }
    for df in (segments, pairs, meta):
        if not df.empty:
            df["locked_index"] = df["case"].astype(str).str.split("_", n=1).str[0]
            df["case_source"] = df["locked_index"].map(lambda x: source_map.get(str(x), {}).get("case_source", ""))
            df["source_groups"] = df["locked_index"].map(lambda x: source_map.get(str(x), {}).get("source_groups", ""))
            df["label"] = df["locked_index"].map(lambda x: source_map.get(str(x), {}).get("label", ""))

    case_summary = meta.merge(delta, on="case", how="left")
    if not pairs.empty:
        pair_case = (
            pairs.groupby("case", as_index=False)
            .agg(
                reversible_m3=("reversible_m3", "sum"),
                addressable_reversible_m3=("reversible_m3", lambda s: float(s[pairs.loc[s.index, "addressable_proxy"].astype(bool)].sum())),
                large_cycle_m3=("reversible_m3", lambda s: float(s[pairs.loc[s.index, "kind"].eq("large_cycle")].sum())),
                chatter_m3=("reversible_m3", lambda s: float(s[pairs.loc[s.index, "kind"].eq("chatter")].sum())),
                mid_cycle_m3=("reversible_m3", lambda s: float(s[pairs.loc[s.index, "kind"].eq("mid_cycle")].sum())),
                cycle_count=("reversible_m3", "size"),
                addressable_cycle_count=("addressable_proxy", "sum"),
            )
        )
        case_summary = case_summary.merge(pair_case, on="case", how="left")
    for c in ["reversible_m3", "addressable_reversible_m3", "large_cycle_m3", "chatter_m3", "mid_cycle_m3", "cycle_count", "addressable_cycle_count"]:
        if c in case_summary:
            case_summary[c] = case_summary[c].fillna(0.0)

    if pairs.empty:
        kind_summary = pd.DataFrame()
        addr_summary = pd.DataFrame()
    else:
        kind_summary = pairs.groupby("kind", as_index=False).agg(
            cycle_count=("reversible_m3", "size"),
            reversible_m3=("reversible_m3", "sum"),
            addressable_reversible_m3=("reversible_m3", lambda s: float(s[pairs.loc[s.index, "addressable_proxy"].astype(bool)].sum())),
        )
        addr_summary = pairs.groupby(["kind", "addressable_proxy"], as_index=False).agg(
            cycle_count=("reversible_m3", "size"),
            reversible_m3=("reversible_m3", "sum"),
        )

    # Concentration of current savings.
    saving_cases = case_summary.copy()
    saving_cases["pump_saved_m3"] = -saving_cases["A1_minus_A0_pump_m3"].fillna(0.0)
    total_saved = float(saving_cases["pump_saved_m3"].clip(lower=0).sum())
    top_saved = saving_cases.sort_values("pump_saved_m3", ascending=False)
    top1_share = float(top_saved["pump_saved_m3"].iloc[0] / total_saved) if total_saved > 1e-9 and len(top_saved) else 0.0
    top5_share = float(top_saved["pump_saved_m3"].head(5).sum() / total_saved) if total_saved > 1e-9 else 0.0

    summary_rows = [
        {"metric": "total_current_saved_m3", "value": total_saved},
        {"metric": "saving_top1_share", "value": top1_share},
        {"metric": "saving_top5_share", "value": top5_share},
        {"metric": "A0_total_pump_m3", "value": float(case_summary["A0_pump_m3"].sum())},
        {"metric": "A0_addressable_pump_m3", "value": float(case_summary["A0_addressable_pump_m3"].sum())},
        {"metric": "cycle_reversible_m3", "value": float(pairs["reversible_m3"].sum()) if not pairs.empty else 0.0},
        {"metric": "addressable_cycle_reversible_m3", "value": float(pairs.loc[pairs["addressable_proxy"].astype(bool), "reversible_m3"].sum()) if not pairs.empty else 0.0},
    ]
    if not kind_summary.empty:
        for _, r in kind_summary.iterrows():
            summary_rows.append({"metric": f"{r['kind']}_reversible_m3", "value": float(r["reversible_m3"])})
            summary_rows.append({"metric": f"{r['kind']}_addressable_reversible_m3", "value": float(r["addressable_reversible_m3"])})
    summary = pd.DataFrame(summary_rows)

    segments.to_csv(dirs["raw"] / "pump_direction_segments.csv", index=False)
    pairs.to_csv(dirs["raw"] / "pump_reversal_cycle_pairs.csv", index=False)
    case_summary.to_csv(dirs["raw"] / "roundtrip_case_decomposition.csv", index=False)
    kind_summary.to_csv(dirs["raw"] / "roundtrip_kind_summary.csv", index=False)
    addr_summary.to_csv(dirs["raw"] / "roundtrip_addressable_summary.csv", index=False)
    summary.to_csv(dirs["raw"] / "step0_summary.csv", index=False)
    summary.to_csv(dirs["out"] / "step0_summary.csv", index=False)

    # Decide next lever from addressable composition.
    large_addr = float(summary.loc[summary["metric"].eq("large_cycle_addressable_reversible_m3"), "value"].sum())
    chatter_addr = float(summary.loc[summary["metric"].eq("chatter_addressable_reversible_m3"), "value"].sum())
    if large_addr >= chatter_addr * 1.25 and large_addr > 50:
        next_lever = "partial_cap_first"
    elif chatter_addr > large_addr * 1.25 and chatter_addr > 50:
        next_lever = "deadband_first"
    else:
        next_lever = "cap_first_with_deadband_as_secondary"

    top_cases = top_saved[["case", "case_source", "pump_saved_m3", "reversible_m3", "addressable_reversible_m3", "large_cycle_m3", "chatter_m3"]].head(10)
    high_headroom = case_summary.sort_values("addressable_reversible_m3", ascending=False)[
        ["case", "case_source", "A0_pump_m3", "addressable_reversible_m3", "large_cycle_m3", "chatter_m3", "A1_minus_A0_pump_m3"]
    ].head(12)

    md = f"""# Relief Round-Trip Decomposition Step 0

## Purpose

Read-only decomposition of the remaining round-trip/chatter pool in the locked
53-case A0/A1 runs. This decides whether the next pump-saving lever should be
partial target capping or relief-aware deadband widening.

## Key Summary

{_md_table(summary)}

## Current Saving Concentration

- top-1 saving share: {top1_share:.1%}
- top-5 saving share: {top5_share:.1%}

{_md_table(top_cases)}

## Round-Trip Type Summary

{_md_table(kind_summary)}

## High Addressable Headroom Cases

{_md_table(high_headroom)}

## Recommendation

Recommended next lever: **{next_lever}**.

- If addressable large-cycle reversible volume dominates, implement a conservative
  partial-refresh cap first.
- If addressable chatter dominates, implement relief-aware economy deadband
  widening first.
- If neither addressable pool is large, preserve the current 6.55% result and
  avoid tuning on the locked holdout.
"""
    for path in (dirs["out"] / "step0_roundtrip_decomposition.md", dirs["paper"] / "step0_roundtrip_decomposition.md"):
        path.write_text(md, encoding="utf-8")

    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
