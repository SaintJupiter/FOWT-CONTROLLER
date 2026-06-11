#!/usr/bin/env python3
"""Pump restart oracle analysis.

This script is intentionally read-only: it scans existing closed_only
timeseries CSVs and estimates how many pump restarts were theoretically
avoidable with perfect future knowledge of the mass-domain error.

The oracle answers a narrow question before any supervisor implementation:
are enough restarts small, transient, and self-relieving to justify a
forecast-aware restart veto?
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs" / "wind_prediction"
DEFAULT_OUT_DIR = DEFAULT_INPUT_ROOT / "pump_restart_oracle"
TANKS = (1, 2, 3)


def parse_horizons(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(float(item)))
    return sorted(set(out))


def discover_closed_only_files(input_root: Path) -> list[Path]:
    files = sorted(input_root.rglob("*closed_only_timeseries.csv"))
    return [p for p in files if p.is_file()]


def case_key(path: Path) -> str:
    name = path.stem
    if name.endswith("_timeseries"):
        name = name[: -len("_timeseries")]
    for prefix in ("onset_", "decay_", "signflip_", "lowrisk_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.endswith("_closed_only"):
        name = name[: -len("_closed_only")]
    return name


def dedupe_files(files: list[Path]) -> list[Path]:
    """Prefer explicit pump-first scenario outputs, then keep one file per time key."""
    preferred_dirs = {
        "planner_a2_pump_first_suppression",
        "planner_a2_pump_first_suppression_full",
        "planner_a2_economic_expanded",
        "planner_a2_onset",
        "planner_a2_smoke",
    }

    def score(path: Path) -> tuple[int, int, str]:
        parent = path.parent.name
        explicit = int(path.name.lower().startswith(("onset_", "decay_", "signflip_", "lowrisk_")))
        preferred = int(parent in preferred_dirs)
        return (explicit, preferred, str(path))

    chosen: dict[str, Path] = {}
    for path in sorted(files, key=score, reverse=True):
        key = case_key(path)
        if key not in chosen:
            chosen[key] = path
    return sorted(chosen.values())


def scenario_group(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    for prefix in ("onset", "decay", "signflip", "lowrisk"):
        if name.startswith(prefix + "_"):
            return prefix
    if "onset" in parent:
        return "onset_existing"
    if "economic_expanded" in parent:
        return "expanded_existing"
    if "smoke" in parent:
        return "smoke_existing"
    return "other_existing"


def read_timeseries(path: Path) -> pd.DataFrame | None:
    required = [
        "t_s",
        "pitch_deg",
        "roll_deg",
        "pump_total_rate_m3_min",
    ]
    for tank in TANKS:
        required.extend(
            [
                f"err_tank{tank}_kg",
                f"pump_latched{tank}",
                f"pump_rate{tank}_m3min",
            ]
        )
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns)
    missing = [c for c in required if c not in cols]
    if missing:
        print(f"[SKIP] {path}: missing columns {missing[:5]}")
        return None
    return pd.read_csv(path, usecols=required)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def median_dt_s(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 1.0
    t = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(dtype=float)
    diffs = np.diff(t[np.isfinite(t)])
    diffs = diffs[diffs > 1e-9]
    return float(np.median(diffs)) if len(diffs) else 1.0


def event_pump_work_m3(
    rate: np.ndarray,
    latched: np.ndarray,
    start_idx: int,
    dt_s: float,
) -> tuple[float, int]:
    end_idx = len(latched)
    for j in range(start_idx + 1, len(latched)):
        if latched[j] <= 0:
            end_idx = j
            break
    event_rate = np.abs(rate[start_idx:end_idx])
    return float(np.sum(event_rate) * dt_s / 60.0), int(end_idx - start_idx)


def future_metrics(
    err_all: np.ndarray,
    pitch_abs_all: np.ndarray,
    roll_abs_all: np.ndarray,
    idx: int,
    current_err: float,
    horizon_min: int,
    dt_s: float,
    relief_ratio: float,
    max_defer_err_kg: float,
) -> dict[str, float | int]:
    steps = max(1, int(round(horizon_min * 60.0 / max(dt_s, 1e-9))))
    end = min(len(err_all), idx + steps + 1)
    err = err_all[idx:end]
    pitch = pitch_abs_all[idx:end]
    roll = roll_abs_all[idx:end]
    abs_err = np.abs(err)
    current_abs = abs(float(current_err))
    min_abs = float(np.min(abs_err)) if len(abs_err) else current_abs
    max_abs = float(np.max(abs_err)) if len(abs_err) else current_abs
    end_abs = float(abs_err[-1]) if len(abs_err) else current_abs
    sign_flip = int(np.any(err * float(current_err) < 0.0)) if len(err) else 0
    relief = int(min_abs <= float(relief_ratio) * current_abs)
    debt_ok = int(max_abs <= float(max_defer_err_kg))
    qualifies = int((relief or sign_flip) and debt_ok)
    return {
        f"h{horizon_min}_end_abs_err_kg": end_abs,
        f"h{horizon_min}_min_abs_err_kg": min_abs,
        f"h{horizon_min}_max_abs_err_kg": max_abs,
        f"h{horizon_min}_relief": relief,
        f"h{horizon_min}_sign_flip": sign_flip,
        f"h{horizon_min}_debt_ok": debt_ok,
        f"h{horizon_min}_qualifies": qualifies,
        f"h{horizon_min}_pitch_p95": float(np.percentile(pitch, 95)) if len(pitch) else 0.0,
        f"h{horizon_min}_roll_p95": float(np.percentile(roll, 95)) if len(roll) else 0.0,
    }


def classify_event(row: dict[str, Any], horizons_min: list[int], args: argparse.Namespace) -> str:
    current_abs = float(row["current_abs_err_kg"])
    if current_abs <= float(args.base_restart_kg):
        return "current_err_at_or_below_base_restart"
    if current_abs > float(args.max_veto_err_kg):
        return "current_err_above_veto_window"

    any_relief_or_flip = False
    any_debt_fail_candidate = False
    for h in horizons_min:
        relief = bool(row.get(f"h{h}_relief", 0))
        flip = bool(row.get(f"h{h}_sign_flip", 0))
        debt_ok = bool(row.get(f"h{h}_debt_ok", 0))
        if relief or flip:
            any_relief_or_flip = True
            if debt_ok:
                row["qualified_horizon_min"] = h
                return "safe_veto_oracle"
            any_debt_fail_candidate = True

    if any_debt_fail_candidate:
        return "future_max_err_exceeds_debt_cap"
    if not any_relief_or_flip:
        return "future_not_relieving_or_reversing"
    return "not_safe_veto"


def analyze_file(path: Path, horizons_min: list[int], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = read_timeseries(path)
    if df is None or df.empty:
        return [], {
            "file": str(path),
            "group": scenario_group(path),
            "case": path.stem.replace("_timeseries", ""),
            "skipped": 1,
        }

    dt_s = median_dt_s(df)
    duration_h = float((safe_float(df["t_s"].iloc[-1]) - safe_float(df["t_s"].iloc[0]) + dt_s) / 3600.0)
    pump_total = np.abs(pd.to_numeric(df["pump_total_rate_m3_min"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    total_pump_work_m3 = float(np.sum(pump_total) * dt_s / 60.0)
    pitch_abs = np.abs(pd.to_numeric(df["pitch_deg"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    roll_abs = np.abs(pd.to_numeric(df["roll_deg"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    t_s = pd.to_numeric(df["t_s"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    events: list[dict[str, Any]] = []
    tank_arrays: dict[int, dict[str, np.ndarray]] = {}
    for tank in TANKS:
        tank_arrays[tank] = {
            "latched": pd.to_numeric(df[f"pump_latched{tank}"], errors="coerce").fillna(0).to_numpy(dtype=int),
            "err": pd.to_numeric(df[f"err_tank{tank}_kg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float),
            "rate": pd.to_numeric(df[f"pump_rate{tank}_m3min"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        }

    for tank in TANKS:
        latched = tank_arrays[tank]["latched"]
        err_arr = tank_arrays[tank]["err"]
        rate_arr = tank_arrays[tank]["rate"]
        starts = np.where((latched[1:] > 0) & (latched[:-1] <= 0))[0] + 1
        for idx in starts:
            current_err = float(err_arr[idx])
            row: dict[str, Any] = {
                "file": str(path),
                "source_dir": path.parent.name,
                "group": scenario_group(path),
                "case": path.stem.replace("_timeseries", ""),
                "tank": tank,
                "idx": int(idx),
                "t_s": float(t_s[idx]),
                "dt_s": dt_s,
                "current_err_kg": current_err,
                "current_abs_err_kg": abs(current_err),
                "current_pitch_abs_deg": float(pitch_abs[idx]),
                "current_roll_abs_deg": float(roll_abs[idx]),
                "current_pump_total_rate_m3_min": float(pump_total[idx]),
                "qualified_horizon_min": 0,
            }
            work_m3, duration_steps = event_pump_work_m3(
                rate=rate_arr,
                latched=latched,
                start_idx=int(idx),
                dt_s=dt_s,
            )
            row["event_pump_work_m3"] = work_m3
            row["event_latched_duration_s"] = float(duration_steps * dt_s)
            for h in horizons_min:
                row.update(
                    future_metrics(
                        err_all=err_arr,
                        pitch_abs_all=pitch_abs,
                        roll_abs_all=roll_abs,
                        idx=int(idx),
                        current_err=current_err,
                        horizon_min=h,
                        dt_s=dt_s,
                        relief_ratio=float(args.relief_ratio),
                        max_defer_err_kg=float(args.max_defer_err_kg),
                    )
                )
            row["oracle_reason"] = classify_event(row, horizons_min=horizons_min, args=args)
            row["safe_veto_oracle"] = int(row["oracle_reason"] == "safe_veto_oracle")
            events.append(row)

    stops = 0
    for tank in TANKS:
        latched = tank_arrays[tank]["latched"]
        stops += int(np.sum((latched[1:] <= 0) & (latched[:-1] > 0)))

    safe_events = [e for e in events if e["safe_veto_oracle"]]
    summary = {
        "file": str(path),
        "source_dir": path.parent.name,
        "group": scenario_group(path),
        "case": path.stem.replace("_timeseries", ""),
        "skipped": 0,
        "duration_h": duration_h,
        "total_pump_work_m3": total_pump_work_m3,
        "restart_events": len(events),
        "stop_events": stops,
        "restart_events_per_h": float(len(events) / max(duration_h, 1e-9)),
        "safe_veto_events": len(safe_events),
        "safe_veto_rate_pct": float(len(safe_events) / max(len(events), 1) * 100.0),
        "event_pump_work_m3": float(sum(e["event_pump_work_m3"] for e in events)),
        "safe_veto_event_pump_work_m3": float(sum(e["event_pump_work_m3"] for e in safe_events)),
        "safe_veto_share_of_total_pump_work_pct": float(
            sum(e["event_pump_work_m3"] for e in safe_events) / max(total_pump_work_m3, 1e-9) * 100.0
        ),
    }
    return events, summary


def bucket_abs_err(x: float) -> str:
    if x <= 500:
        return "<=500"
    if x <= 1000:
        return "500-1000"
    if x <= 2000:
        return "1000-2000"
    if x <= 3000:
        return "2000-3000"
    return ">3000"


def aggregate(events_df: pd.DataFrame, files_df: pd.DataFrame) -> pd.DataFrame:
    if files_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group, fsub in files_df[files_df["skipped"] == 0].groupby("group", sort=True):
        esub = events_df[events_df["group"] == group] if not events_df.empty else pd.DataFrame()
        total_events = int(len(esub))
        safe = esub[esub["safe_veto_oracle"] > 0] if total_events else pd.DataFrame()
        reason_counts = Counter(esub["oracle_reason"].tolist()) if total_events else Counter()
        bucket_counts = Counter(esub["err_bucket"].tolist()) if total_events else Counter()
        rows.append(
            {
                "group": group,
                "files": int(len(fsub)),
                "duration_h": float(fsub["duration_h"].sum()),
                "total_pump_work_m3": float(fsub["total_pump_work_m3"].sum()),
                "restart_events": total_events,
                "restart_events_per_h": float(total_events / max(fsub["duration_h"].sum(), 1e-9)),
                "stop_events": int(fsub["stop_events"].sum()),
                "safe_veto_events": int(len(safe)),
                "safe_veto_rate_pct": float(len(safe) / max(total_events, 1) * 100.0),
                "safe_veto_event_pump_work_m3": float(safe["event_pump_work_m3"].sum()) if len(safe) else 0.0,
                "safe_veto_share_of_total_pump_work_pct": float(
                    (safe["event_pump_work_m3"].sum() if len(safe) else 0.0)
                    / max(fsub["total_pump_work_m3"].sum(), 1e-9)
                    * 100.0
                ),
                "bucket_500_1000": int(bucket_counts.get("500-1000", 0)),
                "bucket_1000_2000": int(bucket_counts.get("1000-2000", 0)),
                "bucket_2000_3000": int(bucket_counts.get("2000-3000", 0)),
                "bucket_gt_3000": int(bucket_counts.get(">3000", 0)),
                "reason_safe": int(reason_counts.get("safe_veto_oracle", 0)),
                "reason_too_large_now": int(reason_counts.get("current_err_above_veto_window", 0)),
                "reason_not_relieving": int(reason_counts.get("future_not_relieving_or_reversing", 0)),
                "reason_debt_cap": int(reason_counts.get("future_max_err_exceeds_debt_cap", 0)),
            }
        )
    return pd.DataFrame(rows)


def decision_text(overall: dict[str, float]) -> str:
    rate = float(overall.get("safe_veto_rate_pct", 0.0))
    share = float(overall.get("safe_veto_share_of_total_pump_work_pct", 0.0))
    if rate < 10.0:
        return (
            "STOP: safe-veto restart share is below 10%, so a pump-aware "
            "restart supervisor is unlikely to create meaningful pump-work savings."
        )
    if rate < 15.0:
        return (
            "LIGHTWEIGHT ONLY: safe-veto share is 10-15%; consider only a default-off "
            "dynamic restart threshold and short n=3/1h simulations."
        )
    if share < 5.0:
        return (
            "AMBIGUOUS: safe-veto event share is above 15%, but associated pump-work "
            "share is small; verify on short simulations before implementing supervisor v1."
        )
    return (
        "PROCEED TO SMALL SIM: safe-veto share is above 15% and pump-work share is "
        "nontrivial; implement a guarded supervisor only after n=3/1h checks."
    )


def write_report(out_dir: Path, events_df: pd.DataFrame, files_df: pd.DataFrame, agg_df: pd.DataFrame, args: argparse.Namespace) -> None:
    total_events = int(len(events_df))
    safe_events = int(events_df["safe_veto_oracle"].sum()) if total_events else 0
    total_pump = float(files_df.loc[files_df["skipped"] == 0, "total_pump_work_m3"].sum()) if not files_df.empty else 0.0
    safe_work = float(events_df.loc[events_df["safe_veto_oracle"] > 0, "event_pump_work_m3"].sum()) if total_events else 0.0
    overall = {
        "safe_veto_rate_pct": float(safe_events / max(total_events, 1) * 100.0),
        "safe_veto_share_of_total_pump_work_pct": float(safe_work / max(total_pump, 1e-9) * 100.0),
    }

    lines = [
        "# Pump restart oracle",
        "",
        "Read-only analysis of existing `closed_only_timeseries.csv` files. No controller changes, no simulation.",
        "",
        "## Config",
        "",
        f"- files scanned: `{int((files_df['skipped'] == 0).sum()) if not files_df.empty else 0}`",
        "- duplicate case timestamps: deduplicated by default",
        f"- horizons_min: `{','.join(str(x) for x in parse_horizons(args.horizons_min))}`",
        f"- base_restart_kg: `{float(args.base_restart_kg):.1f}`",
        f"- veto current-error window: `({float(args.base_restart_kg):.1f}, {float(args.max_veto_err_kg):.1f}] kg`",
        f"- relief_ratio: `{float(args.relief_ratio):.2f}`",
        f"- max_defer_err_kg: `{float(args.max_defer_err_kg):.1f}`",
        "",
        "## Overall decision",
        "",
        f"- restart_events: `{total_events}`",
        f"- safe_veto_events: `{safe_events}` (`{overall['safe_veto_rate_pct']:.2f}%`)",
        f"- safe_veto_event_pump_work: `{safe_work:.2f} m3` (`{overall['safe_veto_share_of_total_pump_work_pct']:.2f}%` of total pump work)",
        f"- decision: **{decision_text(overall)}**",
        "",
        "## By group",
        "",
        "| group | files | restarts | restarts/h | safe_veto | safe_veto% | safe_work_share | too_large | not_relieving | debt_cap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg_df.sort_values("group").iterrows():
        lines.append(
            f"| {r['group']} | {int(r['files'])} | {int(r['restart_events'])} | "
            f"{r['restart_events_per_h']:.2f} | {int(r['safe_veto_events'])} | "
            f"{r['safe_veto_rate_pct']:.2f}% | {r['safe_veto_share_of_total_pump_work_pct']:.2f}% | "
            f"{int(r['reason_too_large_now'])} | {int(r['reason_not_relieving'])} | {int(r['reason_debt_cap'])} |"
        )

    reason_counts = Counter(events_df["oracle_reason"].tolist()) if total_events else Counter()
    bucket_counts = Counter(events_df["err_bucket"].tolist()) if total_events else Counter()
    lines += [
        "",
        "## Restart error buckets",
        "",
        "| bucket | count | pct |",
        "|---|---:|---:|",
    ]
    for bucket in ["<=500", "500-1000", "1000-2000", "2000-3000", ">3000"]:
        count = int(bucket_counts.get(bucket, 0))
        lines.append(f"| {bucket} | {count} | {count / max(total_events, 1) * 100.0:.2f}% |")

    lines += [
        "",
        "## Reject reasons",
        "",
        "| reason | count | pct |",
        "|---|---:|---:|",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {int(count)} | {int(count) / max(total_events, 1) * 100.0:.2f}% |")

    lines += [
        "",
        "## Outputs",
        "",
        "- `restart_oracle_events.csv`",
        "- `restart_oracle_files.csv`",
        "- `restart_oracle_by_group.csv`",
    ]
    (out_dir / "restart_oracle_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--horizons-min", default="10,20,30")
    ap.add_argument("--base-restart-kg", type=float, default=500.0)
    ap.add_argument("--max-veto-err-kg", type=float, default=2000.0)
    ap.add_argument("--relief-ratio", type=float, default=0.60)
    ap.add_argument("--max-defer-err-kg", type=float, default=3000.0)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--no-dedupe", action="store_true", help="count duplicate case timestamps from multiple output dirs")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons_min = parse_horizons(args.horizons_min)

    files = discover_closed_only_files(input_root)
    discovered_count = len(files)
    if not bool(args.no_dedupe):
        files = dedupe_files(files)
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]
    if not files:
        raise FileNotFoundError(f"no *closed_only_timeseries.csv files found under {input_root}")

    all_events: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    print(f"Discovered {discovered_count} closed_only files; analyzing {len(files)} after dedupe.")
    for path in files:
        events, summary = analyze_file(path, horizons_min=horizons_min, args=args)
        all_events.extend(events)
        file_rows.append(summary)
        print(
            f"[OK] {summary.get('group')} {path.name}: "
            f"events={len(events)} safe={sum(e.get('safe_veto_oracle', 0) for e in events)}"
        )

    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        events_df["err_bucket"] = events_df["current_abs_err_kg"].map(bucket_abs_err)
    files_df = pd.DataFrame(file_rows)
    agg_df = aggregate(events_df, files_df)

    events_df.to_csv(out_dir / "restart_oracle_events.csv", index=False)
    files_df.to_csv(out_dir / "restart_oracle_files.csv", index=False)
    agg_df.to_csv(out_dir / "restart_oracle_by_group.csv", index=False)
    write_report(out_dir=out_dir, events_df=events_df, files_df=files_df, agg_df=agg_df, args=args)
    print(f"Report: {out_dir / 'restart_oracle_report.md'}")


if __name__ == "__main__":
    main()
