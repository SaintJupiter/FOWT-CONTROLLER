#!/usr/bin/env python3
"""Mine clean reversible forecast windows for the Path-A falsifier.

This is a read-only audit. It compares learned and current-only forecast
geometry on FINO1 replay windows and writes a small cases CSV that can be fed
directly into run_prediction_primary_casebook.py via --cases-csv.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.ballast_planner import PlannerConfig  # noqa: E402
from wind_prediction.forecast_adapter import ForecastModelAdapter  # noqa: E402


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
BLOCKS = ((0, 2), (2, 4), (4, 6))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_split_timestamps(dataset_dir: Path, split: str, limit: int | None) -> list[str]:
    out: list[str] = []
    with gzip.open(dataset_dir / "sample_index.csv.gz", "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row["split"]).strip() != split:
                continue
            out.append(str(row["history_end"]))
            if limit is not None and len(out) >= int(limit):
                break
    return out


def _load_split_arrays(dataset_dir: Path, metadata: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = metadata["arrays"][split]
    x = np.load(dataset_dir / arrays["X"], mmap_mode="r")
    y_uv_raw = np.load(dataset_dir / arrays["y_uv_raw"], mmap_mode="r")
    return x, y_uv_raw


def _learned_uv(
    adapter: ForecastModelAdapter,
    x: np.ndarray,
    n: int,
    batch_size: int,
) -> np.ndarray:
    adapter._lazy_load_runtime()
    torch = adapter._torch
    rows: list[np.ndarray] = []
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        xb = torch.from_numpy(np.array(x[start:end], dtype=np.float32, copy=True)).to(adapter._device)
        with torch.no_grad():
            pred_uv_scaled, _ = adapter._model(xb)
            pred_uv_scaled = adapter._full_scaled_prediction(pred_uv_scaled, xb)
        pred = adapter._inverse_scale_uv(pred_uv_scaled.detach().cpu().numpy().astype(np.float32))
        rows.append(np.asarray(pred, dtype=np.float32))
    return np.concatenate(rows, axis=0)


def _current_only_uv(x: np.ndarray, metadata: dict[str, Any], scaler: dict[str, Any], n: int) -> np.ndarray:
    feature_columns = list(metadata["feature_columns"])
    u_idx = feature_columns.index("wind_u_ms")
    v_idx = feature_columns.index("wind_v_ms")
    feature_scaler = scaler["feature_scaler"]
    mean = np.array(
        [feature_scaler["wind_u_ms"]["mean"], feature_scaler["wind_v_ms"]["mean"]],
        dtype=np.float32,
    )
    std = np.array(
        [feature_scaler["wind_u_ms"]["std"], feature_scaler["wind_v_ms"]["std"]],
        dtype=np.float32,
    )
    latest_scaled = np.asarray(x[:n, -1, [u_idx, v_idx]], dtype=np.float32)
    latest_raw = latest_scaled * std + mean
    future_steps = int(metadata["future_steps"])
    return np.repeat(latest_raw[:, None, :], future_steps, axis=1).astype(np.float32)


def _speed(uv: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.asarray(uv, dtype=np.float32) ** 2, axis=-1))


def _pressure_vectors(uv: np.ndarray, cfg: PlannerConfig, wind_ref: float) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv, dtype=np.float32)
    n = uv.shape[0]
    vecs = np.zeros((n, 3, 2), dtype=np.float32)
    norms = np.zeros((n, 3), dtype=np.float32)
    db = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=np.float32)
    for block_idx, (start, end) in enumerate(BLOCKS):
        block = uv[:, start:end, :]
        block_speed = _speed(block)
        mean_speed = np.mean(block_speed, axis=1)
        mean_u = np.mean(block[:, :, 0], axis=1)
        mean_v = np.mean(block[:, :, 1], axis=1)
        mean_vec_norm = np.maximum(np.sqrt(mean_u**2 + mean_v**2), 1e-6)
        mag = np.clip(
            (np.maximum(mean_speed, 0.0) / max(float(wind_ref), 1.0)) ** 2,
            0.0,
            max(float(cfg.pressure_norm_cap), 1e-6),
        )
        raw_pitch = float(cfg.pressure_sign_multiplier) * db[0] * mag * mean_v / mean_vec_norm
        raw_roll = -float(cfg.pressure_sign_multiplier) * db[1] * mag * mean_u / mean_vec_norm
        vecs[:, block_idx, 0] = raw_pitch
        vecs[:, block_idx, 1] = raw_roll
        norms[:, block_idx] = np.sqrt((raw_pitch / db[0]) ** 2 + (raw_roll / db[1]) ** 2)
    return vecs, norms


def _dir_shift_deg(a_uv: np.ndarray, b_uv: np.ndarray) -> np.ndarray:
    a = np.asarray(a_uv, dtype=np.float32)
    b = np.asarray(b_uv, dtype=np.float32)
    a_dir = (np.rad2deg(np.arctan2(-a[:, 0], -a[:, 1])) + 360.0) % 360.0
    b_dir = (np.rad2deg(np.arctan2(-b[:, 0], -b[:, 1])) + 360.0) % 360.0
    return np.abs((b_dir - a_dir + 180.0) % 360.0 - 180.0)


def _forecast_features(
    uv: np.ndarray,
    cfg: PlannerConfig,
    wind_ref: float,
    prefix: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    speed = _speed(uv)
    block_speed = np.column_stack([np.mean(speed[:, s:e], axis=1) for s, e in BLOCKS])
    vecs, norms = _pressure_vectors(uv, cfg=cfg, wind_ref=wind_ref)
    dot02 = np.sum(vecs[:, 0, :] * vecs[:, 2, :], axis=1)
    norm02 = np.maximum(
        np.linalg.norm(vecs[:, 0, :], axis=1) * np.linalg.norm(vecs[:, 2, :], axis=1),
        1e-9,
    )
    cos02 = dot02 / norm02
    mean_uv0 = np.mean(uv[:, 0:2, :], axis=1)
    mean_uv2 = np.mean(uv[:, 4:6, :], axis=1)
    dir_shift02 = _dir_shift_deg(mean_uv0, mean_uv2)
    near_peak_speed = np.max(block_speed[:, :2], axis=1)
    near_peak_norm = np.max(norms[:, :2], axis=1)
    far_speed = block_speed[:, 2]
    far_norm = norms[:, 2]
    speed_range = np.max(block_speed, axis=1) - np.min(block_speed, axis=1)
    speed_drop = near_peak_speed - far_speed
    norm_drop = near_peak_norm - far_norm
    candidate = (
        (near_peak_speed >= float(args.min_near_speed_ms))
        & (near_peak_speed <= float(args.max_near_speed_ms))
        & (near_peak_norm >= float(args.min_near_pressure_norm))
        & (near_peak_norm <= float(args.max_near_pressure_norm))
        & (speed_drop >= float(args.min_speed_drop_ms))
        & (norm_drop >= float(args.min_norm_drop))
        & (far_norm <= float(args.max_far_pressure_norm))
        & (dir_shift02 <= float(args.max_dir_shift_deg))
        & (cos02 >= float(args.min_cos02))
        & (speed_range <= float(args.max_speed_range_ms))
    )
    return pd.DataFrame(
        {
            f"{prefix}_b0_speed_ms": block_speed[:, 0],
            f"{prefix}_b1_speed_ms": block_speed[:, 1],
            f"{prefix}_b2_speed_ms": block_speed[:, 2],
            f"{prefix}_near_peak_speed_ms": near_peak_speed,
            f"{prefix}_far_speed_ms": far_speed,
            f"{prefix}_speed_drop_ms": speed_drop,
            f"{prefix}_speed_range_ms": speed_range,
            f"{prefix}_raw_norm0": norms[:, 0],
            f"{prefix}_raw_norm1": norms[:, 1],
            f"{prefix}_raw_norm2": norms[:, 2],
            f"{prefix}_near_peak_norm": near_peak_norm,
            f"{prefix}_norm_drop": norm_drop,
            f"{prefix}_cos02": cos02,
            f"{prefix}_dir_shift02_deg": dir_shift02,
            f"{prefix}_clean_reversible_candidate": candidate.astype(np.int8),
        }
    )


def _thin_by_time(df: pd.DataFrame, limit: int, min_separation_min: float) -> pd.DataFrame:
    selected = []
    used: list[datetime] = []
    sep = timedelta(minutes=float(min_separation_min))
    for _, row in df.iterrows():
        ts = datetime.strptime(str(row["timestamp"]), TIMESTAMP_FMT)
        if all(abs(ts - prev) >= sep for prev in used):
            selected.append(row)
            used.append(ts)
        if len(selected) >= int(limit):
            break
    if not selected:
        return df.head(0).copy()
    return pd.DataFrame(selected)


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_none_"
    d = df.head(int(max_rows)).copy()
    cols = [str(c) for c in d.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in d.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
    )
    parser.add_argument("--split", choices=("test", "validation", "train"), default="test")
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/clean_reversible_prediction_window_mining_v1",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole split.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wind-ref-ms", type=float, default=12.0)
    parser.add_argument("--case-limit", type=int, default=16)
    parser.add_argument("--min-separation-min", type=float, default=180.0)
    parser.add_argument("--min-near-speed-ms", type=float, default=13.0)
    parser.add_argument("--max-near-speed-ms", type=float, default=15.5)
    parser.add_argument("--min-near-pressure-norm", type=float, default=0.90)
    parser.add_argument("--max-near-pressure-norm", type=float, default=1.55)
    parser.add_argument("--min-speed-drop-ms", type=float, default=1.5)
    parser.add_argument("--min-norm-drop", type=float, default=0.20)
    parser.add_argument("--max-far-pressure-norm", type=float, default=1.15)
    parser.add_argument("--max-dir-shift-deg", type=float, default=35.0)
    parser.add_argument("--min-cos02", type=float, default=0.65)
    parser.add_argument("--max-speed-range-ms", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(args.dataset_dir)
    model_dir = _resolve(args.model_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    scaler = json.loads((dataset_dir / "scaler_train.json").read_text(encoding="utf-8"))
    x_all, y_uv_all = _load_split_arrays(dataset_dir, metadata, str(args.split))
    n = int(x_all.shape[0])
    if int(args.limit) > 0:
        n = min(n, int(args.limit))
    timestamps = _read_split_timestamps(dataset_dir, str(args.split), limit=n)
    if len(timestamps) != n:
        raise ValueError(f"timestamp count mismatch: {len(timestamps)} vs {n}")

    adapter = ForecastModelAdapter(model_dir=model_dir, dataset_dir=dataset_dir, device=str(args.device))
    learned_uv = _learned_uv(adapter, x_all, n=n, batch_size=int(args.batch_size))
    current_uv = _current_only_uv(x_all, metadata=metadata, scaler=scaler, n=n)
    oracle_uv = np.asarray(y_uv_all[:n], dtype=np.float32)
    cfg = PlannerConfig()

    base = pd.DataFrame({"split_index": np.arange(n, dtype=np.int64), "timestamp": timestamps})
    learned = _forecast_features(learned_uv, cfg, float(args.wind_ref_ms), "learned", args)
    current = _forecast_features(current_uv, cfg, float(args.wind_ref_ms), "current", args)
    oracle = _forecast_features(oracle_uv, cfg, float(args.wind_ref_ms), "oracle", args)
    df = pd.concat([base, learned, current, oracle], axis=1)

    learned_candidate = df["learned_clean_reversible_candidate"].astype(bool)
    current_candidate = df["current_clean_reversible_candidate"].astype(bool)
    oracle_candidate = df["oracle_clean_reversible_candidate"].astype(bool)
    df["learned_only_candidate"] = (learned_candidate & ~current_candidate).astype(np.int8)
    df["learned_oracle_agree_candidate"] = (learned_candidate & oracle_candidate).astype(np.int8)
    df["mean_abs_uv_diff_learned_current"] = np.mean(np.abs(learned_uv - current_uv), axis=(1, 2))
    df["learned_minus_current_norm_drop"] = df["learned_norm_drop"] - df["current_norm_drop"]

    ranked = df[df["learned_only_candidate"] == 1].copy()
    ranked = ranked.sort_values(
        [
            "learned_oracle_agree_candidate",
            "learned_minus_current_norm_drop",
            "mean_abs_uv_diff_learned_current",
            "learned_norm_drop",
        ],
        ascending=[False, False, False, False],
    )
    selected = _thin_by_time(ranked, limit=int(args.case_limit), min_separation_min=float(args.min_separation_min))
    oracle_ranked = df[df["oracle_clean_reversible_candidate"] == 1].copy()
    oracle_ranked = oracle_ranked.sort_values(
        [
            "learned_clean_reversible_candidate",
            "oracle_norm_drop",
            "learned_norm_drop",
        ],
        ascending=[False, False, False],
    )

    df.to_csv(out_dir / "all_window_forecast_geometry.csv", index=False)
    ranked.to_csv(out_dir / "learned_only_clean_reversible_candidates.csv", index=False)
    oracle_ranked.to_csv(out_dir / "oracle_clean_reversible_candidates_with_learned_geometry.csv", index=False)
    case_rows = []
    for idx, row in enumerate(selected.to_dict("records"), start=1):
        case_rows.append(
            {
                "case_id": f"clean_reversible_{idx:02d}",
                "timestamp": row["timestamp"],
                "label": (
                    "learned-only clean reversible wind-shape candidate; "
                    f"oracle_agree={int(row['learned_oracle_agree_candidate'])}"
                ),
            }
        )
    cases = pd.DataFrame(case_rows)
    cases.to_csv(out_dir / "clean_reversible_cases_for_casebook.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "windows_scanned", "value": int(n)},
            {"metric": "learned_candidates", "value": int(learned_candidate.sum())},
            {"metric": "current_candidates", "value": int(current_candidate.sum())},
            {"metric": "oracle_candidates", "value": int(oracle_candidate.sum())},
            {"metric": "learned_only_candidates", "value": int((learned_candidate & ~current_candidate).sum())},
            {"metric": "oracle_only_candidates", "value": int((oracle_candidate & ~learned_candidate).sum())},
            {"metric": "learned_and_oracle_candidates", "value": int((learned_candidate & oracle_candidate).sum())},
            {"metric": "selected_casebook_cases", "value": int(len(cases))},
        ]
    )
    summary.to_csv(out_dir / "selection_summary.csv", index=False)

    top_cols = [
        "timestamp",
        "learned_near_peak_speed_ms",
        "learned_speed_drop_ms",
        "learned_norm_drop",
        "current_norm_drop",
        "oracle_norm_drop",
        "learned_dir_shift02_deg",
        "mean_abs_uv_diff_learned_current",
        "learned_oracle_agree_candidate",
    ]
    report = [
        "# Clean Reversible Prediction Window Mining v1",
        "",
        "Purpose: find wind-only windows where the learned forecast opens a small, reversible, low-direction-shift opportunity that current-only does not.",
        "",
        "This does not prove closed-loop pump saving. It is the prefilter for the Path-A falsifier; selected rows still need learned vs current-only closed-loop runs and safety metrics.",
        "",
        "## Rules",
        "",
        f"- near peak speed: `{args.min_near_speed_ms}` to `{args.max_near_speed_ms}` m/s.",
        f"- near peak pressure norm: `{args.min_near_pressure_norm}` to `{args.max_near_pressure_norm}`.",
        f"- drop: speed >= `{args.min_speed_drop_ms}` m/s and norm >= `{args.min_norm_drop}`.",
        f"- far pressure norm <= `{args.max_far_pressure_norm}`.",
        f"- direction stability: dir shift <= `{args.max_dir_shift_deg}` deg and cos02 >= `{args.min_cos02}`.",
        f"- block speed range <= `{args.max_speed_range_ms}` m/s.",
        "",
        "## Counts",
        "",
        _markdown_table(summary),
        "",
        "## Selected Learned-Only Candidates",
        "",
        _markdown_table(selected[top_cols] if len(selected) else selected, max_rows=int(args.case_limit)),
        "",
        "## Oracle Opportunities Missed By Learned",
        "",
        _markdown_table(oracle_ranked[top_cols] if len(oracle_ranked) else oracle_ranked, max_rows=int(args.case_limit)),
        "",
        "## Outputs",
        "",
        f"- `{out_dir / 'all_window_forecast_geometry.csv'}`",
        f"- `{out_dir / 'learned_only_clean_reversible_candidates.csv'}`",
        f"- `{out_dir / 'oracle_clean_reversible_candidates_with_learned_geometry.csv'}`",
        f"- `{out_dir / 'clean_reversible_cases_for_casebook.csv'}`",
        "",
        "Suggested next command shape:",
        "",
        "```bash",
        "./.venv312/bin/python scripts/analysis/run_prediction_primary_casebook.py "
        "--cases-csv outputs/wind_prediction/clean_reversible_prediction_window_mining_v1/clean_reversible_cases_for_casebook.csv "
        "--forecast-source learned --duration-s 43200 --skip-figures",
        "```",
    ]
    (out_dir / "selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Report: {out_dir / 'selection_report.md'}")


if __name__ == "__main__":
    main()
