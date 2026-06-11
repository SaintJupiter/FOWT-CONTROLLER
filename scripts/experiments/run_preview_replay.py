#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run passive FINO1 replay forecast feeds through the legacy closed-only controller baseline."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--start-mode", choices=["first_sample", "first_positive"], default="first_positive")
    parser.add_argument("--event-name", default="ballast_attention_event")
    parser.add_argument(
        "--replay-rows",
        type=int,
        default=48,
        help="Number of 10-minute FINO1 rows to replay after the chosen start timestamp.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=10.0,
        help="Simulation dt in seconds. Must divide 600s evenly for FINO1 replay.",
    )
    parser.add_argument("--case-prefix", default="preview_feed_replay")
    parser.add_argument("--with-oracle-feed", action="store_true")
    parser.add_argument("--record-timeseries", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

    from run_validation import discover_stiffness_file, run_closed_loop_case
    from wind_prediction import Fino1ReplayDataset, ForecastModelAdapter, ModelPreviewProvider, OraclePreviewProvider

    replay = Fino1ReplayDataset(dataset_dir=args.dataset_dir, split=args.split)
    if args.start_mode == "first_positive":
        start_timestamp = replay.first_positive_sample_timestamp(event_name=args.event_name)
    else:
        start_timestamp = replay.first_sample_timestamp()

    wind_trace = replay.build_wind_trace(
        start_timestamp=start_timestamp,
        row_count=args.replay_rows,
        dt_s=args.dt,
    )
    n_steps = int(wind_trace["n_steps"])

    adapter = ForecastModelAdapter(model_dir=args.model_dir, dataset_dir=args.dataset_dir, device="cpu")
    model_provider = ModelPreviewProvider(
        replay_dataset=replay,
        forecast_adapter=adapter,
        start_timestamp=start_timestamp,
    )
    oracle_provider = None
    if args.with_oracle_feed:
        oracle_provider = OraclePreviewProvider(
            replay_dataset=replay,
            start_timestamp=start_timestamp,
        )

    excel_path = discover_stiffness_file()
    if not excel_path:
        archive_candidate = (
            repo_root / "archive" / "legacy_fowt_control" / "data" / "副本水平刚度曲线.xlsx"
        )
        if archive_candidate.exists():
            excel_path = str(archive_candidate)
    if not excel_path:
        raise SystemExit("No stiffness file found under data/.")

    summary_rows = []
    runs = [("no_preview", None), ("model_feed_passive", model_provider)]
    if oracle_provider is not None:
        runs.append(("oracle_feed_passive", oracle_provider))

    for label, provider in runs:
        row, _timeseries = run_closed_loop_case(
            excel_path=excel_path,
            case_name=f"{args.case_prefix}_{label}",
            dt=args.dt,
            n_steps=n_steps,
            wind_trace=wind_trace,
            trim_cfg=None,
            control_enabled=True,
            record_timeseries=bool(args.record_timeseries),
            preview_trim_provider=provider,
        )
        row["comparison_label"] = label
        row["replay_start_timestamp"] = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        row["replay_rows"] = int(args.replay_rows)
        row["sim_dt_s"] = float(args.dt)
        summary_rows.append(row)

    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = results_dir / f"{args.case_prefix}_{args.split}_{ts}_summary.csv"
    write_csv(summary_path, summary_rows)
    print(f"Saved passive preview integration summary to {summary_path}")

    model_feed_path = results_dir / f"{args.case_prefix}_{args.split}_{ts}_model_feed.csv"
    write_csv(model_feed_path, model_provider.records)
    print(f"Saved model forecast feed to {model_feed_path}")

    if oracle_provider is not None:
        oracle_feed_path = results_dir / f"{args.case_prefix}_{args.split}_{ts}_oracle_feed.csv"
        write_csv(oracle_feed_path, oracle_provider.records)
        print(f"Saved oracle forecast feed to {oracle_feed_path}")

    for row in summary_rows:
        print(
            f"{row['comparison_label']}: "
            f"pitch_rms_deg={row.get('pitch_rms_deg')}, "
            f"roll_rms_deg={row.get('roll_rms_deg')}, "
            f"switch_per_min={row.get('switch_per_min')}"
        )


if __name__ == "__main__":
    main()
