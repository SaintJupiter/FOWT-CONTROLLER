#!/usr/bin/env python3
"""Export PSC decision signals for downstream control-chain experiments.

This script combines:

1. the base PSC decision model (regime/event/scalar heads), and
2. the replay-value adapter (closed pump waste, deadband opportunity, relax risk).

The output is a row-aligned CSV keyed by source split row and future_start, so a
selector/control-chain replay can consume model signals without knowing PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modeling.train_psc_decision_model_v1 import choose_device
from scripts.modeling.train_psc_replay_value_adapter_v1 import (
    ReplayValueAdapter,
    TARGET_COLUMNS,
    build_base_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path(
            "outputs/wind_prediction/psc_decision_model_v1/"
            "model_gru_h96_l1_mps_screen_20260531/psc_decision_model_best.pt"
        ),
    )
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        default=Path(
            "outputs/wind_prediction/psc_decision_model_v1/"
            "replay_value_adapter_gru_v1/psc_replay_value_adapter_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/decision_signals_gru_adapter_v1"),
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--calibration-json", type=Path, default=None)
    return parser.parse_args()


class SplitDataset(Dataset):
    def __init__(self, source_dataset_dir: Path, split: str) -> None:
        self.x = np.load(source_dataset_dir / f"X_{split}.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.x[idx], dtype=np.float32, copy=True))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    base_checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    adapter_checkpoint = torch.load(args.adapter_checkpoint, map_location=device, weights_only=False)
    base_model = build_base_model(base_checkpoint, args.source_dataset_dir, device)

    encoded_size = int(base_checkpoint["args"]["hidden_size"])
    adapter_args = adapter_checkpoint["args"]
    adapter = ReplayValueAdapter(
        base_model=base_model,
        encoded_size=encoded_size,
        hidden_size=int(adapter_args.get("hidden_size", encoded_size)),
        dropout=float(adapter_args.get("dropout", 0.10)),
    ).to(device)
    adapter.value_head.load_state_dict(adapter_checkpoint["adapter_state"])
    adapter.eval()

    manifest = base_checkpoint["decision_manifest"]
    scalar_mean = np.asarray(base_checkpoint["scalar_mean"], dtype=np.float32)
    scalar_std = np.asarray(base_checkpoint["scalar_std"], dtype=np.float32)
    target_log_mean = np.asarray(adapter_checkpoint["target_log_mean"], dtype=np.float32)
    target_log_std = np.asarray(adapter_checkpoint["target_log_std"], dtype=np.float32)
    calibration = None
    regime_temperature = 1.0
    event_temperatures = np.ones(len(manifest["multilabel_columns"]), dtype=np.float32)
    if args.calibration_json is not None:
        calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
        regime_temperature = float(calibration.get("regime_temperature", 1.0))
        event_temperature_map = calibration.get("event_temperatures", {})
        event_temperatures = np.asarray(
            [float(event_temperature_map.get(name, 1.0)) for name in manifest["multilabel_columns"]],
            dtype=np.float32,
        )
    event_temperature_tensor = torch.tensor(event_temperatures, dtype=torch.float32, device=device)

    dataset = SplitDataset(args.source_dataset_dir, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    parts = []
    row_start = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            pred = base_model(x)
            regime_prob = torch.softmax(pred["regime_logits"] / regime_temperature, dim=1).cpu().numpy()
            event_prob = torch.sigmoid(pred["multilabel_logits"] / event_temperature_tensor).cpu().numpy()
            scalar = pred["scalar"].cpu().numpy() * scalar_std + scalar_mean

            value_scaled = adapter(x).cpu().numpy()
            value_log = value_scaled * target_log_std + target_log_mean
            value_raw = np.maximum(np.expm1(value_log), 0.0)

            n = int(batch.shape[0])
            frame = pd.DataFrame({"split": args.split, "split_row": np.arange(row_start, row_start + n, dtype=np.int64)})
            for idx, name in enumerate(manifest["regime_classes"]):
                frame[f"psc_regime_prob_{name}"] = regime_prob[:, idx]
            frame["psc_regime_argmax"] = [manifest["regime_classes"][i] for i in regime_prob.argmax(axis=1)]
            frame["psc_regime_confidence"] = regime_prob.max(axis=1)
            for idx, name in enumerate(manifest["multilabel_columns"]):
                frame[f"psc_prob_{name}"] = event_prob[:, idx]
            for idx, name in enumerate(manifest["scalar_columns"]):
                frame[f"psc_scalar_{name}"] = scalar[:, idx]
            for idx, name in enumerate(TARGET_COLUMNS):
                frame[f"psc_value_{name}"] = value_raw[:, idx]
            parts.append(frame)
            row_start += n

    signals = pd.concat(parts, ignore_index=True)
    sample_index = pd.read_csv(args.source_dataset_dir / "sample_index.csv.gz")
    sample_index = sample_index.loc[sample_index["split"].eq(args.split)].reset_index(drop=True)
    key_cols = [
        "series_id",
        "history_start",
        "history_end",
        "future_start",
        "future_end",
        "ballast_attention_event",
        "attention_event_0_20m",
        "attention_event_20_40m",
        "attention_event_40_60m",
        "attention_event_60_80m",
        "attention_event_80_100m",
        "attention_event_100_120m",
    ]
    available_key_cols = [col for col in key_cols if col in sample_index.columns]
    out = pd.concat([sample_index[available_key_cols], signals], axis=1)

    output_csv = args.output_dir / f"psc_decision_signals_{args.split}.csv.gz"
    out.to_csv(output_csv, index=False)
    summary = {
        "split": args.split,
        "rows": int(len(out)),
        "device": str(device),
        "base_checkpoint": str(args.base_checkpoint),
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "calibration_json": str(args.calibration_json) if args.calibration_json is not None else None,
        "regime_temperature": regime_temperature,
        "output_csv": str(output_csv),
        "regime_argmax_counts": out["psc_regime_argmax"].value_counts().to_dict(),
        "mean_values": {
            f"psc_value_{name}": float(out[f"psc_value_{name}"].mean()) for name in TARGET_COLUMNS
        },
    }
    (args.output_dir / f"psc_decision_signals_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
