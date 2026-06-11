#!/usr/bin/env python3
"""Post-hoc temperature calibration for PSC regime/event probabilities."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modeling.train_psc_decision_model_v1 import PscDecisionModel, choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        default=Path("data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/dataset"),
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
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/calibration_gru_v1"),
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


class CalibrationDataset(Dataset):
    def __init__(self, source_dataset_dir: Path, label_dir: Path, split: str) -> None:
        self.x = np.load(source_dataset_dir / f"X_{split}.npy", mmap_mode="r")
        self.regime = np.load(label_dir / f"y_regime_{split}.npy", mmap_mode="r")
        self.multilabel = np.load(label_dir / f"y_multilabel_{split}.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.regime.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(np.array(self.x[idx], dtype=np.float32, copy=True)),
            "regime": torch.tensor(int(self.regime[idx]), dtype=torch.long),
            "multilabel": torch.from_numpy(np.array(self.multilabel[idx], dtype=np.float32, copy=True)),
        }


def build_model(checkpoint: dict, device: torch.device) -> PscDecisionModel:
    source_meta = checkpoint["source_metadata"]
    manifest = checkpoint["decision_manifest"]
    model_args = checkpoint["args"]
    model = PscDecisionModel(
        input_size=len(source_meta["feature_columns"]),
        hidden_size=int(model_args["hidden_size"]),
        num_layers=int(model_args["num_layers"]),
        num_regimes=len(manifest["regime_classes"]),
        num_multilabel=len(manifest["multilabel_columns"]),
        num_scalar=len(manifest["scalar_columns"]),
        dropout=float(model_args.get("dropout", 0.10)),
        encoder_type=str(model_args.get("encoder", "gru")),
        tcn_kernel_size=int(model_args.get("tcn_kernel_size", 3)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def collect_logits(
    model: PscDecisionModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    regime_logits = []
    regime_true = []
    event_logits = []
    event_true = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            pred = model(batch["x"].to(device))
            regime_logits.append(pred["regime_logits"].cpu())
            event_logits.append(pred["multilabel_logits"].cpu())
            regime_true.append(batch["regime"].cpu())
            event_true.append(batch["multilabel"].cpu())
    return (
        torch.cat(regime_logits, dim=0),
        torch.cat(regime_true, dim=0),
        torch.cat(event_logits, dim=0),
        torch.cat(event_true, dim=0),
    )


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == y_true
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return ece


def binary_ece(prob: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
    return ece


def tune_scalar_temperature(logits: torch.Tensor, target: torch.Tensor, kind: str) -> tuple[float, float]:
    candidates = torch.tensor(np.concatenate([np.linspace(0.5, 3.0, 51), np.linspace(3.1, 8.0, 50)]), dtype=torch.float32)
    best_temp = 1.0
    best_loss = float("inf")
    if kind == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
        for temp in candidates:
            loss = float(loss_fn(logits / temp, target))
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp)
    elif kind == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
        for temp in candidates:
            loss = float(loss_fn(logits / temp, target))
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp)
    else:
        raise ValueError(kind)
    return best_temp, best_loss


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint, device)
    manifest = checkpoint["decision_manifest"]
    dataset = CalibrationDataset(args.source_dataset_dir, args.label_dir, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    regime_logits, regime_true, event_logits, event_true = collect_logits(model, loader, device, args.max_batches)

    regime_temp, regime_nll = tune_scalar_temperature(regime_logits, regime_true, "multiclass")
    event_temps = []
    event_losses = []
    for idx in range(event_logits.shape[1]):
        temp, loss = tune_scalar_temperature(event_logits[:, idx], event_true[:, idx], "binary")
        event_temps.append(temp)
        event_losses.append(loss)

    raw_probs = torch.softmax(regime_logits, dim=1).numpy()
    cal_probs = torch.softmax(regime_logits / regime_temp, dim=1).numpy()
    raw_event = torch.sigmoid(event_logits).numpy()
    cal_event = torch.sigmoid(event_logits / torch.tensor(event_temps, dtype=torch.float32)).numpy()
    regime_true_np = regime_true.numpy()
    event_true_np = event_true.numpy()

    summary = {
        "split": args.split,
        "rows": int(len(regime_true_np)),
        "base_checkpoint": str(args.base_checkpoint),
        "regime_temperature": regime_temp,
        "regime_nll": regime_nll,
        "regime_ece_raw": expected_calibration_error(raw_probs, regime_true_np),
        "regime_ece_calibrated": expected_calibration_error(cal_probs, regime_true_np),
        "event_temperatures": {
            name: float(event_temps[idx]) for idx, name in enumerate(manifest["multilabel_columns"])
        },
        "event_bce": {
            name: float(event_losses[idx]) for idx, name in enumerate(manifest["multilabel_columns"])
        },
        "event_ece_raw": {
            name: binary_ece(raw_event[:, idx], event_true_np[:, idx])
            for idx, name in enumerate(manifest["multilabel_columns"])
        },
        "event_ece_calibrated": {
            name: binary_ece(cal_event[:, idx], event_true_np[:, idx])
            for idx, name in enumerate(manifest["multilabel_columns"])
        },
    }
    out_path = args.output_dir / "psc_probability_calibration.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
