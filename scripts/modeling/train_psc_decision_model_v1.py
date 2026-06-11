#!/usr/bin/env python3
"""Train a multi-head PSC decision model.

The model is deliberately a decision model, not a better wind-regression model:
one shared temporal encoder produces regime probabilities, event-risk heads,
scalar anticipation/severity heads, and optional calibrated probabilities for a
thin supervisory selector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPLITS = ("train", "validation", "test")


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
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/model_lstm_h128"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--encoder",
        choices=["lstm", "gru", "mean_pool_mlp", "tcn", "tcn_attention"],
        default="lstm",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--scalar-loss-weight", type=float, default=0.25)
    parser.add_argument("--multilabel-loss-weight", type=float, default=0.50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available.")
    return torch.device(requested)


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if device.type != "mps":
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PscDecisionDataset(Dataset):
    def __init__(
        self,
        source_dataset_dir: Path,
        label_dir: Path,
        split: str,
        scalar_mean: np.ndarray,
        scalar_std: np.ndarray,
    ) -> None:
        self.x = np.load(source_dataset_dir / f"X_{split}.npy", mmap_mode="r")
        self.regime = np.load(label_dir / f"y_regime_{split}.npy", mmap_mode="r")
        self.multilabel = np.load(label_dir / f"y_multilabel_{split}.npy", mmap_mode="r")
        self.scalar = np.load(label_dir / f"y_scalar_{split}.npy", mmap_mode="r")
        self.scalar_mean = scalar_mean.astype("float32")
        self.scalar_std = np.maximum(scalar_std.astype("float32"), 1e-6)

    def __len__(self) -> int:
        return int(self.regime.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        scalar = (np.asarray(self.scalar[idx], dtype=np.float32) - self.scalar_mean) / self.scalar_std
        return {
            "x": torch.from_numpy(np.array(self.x[idx], dtype=np.float32, copy=True)),
            "regime": torch.tensor(int(self.regime[idx]), dtype=torch.long),
            "multilabel": torch.from_numpy(np.array(self.multilabel[idx], dtype=np.float32, copy=True)),
            "scalar": torch.from_numpy(scalar.astype("float32")),
        }


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.LayerNorm(out_channels),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.LayerNorm(out_channels),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm expects channels-last; the sequential block temporarily converts.
        out = x
        for layer in self.net:
            if isinstance(layer, nn.LayerNorm):
                out = layer(out.transpose(1, 2)).transpose(1, 2)
            else:
                out = layer(out)
        residual = x if self.downsample is None else self.downsample(x)
        return out + residual


class TcnEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        use_attention: bool,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = input_size
        for layer_idx in range(num_layers):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
            )
            in_channels = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.use_attention = use_attention
        self.attention = nn.Linear(hidden_size, 1) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        if not self.use_attention:
            return z[:, -1, :]
        scores = self.attention(z).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (z * weights).sum(dim=1)


class PscDecisionModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_regimes: int,
        num_multilabel: int,
        num_scalar: int,
        dropout: float,
        encoder_type: str = "lstm",
        tcn_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type in {"lstm", "gru"}:
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            recurrent_cls = nn.LSTM if encoder_type == "lstm" else nn.GRU
            self.encoder = recurrent_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            encoded_size = hidden_size
        elif encoder_type in {"tcn", "tcn_attention"}:
            self.encoder = TcnEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                kernel_size=tcn_kernel_size,
                dropout=dropout,
                use_attention=encoder_type == "tcn_attention",
            )
            encoded_size = hidden_size
        elif encoder_type == "mean_pool_mlp":
            encoded_size = hidden_size
            self.encoder = nn.Sequential(
                nn.Linear(input_size * 4, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported encoder_type={encoder_type}")
        self.norm = nn.LayerNorm(encoded_size)
        self.drop = nn.Dropout(dropout)
        self.regime_head = nn.Linear(encoded_size, num_regimes)
        self.multilabel_head = nn.Linear(encoded_size, num_multilabel)
        self.scalar_head = nn.Linear(encoded_size, num_scalar)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_type in {"lstm", "gru"}:
            out, _ = self.encoder(x)
            h = out[:, -1, :]
        elif self.encoder_type in {"tcn", "tcn_attention"}:
            h = self.encoder(x)
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1, unbiased=False)
            last = x[:, -1, :]
            delta = x[:, -1, :] - x[:, 0, :]
            h = self.encoder(torch.cat([mean, std, last, delta], dim=1))
        h = self.drop(self.norm(h))
        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {
            "regime_logits": self.regime_head(h),
            "multilabel_logits": self.multilabel_head(h),
            "scalar": self.scalar_head(h),
        }


def class_weights(y_regime: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(y_regime, dtype=np.int64), minlength=num_classes).astype("float32")
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.from_numpy(np.clip(weights, 0.25, 20.0).astype("float32"))


def multilabel_pos_weight(y_multilabel: np.ndarray) -> torch.Tensor:
    rates = np.asarray(y_multilabel[:], dtype=np.float32).mean(axis=0)
    rates = np.clip(rates, 1e-4, 1.0 - 1e-4)
    weights = (1.0 - rates) / rates
    return torch.from_numpy(np.clip(weights, 1.0, 50.0).astype("float32"))


def scalar_stats(label_dir: Path, scalar_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train = np.load(label_dir / "y_scalar_train.npy", mmap_mode="r")
    mean = np.asarray(train[:], dtype=np.float32).mean(axis=0)
    std = np.asarray(train[:], dtype=np.float32).std(axis=0)
    stats = pd.DataFrame({"scalar": scalar_columns, "mean": mean, "std": np.maximum(std, 1e-6)})
    stats.to_csv(label_dir / "scalar_normalization_from_train.csv", index=False)
    return mean, np.maximum(std, 1e-6)


def batch_loss(
    model: PscDecisionModel,
    batch: dict[str, torch.Tensor],
    ce_loss: nn.Module,
    bce_loss: nn.Module,
    scalar_loss: nn.Module,
    device: torch.device,
    scalar_loss_weight: float,
    multilabel_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["x"].to(device)
    regime = batch["regime"].to(device)
    multilabel = batch["multilabel"].to(device)
    scalar = batch["scalar"].to(device)
    pred = model(x)
    loss_regime = ce_loss(pred["regime_logits"], regime)
    loss_multilabel = bce_loss(pred["multilabel_logits"], multilabel)
    loss_scalar = scalar_loss(pred["scalar"], scalar)
    loss = loss_regime + multilabel_loss_weight * loss_multilabel + scalar_loss_weight * loss_scalar
    return loss, {
        "loss": float(loss.detach().cpu()),
        "loss_regime": float(loss_regime.detach().cpu()),
        "loss_multilabel": float(loss_multilabel.detach().cpu()),
        "loss_scalar": float(loss_scalar.detach().cpu()),
    }


def run_epoch(
    model: PscDecisionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    ce_loss: nn.Module,
    bce_loss: nn.Module,
    scalar_loss: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    n_batches = 0
    correct = 0
    total = 0
    with torch.set_grad_enabled(training):
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            loss, parts = batch_loss(
                model=model,
                batch=batch,
                ce_loss=ce_loss,
                bce_loss=bce_loss,
                scalar_loss=scalar_loss,
                device=device,
                scalar_loss_weight=args.scalar_loss_weight,
                multilabel_loss_weight=args.multilabel_loss_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            with torch.no_grad():
                pred = model(batch["x"].to(device))["regime_logits"].argmax(dim=1).cpu()
                correct += int((pred == batch["regime"]).sum())
                total += int(batch["regime"].numel())
            n_batches += 1
    out = {key: value / max(n_batches, 1) for key, value in totals.items()}
    out["regime_accuracy"] = correct / max(total, 1)
    return out


def evaluate_predictions(
    model: PscDecisionModel,
    loader: DataLoader,
    device: torch.device,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
    manifest: dict,
    max_batches: int | None,
) -> dict:
    model.eval()
    num_regimes = len(manifest["regime_classes"])
    confusion = np.zeros((num_regimes, num_regimes), dtype=np.int64)
    scalar_abs_errors = []
    multilabel_probs = []
    multilabel_true = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            pred = model(batch["x"].to(device))
            regime_pred = pred["regime_logits"].argmax(dim=1).cpu().numpy()
            regime_true = batch["regime"].numpy()
            for t, p in zip(regime_true, regime_pred):
                confusion[int(t), int(p)] += 1
            scalar_pred = pred["scalar"].cpu().numpy() * scalar_std + scalar_mean
            scalar_true = batch["scalar"].numpy() * scalar_std + scalar_mean
            scalar_abs_errors.append(np.abs(scalar_pred - scalar_true))
            multilabel_probs.append(torch.sigmoid(pred["multilabel_logits"]).cpu().numpy())
            multilabel_true.append(batch["multilabel"].numpy())

    scalar_abs = np.concatenate(scalar_abs_errors, axis=0) if scalar_abs_errors else np.zeros((0, len(manifest["scalar_columns"])))
    probs = np.concatenate(multilabel_probs, axis=0) if multilabel_probs else np.zeros((0, len(manifest["multilabel_columns"])))
    true = np.concatenate(multilabel_true, axis=0) if multilabel_true else np.zeros_like(probs)
    multilabel_rows = []
    for i, name in enumerate(manifest["multilabel_columns"]):
        if true.shape[0] == 0:
            continue
        pred_binary = probs[:, i] >= 0.5
        y = true[:, i] >= 0.5
        tp = int((pred_binary & y).sum())
        fp = int((pred_binary & ~y).sum())
        fn = int((~pred_binary & y).sum())
        tn = int((~pred_binary & ~y).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        multilabel_rows.append(
            {
                "label": name,
                "base_rate": float(y.mean()),
                "precision_at_0p5": precision,
                "recall_at_0p5": recall,
                "pred_rate_at_0p5": float(pred_binary.mean()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return {
        "confusion_matrix": confusion.tolist(),
        "scalar_mae": {
            name: float(np.nanmean(scalar_abs[:, i])) if scalar_abs.shape[0] else math.nan
            for i, name in enumerate(manifest["scalar_columns"])
        },
        "multilabel_at_0p5": multilabel_rows,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    set_seed(args.seed, device)

    source_meta = json.loads((args.source_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.label_dir / "psc_decision_dataset_manifest.json").read_text(encoding="utf-8"))
    scalar_mean, scalar_std = scalar_stats(args.label_dir, manifest["scalar_columns"])

    datasets = {
        split: PscDecisionDataset(args.source_dataset_dir, args.label_dir, split, scalar_mean, scalar_std)
        for split in SPLITS
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0),
    }

    input_size = len(source_meta["feature_columns"])
    model = PscDecisionModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_regimes=len(manifest["regime_classes"]),
        num_multilabel=len(manifest["multilabel_columns"]),
        num_scalar=len(manifest["scalar_columns"]),
        dropout=args.dropout,
        encoder_type=args.encoder,
        tcn_kernel_size=args.tcn_kernel_size,
    ).to(device)

    ce_loss = nn.CrossEntropyLoss(
        weight=class_weights(np.load(args.label_dir / "y_regime_train.npy", mmap_mode="r"), len(manifest["regime_classes"])).to(device)
    )
    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=multilabel_pos_weight(np.load(args.label_dir / "y_multilabel_train.npy", mmap_mode="r")).to(device)
    )
    scalar_loss = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val = float("inf")
    best_path = args.output_dir / "psc_decision_model_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            optimizer,
            ce_loss,
            bce_loss,
            scalar_loss,
            device,
            args,
            args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            loaders["validation"],
            None,
            ce_loss,
            bce_loss,
            scalar_loss,
            device,
            args,
            args.max_eval_batches,
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "source_metadata": source_meta,
                    "decision_manifest": manifest,
                    "scalar_mean": scalar_mean.tolist(),
                    "scalar_std": scalar_std.tolist(),
                },
                best_path,
            )

    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    eval_summary = {}
    for split in ("validation", "test"):
        eval_summary[split] = evaluate_predictions(
            model,
            loaders[split],
            device,
            scalar_mean,
            scalar_std,
            manifest,
            args.max_eval_batches,
        )
        pd.DataFrame(
            eval_summary[split]["confusion_matrix"],
            index=manifest["regime_classes"],
            columns=manifest["regime_classes"],
        ).to_csv(args.output_dir / f"confusion_{split}.csv")
        pd.DataFrame(eval_summary[split]["multilabel_at_0p5"]).to_csv(
            args.output_dir / f"multilabel_{split}_at_0p5.csv",
            index=False,
        )

    summary = {
        "best_validation_loss": best_val,
        "device": str(device),
        "output_checkpoint": str(best_path),
        "eval": eval_summary,
    }
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Wrote PSC decision model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
