#!/usr/bin/env python3
"""Train replay-derived control-value heads on top of a PSC encoder.

The base PSC model predicts regime/event structure from history. This adapter
keeps that encoder fixed and learns labels measured from actual closed-vs-
deadband replay windows:

- closed_pump_waste_m3: closed-loop pump work over the horizon.
- deadband_opportunity_m3: pump work avoided by the retained deadband replay.
- relax_event_exposure_risk_s: extra time>5/fallback exposure when relaxing.

Replay labels are sparse and currently come from retained D1/C3 casebooks, so
the adapter uses an episode-grouped split inside those labeled windows. This is
not a replacement for the full PSC model; it is a control-value add-on for the
supervisory selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modeling.train_psc_decision_model_v1 import PscDecisionModel, choose_device, set_seed


TARGET_COLUMNS = [
    "closed_pump_waste_m3",
    "deadband_opportunity_m3",
    "relax_event_exposure_risk_s",
]


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
        "--replay-label-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/replay_value_labels_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wind_prediction/psc_decision_model_v1/replay_value_adapter_gru_v1"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--risk-loss-weight", type=float, default=1.50)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def stable_group_bucket(group_key: str) -> float:
    digest = hashlib.sha1(group_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def add_adapter_split(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["group_key"] = rows["family"].astype(str) + "|" + rows["episode_start"].astype(str)
    groups = rows[["group_key", "family", "episode_start"]].drop_duplicates().copy()
    groups["bucket"] = groups["group_key"].map(stable_group_bucket)
    groups["adapter_split"] = np.where(
        groups["bucket"] < 0.70,
        "adapter_train",
        np.where(groups["bucket"] < 0.85, "adapter_validation", "adapter_holdout"),
    )

    # Guard against unlucky tiny-family hashes.
    counts = groups["adapter_split"].value_counts()
    if any(counts.get(split, 0) == 0 for split in ("adapter_train", "adapter_validation", "adapter_holdout")):
        groups = groups.sort_values(["family", "episode_start"]).reset_index(drop=True)
        n = len(groups)
        train_end = max(1, int(round(n * 0.70)))
        val_end = max(train_end + 1, int(round(n * 0.85)))
        val_end = min(val_end, n - 1)
        groups["adapter_split"] = "adapter_holdout"
        groups.loc[: train_end - 1, "adapter_split"] = "adapter_train"
        groups.loc[train_end : val_end - 1, "adapter_split"] = "adapter_validation"

    return rows.merge(groups[["group_key", "adapter_split"]], on="group_key", how="left")


class ReplayValueDataset(Dataset):
    def __init__(
        self,
        x_test_path: Path,
        rows: pd.DataFrame,
        target_log_mean: np.ndarray,
        target_log_std: np.ndarray,
    ) -> None:
        self.x = np.load(x_test_path, mmap_mode="r")
        self.indices = rows["split_row"].to_numpy(dtype=np.int64)
        raw = rows[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        self.target_raw = np.maximum(raw, 0.0)
        target_log = np.log1p(self.target_raw)
        self.target = (target_log - target_log_mean.astype("float32")) / np.maximum(
            target_log_std.astype("float32"), 1e-6
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row_idx = int(self.indices[idx])
        return {
            "x": torch.from_numpy(np.array(self.x[row_idx], dtype=np.float32, copy=True)),
            "target": torch.from_numpy(np.array(self.target[idx], dtype=np.float32, copy=True)),
            "target_raw": torch.from_numpy(np.array(self.target_raw[idx], dtype=np.float32, copy=True)),
        }


class ReplayValueAdapter(nn.Module):
    def __init__(self, base_model: PscDecisionModel, encoded_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.value_head = nn.Sequential(
            nn.Linear(encoded_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, len(TARGET_COLUMNS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.base_model.encode(x)
        return self.value_head(h)


def build_base_model(checkpoint: dict, source_dataset_dir: Path, device: torch.device) -> PscDecisionModel:
    source_meta = checkpoint["source_metadata"]
    manifest = checkpoint["decision_manifest"]
    model_args = checkpoint["args"]
    input_size = len(source_meta["feature_columns"])
    base = PscDecisionModel(
        input_size=input_size,
        hidden_size=int(model_args["hidden_size"]),
        num_layers=int(model_args["num_layers"]),
        num_regimes=len(manifest["regime_classes"]),
        num_multilabel=len(manifest["multilabel_columns"]),
        num_scalar=len(manifest["scalar_columns"]),
        dropout=float(model_args.get("dropout", 0.10)),
        encoder_type=str(model_args.get("encoder", "gru")),
        tcn_kernel_size=int(model_args.get("tcn_kernel_size", 3)),
    )
    base.load_state_dict(checkpoint["model_state"])
    base.to(device)
    base.eval()

    metadata = json.loads((source_dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    if len(metadata["feature_columns"]) != input_size:
        raise SystemExit("Source dataset feature count does not match base checkpoint.")
    return base


def prediction_metrics(pred_raw: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, name in enumerate(TARGET_COLUMNS):
        y = true_raw[:, idx]
        p = pred_raw[:, idx]
        out[f"{name}_mae"] = float(np.mean(np.abs(p - y)))
        out[f"{name}_median_ae"] = float(np.median(np.abs(p - y)))
        if np.std(y) > 1e-8 and np.std(p) > 1e-8:
            out[f"{name}_pearson"] = float(np.corrcoef(y, p)[0, 1])
        else:
            out[f"{name}_pearson"] = float("nan")
    opportunity = true_raw[:, 1]
    pred_opportunity = pred_raw[:, 1]
    if len(opportunity) >= 10:
        top_n = max(1, int(round(len(opportunity) * 0.20)))
        top_idx = np.argsort(pred_opportunity)[-top_n:]
        out["top20pct_predicted_opportunity_true_mean_m3"] = float(opportunity[top_idx].mean())
        out["all_opportunity_true_mean_m3"] = float(opportunity.mean())
        out["top20pct_opportunity_lift"] = float(opportunity[top_idx].mean() / max(opportunity.mean(), 1e-6))
    return out


def evaluate(
    model: ReplayValueAdapter,
    loader: DataLoader,
    device: torch.device,
    target_log_mean: np.ndarray,
    target_log_std: np.ndarray,
    loss_fn: nn.Module,
    risk_loss_weight: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = []
    pred_raw_parts = []
    true_raw_parts = []
    weights = torch.tensor([1.0, 1.0, risk_loss_weight], dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            pred = model(x)
            loss = (loss_fn(pred, target) * weights).mean()
            losses.append(float(loss.detach().cpu()))
            pred_log = pred.detach().cpu().numpy() * target_log_std + target_log_mean
            pred_raw_parts.append(np.maximum(np.expm1(pred_log), 0.0))
            true_raw_parts.append(batch["target_raw"].numpy())
    pred_raw = np.concatenate(pred_raw_parts, axis=0) if pred_raw_parts else np.zeros((0, len(TARGET_COLUMNS)))
    true_raw = np.concatenate(true_raw_parts, axis=0) if true_raw_parts else np.zeros_like(pred_raw)
    metrics = prediction_metrics(pred_raw, true_raw) if len(pred_raw) else {}
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics["loss"], metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    set_seed(args.seed, device)

    rows = pd.read_csv(args.replay_label_dir / "replay_value_rows.csv")
    rows = rows.loc[rows["split"].eq("test")].copy()
    rows = rows.dropna(subset=["split_row", *TARGET_COLUMNS, "family", "episode_start"])
    rows = add_adapter_split(rows)
    rows.to_csv(args.output_dir / "replay_value_adapter_rows_with_split.csv", index=False)

    split_counts = rows.groupby(["adapter_split", "family"]).size().reset_index(name="rows")
    split_counts.to_csv(args.output_dir / "adapter_split_counts.csv", index=False)

    train_rows = rows.loc[rows["adapter_split"].eq("adapter_train")].copy()
    target_log = np.log1p(np.maximum(train_rows[TARGET_COLUMNS].to_numpy(dtype=np.float32), 0.0))
    target_log_mean = target_log.mean(axis=0).astype("float32")
    target_log_std = np.maximum(target_log.std(axis=0).astype("float32"), 1e-6)

    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base = build_base_model(checkpoint, args.source_dataset_dir, device)
    encoded_size = int(checkpoint["args"]["hidden_size"])
    model = ReplayValueAdapter(base, encoded_size, args.hidden_size, args.dropout).to(device)

    datasets = {
        split: ReplayValueDataset(
            args.source_dataset_dir / "X_test.npy",
            rows.loc[rows["adapter_split"].eq(split)].copy(),
            target_log_mean,
            target_log_std,
        )
        for split in ("adapter_train", "adapter_validation", "adapter_holdout")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "adapter_train"), num_workers=0)
        for split, dataset in datasets.items()
    }

    optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    weights = torch.tensor([1.0, 1.0, args.risk_loss_weight], dtype=torch.float32, device=device)

    history = []
    best_val = float("inf")
    stale_epochs = 0
    best_path = args.output_dir / "psc_replay_value_adapter_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["adapter_train"]:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            pred = model(x)
            loss = (loss_fn(pred, target) * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.value_head.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss, val_metrics = evaluate(
            model,
            loaders["adapter_validation"],
            device,
            target_log_mean,
            target_log_std,
            loss_fn,
            args.risk_loss_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_loss + args.min_delta < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "adapter_state": model.value_head.state_dict(),
                    "base_checkpoint": str(args.base_checkpoint),
                    "target_columns": TARGET_COLUMNS,
                    "target_log_mean": target_log_mean.tolist(),
                    "target_log_std": target_log_std.tolist(),
                    "args": vars(args),
                },
                best_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    pd.DataFrame(history).to_csv(args.output_dir / "adapter_training_history.csv", index=False)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.value_head.load_state_dict(best_checkpoint["adapter_state"])
    summary = {
        "device": str(device),
        "best_validation_loss": best_val,
        "target_columns": TARGET_COLUMNS,
        "target_log_mean": target_log_mean.tolist(),
        "target_log_std": target_log_std.tolist(),
        "split_counts": split_counts.to_dict(orient="records"),
        "eval": {},
    }
    for split, loader in loaders.items():
        _, metrics = evaluate(
            model,
            loader,
            device,
            target_log_mean,
            target_log_std,
            loss_fn,
            args.risk_loss_weight,
        )
        summary["eval"][split] = metrics

    (args.output_dir / "adapter_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (args.output_dir / "adapter_training_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote replay value adapter artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
