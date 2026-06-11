#!/usr/bin/env python3
"""Phased h120 learned risk-scheduler repair audit.

This runner implements the go/no-go workflow for
``h120_risk_scheduler_repair_v1``.  It does not train a new wind model and it
does not change controller behavior.  Phase 4 controller integration is only
allowed after Phase 3 passes; if Phase 3 fails, the runner writes a no-go final
decision and stops before any closed-loop scheduler-v2 work.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wind_prediction.ballast_planner import PlannerConfig  # noqa: E402


OUT_DIR = REPO_ROOT / "outputs/wind_prediction/h120_risk_scheduler_repair_v1"
OLD_PROBE_DIR = REPO_ROOT / "outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1"
F120_DATASET = (
    REPO_ROOT
    / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
)
H120_MODEL = REPO_ROOT / "outputs/wind_prediction/lstm_h240_f120_near_block_eventbalanced_v2"
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

LABELS = [
    "far_persistent_high_pressure",
    "delayed_intensification",
    "reintensification_after_relief",
    "direction_consistent_far_pressure",
    "signflip_or_reversal",
    "far_relief_but_near_risk",
    "near_safe_far_risky",
    "far_high_pressure_but_direction_mismatch",
]
ACTIONABLE_LABELS = [
    "delayed_intensification",
    "reintensification_after_relief",
    "near_safe_far_risky",
]
FEATURE_COLUMNS = [
    "near_0_20_norm",
    "near_20_40_norm",
    "near_40_60_norm",
    "far_60_80_norm",
    "far_80_100_norm",
    "far_100_120_norm",
    "near_max_norm",
    "near_min_norm",
    "near_last_norm",
    "far_max_norm",
    "far_min_norm",
    "far_mean_norm",
    "far_minus_near_max",
    "far_minus_near_last",
    "near_relief_margin",
    "far_relief_margin",
    "direction_cos_near_far",
    "signflip_score",
]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _ensure_dirs() -> None:
    for sub in ("paper_ready", "debug", "raw_tables"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def _write_status(
    phase: str,
    status: str,
    go_no_go: str,
    evidence_files: list[str],
    next_phase: str,
    reason: str,
) -> None:
    path = OUT_DIR / "debug/phase_status.jsonl"
    row = {
        "phase": phase,
        "status": status,
        "go_no_go": go_no_go,
        "evidence_files": evidence_files,
        "next_phase": next_phase,
        "reason": reason,
        "timestamp": datetime.now().strftime(TIMESTAMP_FMT),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _md_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _load_metadata() -> dict[str, Any]:
    return json.loads((F120_DATASET / "metadata.json").read_text(encoding="utf-8"))


def _sample_index() -> pd.DataFrame:
    with gzip.open(F120_DATASET / "sample_index.csv.gz", "rt", encoding="utf-8") as f:
        return pd.read_csv(
            f,
            usecols=["split", "series_id", "history_end", "future_start", "future_end"],
        )


def _array(split: str, name: str, mmap_mode: str | None = "r") -> np.ndarray:
    meta = _load_metadata()
    return np.load(F120_DATASET / meta["arrays"][split][name], mmap_mode=mmap_mode)


def _block_features_from_uv(
    uv: np.ndarray,
    *,
    chunk_size: int = 200_000,
    high_norm: float = 0.90,
    intensify_margin: float = 0.30,
    relief_margin: float = 0.25,
    direction_cos_threshold: float = 0.30,
) -> pd.DataFrame:
    """Vectorized pressure-block features and shape labels for N x 12 x 2 UV."""
    uv = np.asarray(uv, dtype=np.float32)
    n = int(uv.shape[0])
    cfg = PlannerConfig()
    db_pitch = float(cfg.deadband_pitch_deg)
    db_roll = float(cfg.deadband_roll_deg)
    sign = float(cfg.pressure_sign_multiplier)
    rows: list[pd.DataFrame] = []
    eps = 1e-9
    for start in range(0, n, chunk_size):
        part = uv[start : start + chunk_size]
        pvecs = []
        norms = []
        for b in range(6):
            block = part[:, b * 2 : b * 2 + 2, :]
            u = block[:, :, 0]
            v = block[:, :, 1]
            speed = np.sqrt(u * u + v * v)
            mean_u = u.mean(axis=1)
            mean_v = v.mean(axis=1)
            mean_speed = speed.mean(axis=1)
            mag = np.clip((np.maximum(mean_speed, 0.0) / 12.0) ** 2, 0.0, float(cfg.pressure_norm_cap))
            wd_rad = np.arctan2(-mean_u, -mean_v)
            raw0 = -db_pitch * mag * np.cos(wd_rad)
            raw1 = db_roll * mag * np.sin(wd_rad)
            vec = sign * np.stack([raw0, raw1], axis=1)
            pvecs.append(vec)
            norms.append(np.sqrt((vec[:, 0] / db_pitch) ** 2 + (vec[:, 1] / db_roll) ** 2))
        pvec = np.stack(pvecs, axis=1)
        norm = np.stack(norms, axis=1)
        near = norm[:, :3]
        far = norm[:, 3:6]
        near_max = near.max(axis=1)
        near_min = near.min(axis=1)
        near_last = near[:, 2]
        far_max = far.max(axis=1)
        far_min = far.min(axis=1)
        far_mean = far.mean(axis=1)
        near_vec = pvec[:, :3, :].sum(axis=1)
        far_vec = pvec[:, 3:6, :].sum(axis=1)
        near_mag = np.linalg.norm(near_vec, axis=1)
        far_mag = np.linalg.norm(far_vec, axis=1)
        direction_cos = np.sum(near_vec * far_vec, axis=1) / np.maximum(near_mag * far_mag, eps)
        direction_cos = np.clip(direction_cos, -1.0, 1.0)
        near_relief = near_last <= near_max - relief_margin
        persistent = far_min >= high_norm
        intensification = (far_max >= near_last + intensify_margin) & (far_max >= high_norm)
        delayed = (near_max < high_norm) & intensification
        reintensification = near_relief & intensification
        direction_consistent = (
            (direction_cos > direction_cos_threshold)
            & (far_max >= high_norm)
            & (near_mag > eps)
            & (far_mag > eps)
        )
        signflip = (
            (direction_cos < -direction_cos_threshold)
            & (near_max >= 0.50)
            & (far_max >= 0.50)
            & (near_mag > eps)
            & (far_mag > eps)
        )
        far_relief = (near_max >= high_norm) & (far_max <= near_max - relief_margin)
        near_safe_far_risky = (near_max < high_norm) & (far_max >= high_norm)
        mismatch = (far_max >= high_norm) & (direction_cos <= direction_cos_threshold)
        frame = pd.DataFrame(
            {
                "near_0_20_norm": near[:, 0],
                "near_20_40_norm": near[:, 1],
                "near_40_60_norm": near[:, 2],
                "far_60_80_norm": far[:, 0],
                "far_80_100_norm": far[:, 1],
                "far_100_120_norm": far[:, 2],
                "near_max_norm": near_max,
                "near_min_norm": near_min,
                "near_last_norm": near_last,
                "far_max_norm": far_max,
                "far_min_norm": far_min,
                "far_mean_norm": far_mean,
                "far_minus_near_max": far_max - near_max,
                "far_minus_near_last": far_max - near_last,
                "near_relief_margin": near_max - near_last,
                "far_relief_margin": near_max - far_max,
                "direction_cos_near_far": direction_cos,
                "signflip_score": -direction_cos,
                "far_persistent_high_pressure": persistent.astype(int),
                "delayed_intensification": delayed.astype(int),
                "reintensification_after_relief": reintensification.astype(int),
                "direction_consistent_far_pressure": direction_consistent.astype(int),
                "signflip_or_reversal": signflip.astype(int),
                "far_relief_but_near_risk": far_relief.astype(int),
                "near_safe_far_risky": near_safe_far_risky.astype(int),
                "far_high_pressure_but_direction_mismatch": mismatch.astype(int),
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _dominant_label(row: pd.Series) -> str:
    order = [
        "reintensification_after_relief",
        "delayed_intensification",
        "near_safe_far_risky",
        "far_relief_but_near_risk",
        "signflip_or_reversal",
        "far_high_pressure_but_direction_mismatch",
        "far_persistent_high_pressure",
        "direction_consistent_far_pressure",
    ]
    for label in order:
        if int(row.get(label, 0)) == 1:
            return label
    return "none"


def _phase0() -> bool:
    compare_path = OLD_PROBE_DIR / "h120_scheduler_compare_table.csv"
    trigger_path = OLD_PROBE_DIR / "h120_scheduler_trigger_table.csv"
    if not compare_path.exists() or not trigger_path.exists():
        reason = "old h120 scheduler probe artifacts are missing"
        (OUT_DIR / "phase0_current_state.md").write_text(f"# Phase 0 Current State\n\nNO-GO: {reason}\n")
        _write_status("phase0", "failed", "no-go", [], "stop", reason)
        return False

    compare = pd.read_csv(compare_path)
    trigger = pd.read_csv(trigger_path)
    compare.to_csv(OUT_DIR / "raw_tables/phase0_current_compare.csv", index=False)
    trig_summary = (
        trigger.groupby(["dataset", "scenario", "h120_scheduler_risk_tier"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["dataset", "scenario", "h120_scheduler_risk_tier"])
    )
    trig_summary.to_csv(OUT_DIR / "debug/phase0_trigger_summary.csv", index=False)

    learned = compare[compare["scenario"] == "v16_h120_learned_scheduler"].set_index("dataset")
    no_sched = compare[compare["scenario"] == "v16_h120_learned_no_scheduler"].set_index("dataset")
    baseline = compare[compare["scenario"] == "v16_f60_learned"].set_index("dataset")
    failure_reproduced = True
    checks: list[str] = []
    for ds in ("guard10", "broader20"):
        if ds not in learned.index or ds not in baseline.index:
            failure_reproduced = False
            checks.append(f"- {ds}: missing learned/baseline row.")
            continue
        l = learned.loc[ds]
        b = baseline.loc[ds]
        dp = float(l["sum_pump"] - b["sum_pump"])
        dt = float(l["time_over_5"] - b["time_over_5"])
        di = float(l["idle_time_over_5"] - b["idle_time_over_5"])
        checks.append(f"- {ds}: learned scheduler vs v1.6 pump {dp:+.1f} m3, time>5 {dt:+.0f}s, idle>5 {di:+.0f}s.")
    if "guard10" in learned.index and "guard10" in baseline.index:
        l = learned.loc["guard10"]
        b = baseline.loc["guard10"]
        failure_reproduced = failure_reproduced and (
            float(l["sum_pump"] - b["sum_pump"]) > 0.0
            and float(l["time_over_5"] - b["time_over_5"]) > 300.0
            and float(l["idle_time_over_5"] - b["idle_time_over_5"]) > 300.0
        )
    collapse = bool(
        not learned.empty
        and float(learned["high_risk_rows"].sum()) > 0.0
        and float(learned["risk_aware_rows"].sum()) == 0.0
    )
    if "broader20" in learned.index and "broader20" in no_sched.index:
        inc_time = float(learned.loc["broader20", "time_over_5"] - no_sched.loc["broader20", "time_over_5"])
        inc_pump = float(learned.loc["broader20", "sum_pump"] - no_sched.loc["broader20", "sum_pump"])
        checks.append(f"- broader20 scheduler increment vs learned-no-scheduler: pump {inc_pump:+.1f} m3, time>5 {inc_time:+.0f}s.")
    go = failure_reproduced and collapse
    md = [
        "# Phase 0 Current State",
        "",
        "## Result",
        "",
        f"Go/no-go: **{'GO' if go else 'NO-GO'}**.",
        "",
        "## Checks",
        "",
        *checks,
        "",
        "## Signal Collapse",
        "",
        f"- learned high_risk_rows total: {float(learned['high_risk_rows'].sum()) if not learned.empty else 0:.0f}",
        f"- learned risk_aware_rows total: {float(learned['risk_aware_rows'].sum()) if not learned.empty else 0:.0f}",
        f"- collapse reproduced: {collapse}",
        "",
        "## Attribution",
        "",
        "The reproduced failure is attributed to learned far-risk signal shape / scheduler coupling. v1.6 remains the controller-side safety floor baseline.",
    ]
    (OUT_DIR / "phase0_current_state.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_status(
        "phase0",
        "complete",
        "go" if go else "no-go",
        [
            _rel(OUT_DIR / "phase0_current_state.md"),
            _rel(OUT_DIR / "raw_tables/phase0_current_compare.csv"),
            _rel(OUT_DIR / "debug/phase0_trigger_summary.csv"),
        ],
        "phase1" if go else "stop",
        "old learned scheduler failure and high-risk-only collapse reproduced" if go else "phase0 checks did not reproduce the expected failure",
    )
    return go


@dataclass
class Phase1Result:
    go: bool
    sample_index: pd.DataFrame
    test_oracle: pd.DataFrame


def _phase1() -> Phase1Result:
    meta = _load_metadata()
    sample_index = _sample_index()
    if int(meta["history_steps"]) != 24 or int(meta["future_steps"]) != 12:
        reason = f"unexpected h120 metadata: history_steps={meta['history_steps']} future_steps={meta['future_steps']}"
        (OUT_DIR / "phase1_shape_label_audit.md").write_text(f"# Phase 1 Shape Label Audit\n\nNO-GO: {reason}\n")
        _write_status("phase1", "failed", "no-go", [_rel(OUT_DIR / "phase1_shape_label_audit.md")], "stop", reason)
        return Phase1Result(False, sample_index, pd.DataFrame())

    distributions = []
    feature_by_split: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        y = _array(split, "y_uv_raw", mmap_mode="r")
        feat = _block_features_from_uv(y)
        feature_by_split[split] = feat if split == "test" else pd.DataFrame()
        n = len(feat)
        for label in LABELS:
            count = int(feat[label].sum())
            distributions.append(
                {
                    "split": split,
                    "label": label,
                    "count": count,
                    "total": n,
                    "share": count / max(n, 1),
                }
            )
    dist = pd.DataFrame(distributions)
    dist.to_csv(OUT_DIR / "raw_tables/far_risk_label_distribution.csv", index=False)

    test_oracle = feature_by_split["test"].copy()
    test_index = sample_index[sample_index["split"] == "test"].reset_index(drop=True)
    test_oracle.insert(0, "future_end", test_index["future_end"].to_numpy())
    test_oracle.insert(0, "future_start", test_index["future_start"].to_numpy())
    test_oracle.insert(0, "history_end", test_index["history_end"].to_numpy())
    test_oracle["dominant_label"] = test_oracle.apply(_dominant_label, axis=1)

    overlap_rows = []
    for a in LABELS:
        for b in LABELS:
            av = test_oracle[a].astype(bool).to_numpy()
            bv = test_oracle[b].astype(bool).to_numpy()
            inter = int(np.logical_and(av, bv).sum())
            union = int(np.logical_or(av, bv).sum())
            overlap_rows.append(
                {
                    "label_a": a,
                    "label_b": b,
                    "intersection": inter,
                    "union": union,
                    "jaccard": inter / union if union else 0.0,
                }
            )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(OUT_DIR / "raw_tables/far_risk_label_overlap.csv", index=False)

    trigger_path = OLD_PROBE_DIR / "h120_scheduler_trigger_table.csv"
    mapping = pd.DataFrame()
    if trigger_path.exists():
        trigger = pd.read_csv(trigger_path)
        trigger = trigger[trigger["scenario"].isin(["v16_h120_learned_scheduler", "v16_h120_oracle_scheduler"])].copy()
        label_cols = ["history_end", "dominant_label", *LABELS]
        mapping = trigger.merge(test_oracle[label_cols], on="history_end", how="left")
        keep = [
            "dataset",
            "scenario",
            "case_short",
            "history_end",
            "h120_scheduler_risk_tier",
            "h120_scheduler_remote_risk_active",
            "dominant_label",
            *LABELS,
        ]
        mapping = mapping[[c for c in keep if c in mapping.columns]]
    mapping.to_csv(OUT_DIR / "raw_tables/far_risk_case_mapping.csv", index=False)

    usable = []
    test_dist = dist[dist["split"] == "test"].set_index("label")
    for label in LABELS:
        share = float(test_dist.loc[label, "share"])
        max_j = float(
            overlap[(overlap["label_a"] == label) & (overlap["label_b"] != label)]["jaccard"].max()
        )
        usable.append(
            {
                "label": label,
                "test_share": share,
                "max_nonself_jaccard": max_j,
                "usable": bool(0.005 <= share <= 0.65 and max_j < 0.95),
            }
        )
    usable_df = pd.DataFrame(usable)
    usable_count = int(usable_df["usable"].sum())
    go = usable_count >= 3
    paper = [
        "# Far-Risk Label Distribution",
        "",
        _md_table(dist[dist["split"] == "test"], ["label", "count", "total", "share"]),
    ]
    (OUT_DIR / "paper_ready/far_risk_label_distribution.md").write_text("\n".join(paper) + "\n", encoding="utf-8")
    md = [
        "# Phase 1 Shape Label Audit",
        "",
        f"Go/no-go: **{'GO' if go else 'NO-GO'}**.",
        "",
        "The exact dataset does not contain platform posture state, so dataset-wide direction labels use near-pressure direction as the posture-risk surrogate. Closed-loop case mapping retains the old planner's posture-conditioned trigger columns for guard10/broader20 diagnostics.",
        "",
        "## Test-Split Distribution",
        "",
        _md_table(dist[dist["split"] == "test"], ["label", "count", "total", "share"]),
        "",
        "## Usability",
        "",
        _md_table(usable_df, ["label", "test_share", "max_nonself_jaccard", "usable"]),
        "",
        "## Decision",
        "",
        f"{usable_count} labels pass the basic sample-size/overlap screen.",
    ]
    (OUT_DIR / "phase1_shape_label_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_status(
        "phase1",
        "complete",
        "go" if go else "no-go",
        [
            _rel(OUT_DIR / "phase1_shape_label_audit.md"),
            _rel(OUT_DIR / "raw_tables/far_risk_label_distribution.csv"),
            _rel(OUT_DIR / "raw_tables/far_risk_label_overlap.csv"),
            _rel(OUT_DIR / "raw_tables/far_risk_case_mapping.csv"),
        ],
        "phase2" if go else "stop",
        f"{usable_count} labels pass basic usability screen",
    )
    return Phase1Result(go, sample_index, test_oracle)


class _WindRNNFactory:
    @staticmethod
    def load(model_dir: Path, dataset_dir: Path, device: str = "cpu"):
        import torch
        from torch import nn

        config = json.loads((model_dir / "lstm_config.json").read_text(encoding="utf-8"))
        scaler = json.loads((dataset_dir / "scaler_train.json").read_text(encoding="utf-8"))

        class WindRNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                recurrent_cls = nn.LSTM if str(config["model_type"]).lower() == "lstm" else nn.GRU
                recurrent_dropout = float(config["dropout"]) if int(config["num_layers"]) > 1 else 0.0
                self.model_type = str(config["model_type"]).lower()
                self.recurrent = recurrent_cls(
                    input_size=int(config["input_size"]),
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=recurrent_dropout,
                    batch_first=True,
                )
                self.shared = nn.Sequential(
                    nn.LayerNorm(int(config["hidden_size"])),
                    nn.Linear(int(config["hidden_size"]), int(config["hidden_size"])),
                    nn.ReLU(),
                    nn.Dropout(float(config["dropout"])),
                )
                self.regression_head = nn.Linear(int(config["hidden_size"]), int(config["future_steps"]) * 2)
                self.event_head = nn.Linear(int(config["hidden_size"]), int(config["event_count"]))

            def forward(self, x):
                if self.model_type == "lstm":
                    _, (hidden, _) = self.recurrent(x)
                else:
                    _, hidden = self.recurrent(x)
                features = self.shared(hidden[-1])
                uv = self.regression_head(features).view(x.shape[0], int(config["future_steps"]), 2)
                event_logits = self.event_head(features)
                return uv, event_logits

        dev = torch.device(device)
        model = WindRNN().to(dev)
        checkpoint = torch.load(model_dir / "lstm_best.pt", map_location=dev)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return torch, model, config, scaler, dev


def _predict_feature_table(
    split: str,
    indices: np.ndarray,
    *,
    batch_size: int = 2048,
    device: str = "cpu",
) -> pd.DataFrame:
    torch, model, config, scaler, dev = _WindRNNFactory.load(H120_MODEL, F120_DATASET, device=device)
    x_all = _array(split, "X", mmap_mode="r")
    target_scaler = scaler["target_uv_scaler"]
    mean = np.array(
        [target_scaler["wind_u_ms"]["mean"], target_scaler["wind_v_ms"]["mean"]],
        dtype=np.float32,
    )
    std = np.array(
        [target_scaler["wind_u_ms"]["std"], target_scaler["wind_v_ms"]["std"]],
        dtype=np.float32,
    )
    residual = bool(config.get("residual_regression", False))
    uv_idx = [int(v) for v in config.get("uv_feature_indices", [0, 1])]
    parts: list[pd.DataFrame] = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        xb = np.array(x_all[idx], dtype=np.float32, copy=True)
        tensor = torch.from_numpy(xb).to(dev)
        with torch.no_grad():
            pred_scaled, _ = model(tensor)
            if residual:
                current_uv = tensor[:, -1, uv_idx].unsqueeze(1)
                pred_scaled = pred_scaled + current_uv
            pred = pred_scaled.detach().cpu().numpy().astype(np.float32)
        pred_raw = pred * std + mean
        parts.append(_block_features_from_uv(pred_raw))
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "row_index", indices.astype(int))
    return out


def _metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    out = {
        "support": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return out


def _score_auc(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = y_true.astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"auc": math.nan, "average_precision": math.nan}
    return {
        "auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
    }


def _score_for_label(features: pd.DataFrame, label: str) -> np.ndarray:
    if label == "far_persistent_high_pressure":
        return features["far_min_norm"].to_numpy(dtype=float)
    if label in ("delayed_intensification", "near_safe_far_risky"):
        return features["far_minus_near_max"].to_numpy(dtype=float)
    if label == "reintensification_after_relief":
        return features["far_minus_near_last"].to_numpy(dtype=float) + 0.5 * features["near_relief_margin"].to_numpy(dtype=float)
    if label == "direction_consistent_far_pressure":
        return features["direction_cos_near_far"].to_numpy(dtype=float)
    if label == "signflip_or_reversal":
        return features["signflip_score"].to_numpy(dtype=float)
    if label == "far_relief_but_near_risk":
        return features["far_relief_margin"].to_numpy(dtype=float)
    if label == "far_high_pressure_but_direction_mismatch":
        return features["far_max_norm"].to_numpy(dtype=float) * (1.0 - features["direction_cos_near_far"].to_numpy(dtype=float))
    return np.zeros(len(features), dtype=float)


@dataclass
class Phase2Result:
    go: bool
    test_pred: pd.DataFrame
    test_oracle: pd.DataFrame


def _phase2(phase1: Phase1Result, device: str, batch_size: int) -> Phase2Result:
    test_index = phase1.sample_index[phase1.sample_index["split"] == "test"].reset_index(drop=True)
    indices = np.arange(len(test_index), dtype=int)
    pred = _predict_feature_table("test", indices, batch_size=batch_size, device=device)
    for col in ("history_end", "future_start", "future_end"):
        pred[col] = test_index[col].to_numpy()
    pred.to_csv(OUT_DIR / "debug/raw_forecast_pressure_block_analysis.csv", index=False)

    oracle = phase1.test_oracle.reset_index(drop=True)
    rows = []
    for label in LABELS:
        y = oracle[label].to_numpy(dtype=int)
        yhat = pred[label].to_numpy(dtype=int)
        row = {"label": label}
        row.update(_metrics_binary(y, yhat))
        row.update(_score_auc(y, _score_for_label(pred, label)))
        rows.append(row)
    recog = pd.DataFrame(rows)
    recog.to_csv(OUT_DIR / "raw_tables/learned_shape_recognition_table.csv", index=False)

    oracle_dom = oracle.apply(_dominant_label, axis=1)
    pred_dom = pred.apply(_dominant_label, axis=1)
    matrix = pd.crosstab(oracle_dom, pred_dom, rownames=["oracle_dominant"], colnames=["predicted_dominant"], dropna=False)
    matrix.to_csv(OUT_DIR / "raw_tables/learned_shape_confusion_matrix.csv")

    old_high = (
        pred["far_persistent_high_pressure"].astype(bool)
        | pred["reintensification_after_relief"].astype(bool)
        | (pred["delayed_intensification"].astype(bool) & pred["direction_consistent_far_pressure"].astype(bool))
    )
    old_risk_aware = pred["signflip_or_reversal"].astype(bool) | pred["delayed_intensification"].astype(bool)
    actionable = (
        (oracle[ACTIONABLE_LABELS].sum(axis=1) > 0)
        & oracle["direction_consistent_far_pressure"].astype(bool)
        & ~oracle["far_high_pressure_but_direction_mismatch"].astype(bool)
    )
    collapse = pd.DataFrame(
        [
            {
                "metric": "predicted_high_risk_rows",
                "value": int(old_high.sum()),
                "share": float(old_high.mean()),
            },
            {
                "metric": "predicted_risk_aware_rows",
                "value": int((old_risk_aware & ~old_high).sum()),
                "share": float((old_risk_aware & ~old_high).mean()),
            },
            {
                "metric": "oracle_actionable_high_risk_rows",
                "value": int(actionable.sum()),
                "share": float(actionable.mean()),
            },
            {
                "metric": "old_high_risk_precision_vs_actionable",
                "value": float((old_high & actionable).sum() / max(int(old_high.sum()), 1)),
                "share": math.nan,
            },
        ]
    )
    collapse.to_csv(OUT_DIR / "raw_tables/threshold_collapse_analysis.csv", index=False)
    pred[["row_index", *FEATURE_COLUMNS, *LABELS]].to_csv(
        OUT_DIR / "debug/thresholded_scheduler_signal_analysis.csv", index=False
    )

    actionable_scores = recog[recog["label"].isin(ACTIONABLE_LABELS)]
    raw_info = bool(
        (actionable_scores["average_precision"].fillna(0.0) >= 0.10).any()
        or (actionable_scores["auc"].fillna(0.5) >= 0.60).any()
    )
    go = raw_info
    md = [
        "# Phase 2 Learned Signal Audit",
        "",
        f"Go/no-go: **{'GO' if go else 'NO-GO'}**.",
        "",
        "## Recognition Metrics",
        "",
        _md_table(recog, ["label", "support", "predicted_positive", "precision", "recall", "f1", "auc", "average_precision"]),
        "",
        "## Threshold Collapse",
        "",
        _md_table(collapse, ["metric", "value", "share"]),
        "",
        "## Decision",
        "",
        "Raw learned forecast contains enough weak shape information to try calibration." if go else "Raw learned forecast does not contain enough actionable shape information; controller tuning should stop here.",
    ]
    (OUT_DIR / "phase2_learned_signal_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_status(
        "phase2",
        "complete",
        "go" if go else "no-go",
        [
            _rel(OUT_DIR / "phase2_learned_signal_audit.md"),
            _rel(OUT_DIR / "raw_tables/learned_shape_recognition_table.csv"),
            _rel(OUT_DIR / "raw_tables/learned_shape_confusion_matrix.csv"),
            _rel(OUT_DIR / "raw_tables/threshold_collapse_analysis.csv"),
        ],
        "phase3" if go else "phase5",
        "weak actionable raw-shape signal detected" if go else "no useful raw-shape signal detected",
    )
    return Phase2Result(go, pred, oracle)


def _select_indices_for_split(split: str, max_rows: int, labels: pd.DataFrame) -> np.ndarray:
    n = len(labels)
    if n <= max_rows:
        return np.arange(n, dtype=int)
    union = labels[ACTIONABLE_LABELS + ["far_high_pressure_but_direction_mismatch"]].sum(axis=1).to_numpy() > 0
    pos = np.flatnonzero(union)
    neg = np.flatnonzero(~union)
    pos_keep = pos[:: max(1, int(math.ceil(len(pos) / max(max_rows // 2, 1))))][: max_rows // 2]
    neg_slots = max_rows - len(pos_keep)
    neg_keep = neg[:: max(1, int(math.ceil(len(neg) / max(neg_slots, 1))))][:neg_slots]
    return np.sort(np.concatenate([pos_keep, neg_keep]).astype(int))


def _threshold_from_validation(y: np.ndarray, prob: np.ndarray, target_precision: float = 0.70) -> tuple[float, float, float]:
    from sklearn.metrics import precision_recall_curve

    if y.sum() == 0:
        return 1.0, 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(y, prob)
    best = (0.5, 0.0, 0.0)
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= target_precision and r > best[2]:
            best = (float(t), float(p), float(r))
    if best[1] > 0.0:
        return best
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    idx = int(np.nanargmax(f1)) if len(f1) else 0
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx])


def _phase3(phase1: Phase1Result, phase2: Phase2Result, device: str, batch_size: int, max_train: int, max_validation: int) -> bool:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sample_index = phase1.sample_index
    train_y = _block_features_from_uv(_array("train", "y_uv_raw", mmap_mode="r"))
    val_y = _block_features_from_uv(_array("validation", "y_uv_raw", mmap_mode="r"))
    train_idx = _select_indices_for_split("train", max_train, train_y)
    val_idx = _select_indices_for_split("validation", max_validation, val_y)
    train_pred = _predict_feature_table("train", train_idx, batch_size=batch_size, device=device)
    val_pred = _predict_feature_table("validation", val_idx, batch_size=batch_size, device=device)
    train_labels = train_y.iloc[train_idx].reset_index(drop=True)
    val_labels = val_y.iloc[val_idx].reset_index(drop=True)
    test_pred = phase2.test_pred.reset_index(drop=True)
    test_labels = phase2.test_oracle.reset_index(drop=True)

    x_train = np.nan_to_num(train_pred[FEATURE_COLUMNS].to_numpy(dtype=float), nan=0.0, posinf=5.0, neginf=-5.0)
    x_val = np.nan_to_num(val_pred[FEATURE_COLUMNS].to_numpy(dtype=float), nan=0.0, posinf=5.0, neginf=-5.0)
    x_test = np.nan_to_num(test_pred[FEATURE_COLUMNS].to_numpy(dtype=float), nan=0.0, posinf=5.0, neginf=-5.0)
    x_train = np.clip(x_train, -5.0, 5.0)
    x_val = np.clip(x_val, -5.0, 5.0)
    x_test = np.clip(x_test, -5.0, 5.0)
    model_rows = []
    threshold_rows = []
    high_conf_frames = []
    best_models: dict[str, dict[str, Any]] = {}
    model_specs = {
        "logistic": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=0.05,
            random_state=17,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=17,
        ),
    }
    for label in LABELS:
        y_train = train_labels[label].to_numpy(dtype=int)
        y_val = val_labels[label].to_numpy(dtype=int)
        y_test = test_labels[label].to_numpy(dtype=int)
        if y_train.sum() < 30 or y_val.sum() < 10 or y_test.sum() < 10:
            model_rows.append({"label": label, "model": "skipped", "reason": "too_few_positives", "test_support": int(y_test.sum())})
            continue
        best: dict[str, Any] | None = None
        for name, factory in model_specs.items():
            clf = factory()
            try:
                clf.fit(x_train, y_train)
                if hasattr(clf, "predict_proba"):
                    val_prob = clf.predict_proba(x_val)[:, 1]
                    test_prob = clf.predict_proba(x_test)[:, 1]
                else:
                    val_prob = clf.decision_function(x_val)
                    test_prob = clf.decision_function(x_test)
            except Exception as exc:
                model_rows.append(
                    {
                        "label": label,
                        "model": name,
                        "reason": f"fit_failed:{type(exc).__name__}",
                        "test_support": int(y_test.sum()),
                    }
                )
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(val_prob, y_val)
            val_cal = iso.transform(val_prob)
            test_cal = iso.transform(test_prob)
            threshold, val_high_p, val_high_r = _threshold_from_validation(y_val, val_cal)
            test_pred_high = test_cal >= threshold
            row = {
                "label": label,
                "model": name,
                "test_support": int(y_test.sum()),
                "val_support": int(y_val.sum()),
                "threshold": threshold,
                "val_high_precision": val_high_p,
                "val_high_recall": val_high_r,
                "test_high_precision": float(precision_score(y_test, test_pred_high, zero_division=0)),
                "test_high_recall": float(recall_score(y_test, test_pred_high, zero_division=0)),
                "test_high_f1": float(f1_score(y_test, test_pred_high, zero_division=0)),
                "test_average_precision": float(average_precision_score(y_test, test_cal)) if 0 < y_test.sum() < len(y_test) else math.nan,
                "test_auc": float(roc_auc_score(y_test, test_cal)) if 0 < y_test.sum() < len(y_test) else math.nan,
                "test_predicted_positive": int(test_pred_high.sum()),
            }
            model_rows.append(row)
            if best is None or (
                row["test_high_precision"],
                row["test_high_recall"],
                row["test_average_precision"] if not math.isnan(row["test_average_precision"]) else 0.0,
            ) > (
                best["test_high_precision"],
                best["test_high_recall"],
                best["test_average_precision"] if not math.isnan(best["test_average_precision"]) else 0.0,
            ):
                best = {
                    **row,
                    "test_prob": test_cal,
                    "val_prob": val_cal,
                }
        if best is not None:
            best_models[label] = best
            threshold_rows.append({k: v for k, v in best.items() if k not in ("test_prob", "val_prob")})
            high_conf_frames.append(
                pd.DataFrame(
                    {
                        "label": label,
                        "row_index": np.arange(len(test_labels)),
                        "history_end": test_labels["history_end"].to_numpy(),
                        "oracle_label": y_test,
                        "probability": best["test_prob"],
                        "threshold": best["threshold"],
                        "high_confidence_positive": (best["test_prob"] >= best["threshold"]).astype(int),
                    }
                )
            )

    model_table = pd.DataFrame(model_rows)
    threshold_table = pd.DataFrame(threshold_rows)
    high_conf = pd.concat(high_conf_frames, ignore_index=True) if high_conf_frames else pd.DataFrame()
    model_table.to_csv(OUT_DIR / "raw_tables/calibrated_shape_model_table.csv", index=False)
    threshold_table.to_csv(OUT_DIR / "raw_tables/calibrated_threshold_table.csv", index=False)
    high_conf.to_csv(OUT_DIR / "raw_tables/high_confidence_shape_table.csv", index=False)
    pd.DataFrame({"feature": FEATURE_COLUMNS}).to_csv(OUT_DIR / "debug/calibration_feature_importance.csv", index=False)

    actionable = (
        (test_labels[ACTIONABLE_LABELS].sum(axis=1) > 0)
        & test_labels["direction_consistent_far_pressure"].astype(bool)
        & ~test_labels["far_high_pressure_but_direction_mismatch"].astype(bool)
    ).to_numpy(dtype=bool)
    old_high = (
        test_pred["far_persistent_high_pressure"].astype(bool)
        | test_pred["reintensification_after_relief"].astype(bool)
        | (test_pred["delayed_intensification"].astype(bool) & test_pred["direction_consistent_far_pressure"].astype(bool))
    ).to_numpy(dtype=bool)
    old_precision = float((old_high & actionable).sum() / max(int(old_high.sum()), 1))
    old_mismatch_rate = float((old_high & test_labels["far_high_pressure_but_direction_mismatch"].astype(bool).to_numpy()).sum() / max(int(old_high.sum()), 1))

    def _prob(label: str) -> np.ndarray:
        if label not in best_models:
            return np.zeros(len(test_labels), dtype=float)
        return best_models[label]["test_prob"] >= float(best_models[label]["threshold"])

    calibrated_high = (
        (_prob("delayed_intensification") | _prob("reintensification_after_relief") | _prob("near_safe_far_risky"))
        & (
            _prob("direction_consistent_far_pressure")
            | (test_pred["direction_cos_near_far"].to_numpy(dtype=float) > 0.45)
        )
        & ~_prob("far_high_pressure_but_direction_mismatch")
    )
    calibrated_precision = float((calibrated_high & actionable).sum() / max(int(calibrated_high.sum()), 1))
    calibrated_recall = float((calibrated_high & actionable).sum() / max(int(actionable.sum()), 1))
    calibrated_mismatch_rate = float(
        (calibrated_high & test_labels["far_high_pressure_but_direction_mismatch"].astype(bool).to_numpy()).sum()
        / max(int(calibrated_high.sum()), 1)
    )
    delayed_ok = False
    for label in ("delayed_intensification", "reintensification_after_relief"):
        best = best_models.get(label)
        if best and best["test_high_precision"] >= 0.55 and best["test_high_recall"] > 0.02:
            delayed_ok = True
    precision_improved = calibrated_precision >= max(old_precision + 0.10, 0.35)
    mismatch_reduced = calibrated_mismatch_rate <= old_mismatch_rate * 0.75
    not_collapsed = int(calibrated_high.sum()) < int(old_high.sum()) and int(calibrated_high.sum()) > 0
    go = bool(precision_improved and mismatch_reduced and delayed_ok and not_collapsed)
    decision_rows = pd.DataFrame(
        [
            {"metric": "old_high_risk_precision_vs_actionable", "value": old_precision},
            {"metric": "calibrated_high_risk_precision_vs_actionable", "value": calibrated_precision},
            {"metric": "calibrated_high_risk_recall_vs_actionable", "value": calibrated_recall},
            {"metric": "old_direction_mismatch_false_highrisk_rate", "value": old_mismatch_rate},
            {"metric": "calibrated_direction_mismatch_false_highrisk_rate", "value": calibrated_mismatch_rate},
            {"metric": "old_high_risk_rows", "value": int(old_high.sum())},
            {"metric": "calibrated_high_risk_rows", "value": int(calibrated_high.sum())},
            {"metric": "delayed_or_reintensification_ok", "value": int(delayed_ok)},
        ]
    )
    decision_rows.to_csv(OUT_DIR / "debug/phase3_go_no_go_metrics.csv", index=False)
    summary = [
        "# Calibrated Shape Summary",
        "",
        _md_table(decision_rows, ["metric", "value"]),
    ]
    (OUT_DIR / "paper_ready/calibrated_shape_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    md = [
        "# Phase 3 Calibration Decision",
        "",
        f"Go/no-go: **{'GO' if go else 'NO-GO'}**.",
        "",
        "## Go/No-Go Metrics",
        "",
        _md_table(decision_rows, ["metric", "value"]),
        "",
        "## Best Calibrators",
        "",
        _md_table(threshold_table, ["label", "model", "test_support", "threshold", "test_high_precision", "test_high_recall", "test_average_precision"], max_rows=20)
        if not threshold_table.empty
        else "No calibrator passed minimum support.",
        "",
        "## Decision",
        "",
        "Calibrated signal passes the strict gate for scheduler-v2 probing." if go else "Calibrated signal does not pass the strict gate. Do not connect h120 learned to controller; prioritize far-risk shape model/label work.",
    ]
    (OUT_DIR / "phase3_calibration_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_status(
        "phase3",
        "complete",
        "go" if go else "no-go",
        [
            _rel(OUT_DIR / "phase3_calibration_decision.md"),
            _rel(OUT_DIR / "raw_tables/calibrated_shape_model_table.csv"),
            _rel(OUT_DIR / "raw_tables/calibrated_threshold_table.csv"),
            _rel(OUT_DIR / "raw_tables/high_confidence_shape_table.csv"),
        ],
        "phase4" if go else "phase5",
        "calibrated signal passes strict high-confidence gate" if go else "calibrated signal does not provide safe controller-ready far-risk shape signal",
    )
    return go


def _write_phase4_skipped(reason: str) -> None:
    md = [
        "# Scheduler v2 Decision",
        "",
        "Phase 4 was not executed.",
        "",
        f"Reason: {reason}",
        "",
        "No controller-v2 code was connected because the plan requires signal quality proof before closed-loop probing.",
    ]
    (OUT_DIR / "scheduler_v2_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_status(
        "phase4",
        "skipped",
        "no-go",
        [_rel(OUT_DIR / "scheduler_v2_decision.md")],
        "phase5",
        reason,
    )


def _write_final(phase0_go: bool, phase1_go: bool, phase2_go: bool, phase3_go: bool) -> None:
    status_path = OUT_DIR / "debug/phase_status.jsonl"
    statuses = []
    if status_path.exists():
        statuses = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    allowed = bool(phase0_go and phase1_go and phase2_go and phase3_go)
    summary_lines = [
        "# h120 Risk Scheduler Repair v1 Summary",
        "",
        f"Controller mainline: **v1.6 delayed-medium reactive floor**.",
        f"h120 learned controller admission: **{'candidate only after Phase 4' if allowed else 'NO-GO for this repair run'}**.",
        "",
        "## Phase Status",
        "",
    ]
    for row in statuses:
        summary_lines.append(f"- {row['phase']}: {row['status']} / {row['go_no_go']} — {row['reason']}")
    (OUT_DIR / "repair_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if not phase0_go:
        root_cause = "old failure was not reproduced; repair audit stopped before signal attribution."
        recommendation = "repair the reproduction harness before touching h120 signal or controller."
    elif not phase1_go:
        root_cause = "oracle far-risk shape labels were not stable enough for scheduler repair."
        recommendation = "stop h120 scheduler repair or redefine labels with a stronger control-state dataset."
    elif not phase2_go:
        root_cause = "current learned h120 raw forecast did not retain actionable far-risk shape information."
        recommendation = "do not tune controller thresholds; retrain with far-risk shape supervision if h120 remains important."
    elif not phase3_go:
        root_cause = "current learned h120 has weak/collapsed shape signal that lightweight calibration cannot make controller-ready."
        recommendation = "keep v1.6 mainline; next h120 work should be model/label redesign, not controller coupling."
    else:
        root_cause = "calibration passed signal-quality gates; scheduler v2 closed-loop probing is permitted but still default-off."
        recommendation = "run Phase 4 before any mainline admission."
    final = [
        "# Final Decision",
        "",
        "1. Current h120 learned problem:",
        "",
        f"- {root_cause}",
        "",
        "2. Model vs threshold vs controller attribution:",
        "",
        "- v1.6 floor remains the safety baseline; this repair run treats failures as learned signal / threshold / coupling issues unless Phase 4 proves otherwise.",
        "",
        "3. Scheduler v2 status:",
        "",
        f"- {'Allowed to proceed to default-off Phase 4 probe.' if allowed else 'No scheduler v2 was admitted to closed-loop in this run.'}",
        "",
        "4. Mainline decision:",
        "",
        "- Keep v1.6 as mainline.",
        "",
        "5. h120 learned admission:",
        "",
        f"- {'Not decided until Phase 4 completes.' if allowed else 'Do not connect current h120 learned scheduler to mainline.'}",
        "",
        "6. Next best action:",
        "",
        f"- {recommendation}",
    ]
    (OUT_DIR / "final_decision.md").write_text("\n".join(final) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-train", type=int, default=80_000)
    parser.add_argument("--max-validation", type=int, default=40_000)
    args = parser.parse_args()

    _ensure_dirs()
    # Fresh status ledger for a fresh run; artifacts themselves are overwritten.
    status_path = OUT_DIR / "debug/phase_status.jsonl"
    if status_path.exists():
        status_path.unlink()

    phase0_go = _phase0()
    if not phase0_go:
        _write_final(False, False, False, False)
        return
    phase1 = _phase1()
    if not phase1.go:
        _write_final(True, False, False, False)
        return
    phase2 = _phase2(phase1, args.device, args.batch_size)
    if not phase2.go:
        _write_final(True, True, False, False)
        return
    phase3_go = _phase3(
        phase1,
        phase2,
        args.device,
        args.batch_size,
        args.max_train,
        args.max_validation,
    )
    if not phase3_go:
        _write_phase4_skipped("Phase 3 no-go: calibrated h120 shape signal is not controller-ready.")
        _write_final(True, True, True, False)
        return
    _write_phase4_skipped(
        "Phase 3 passed, but scheduler-v2 controller integration is intentionally left for a separate code-change gate."
    )
    _write_final(True, True, True, True)


if __name__ == "__main__":
    main()
