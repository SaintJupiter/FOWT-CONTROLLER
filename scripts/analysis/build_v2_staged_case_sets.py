#!/usr/bin/env python3
"""Build result-blind 10/20/30-case validation sets from FINO1 test data.

The selector uses wind observations only. It neither reads controller outputs nor
runs a simulation. All cases are complete six-hour intervals, disjoint from one
another, and separated from the historical V1 case intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CASE_DURATION = pd.Timedelta(hours=6)
SAMPLE_PERIOD = pd.Timedelta(minutes=10)
EXPECTED_ROWS = 36
MIN_CASE_SEPARATION = pd.Timedelta(hours=48)
SEED = "fowt-v2-staged-cases-20260812"


@dataclass(frozen=True)
class Allocation:
    strengthening: int
    relief: int
    reversal: int
    oscillation: int
    low_disturbance: int


ALLOCATIONS = {
    "stage10": Allocation(3, 2, 1, 2, 2),
    "stage20": Allocation(5, 3, 2, 5, 5),
    "stage30": Allocation(8, 4, 4, 7, 7),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path(
            "data/processed/wind_ml_10min/fino1_platform_10min/"
            "canonical_observations_10min.csv.gz"
        ),
    )
    parser.add_argument(
        "--historical-cases",
        type=Path,
        default=Path(
            "paper_snapshot/2026-08-11/evidence/cases/original170_cases.csv"
        ),
    )
    parser.add_argument(
        "--smoke-cases",
        type=Path,
        default=Path("configs/control_chain_smoke_gate_cases_v1.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("configs/validation_cases_v2"),
    )
    return parser.parse_args()


def circular_angle_deg(first: np.ndarray, last: np.ndarray) -> float:
    denom = float(np.linalg.norm(first) * np.linalg.norm(last))
    if denom < 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(first, last) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def season(timestamp: pd.Timestamp) -> str:
    return {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }[timestamp.month]


def load_excluded_intervals(paths: list[Path]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                start = pd.Timestamp(row["timestamp"])
                intervals.append((start, start + CASE_DURATION))
    return intervals


def overlaps_any(
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end in intervals)


def candidate_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.loc[observations["split"].eq("test")].copy()
    observations["timestamp"] = pd.to_datetime(observations["timestamp"])
    observations = observations.set_index("timestamp").sort_index()

    starts = observations.index[
        (observations.index.minute == 0)
        & observations.index.hour.isin([0, 6, 12, 18])
    ]
    records: list[dict[str, object]] = []
    for start in starts:
        end = start + CASE_DURATION
        block = observations.loc[start : end - SAMPLE_PERIOD]
        if len(block) != EXPECTED_ROWS:
            continue
        if not np.all(np.diff(block.index.values) == np.timedelta64(10, "m")):
            continue
        if block[["wind_speed_ms", "wind_u_ms", "wind_v_ms"]].isna().any().any():
            continue

        speed = block["wind_speed_ms"].to_numpy(dtype=float)
        uv = block[["wind_u_ms", "wind_v_ms"]].to_numpy(dtype=float)
        smooth_uv = (
            pd.DataFrame(uv).rolling(3, center=True, min_periods=1).mean().to_numpy()
        )
        smooth_speed = (
            pd.Series(speed).rolling(3, center=True, min_periods=1).mean().to_numpy()
        )
        start_uv = uv[:6].mean(axis=0)
        end_uv = uv[-6:].mean(axis=0)
        speed_delta = float(speed[-6:].mean() - speed[:6].mean())
        speed_range = float(np.quantile(speed, 0.95) - np.quantile(speed, 0.05))
        vector_path = float(np.linalg.norm(np.diff(smooth_uv, axis=0), axis=1).sum())
        vector_net = float(np.linalg.norm(smooth_uv[-1] - smooth_uv[0]))
        path_excess = vector_path / (vector_net + 0.5)
        direction_shift = circular_angle_deg(start_uv, end_uv)
        increments = np.diff(smooth_speed)
        signs = np.sign(increments[np.abs(increments) >= 0.15])
        turns = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

        records.append(
            {
                "timestamp": start,
                "interval_end": end,
                "speed_delta_ms": speed_delta,
                "speed_range_ms": speed_range,
                "speed_std_ms": float(np.std(speed)),
                "vector_path_ms": vector_path,
                "vector_net_ms": vector_net,
                "path_excess": path_excess,
                "direction_shift_deg": direction_shift,
                "turn_count": turns,
                "season": season(start),
                "year": start.year,
            }
        )
    return pd.DataFrame.from_records(records)


def classify_candidates(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = {
        "trend_ms": 1.5,
        "low_trend_ms": 0.75,
        "strengthening_direction_deg": 45.0,
        "relief_direction_deg": 60.0,
        "reversal_direction_deg": 60.0,
        "reversal_vector_net_ms": 3.0,
        "oscillation_path_excess_q75": float(candidates["path_excess"].quantile(0.75)),
        "oscillation_speed_range_q60": float(candidates["speed_range_ms"].quantile(0.60)),
        "low_speed_range_q25": float(candidates["speed_range_ms"].quantile(0.25)),
        "low_vector_path_q25": float(candidates["vector_path_ms"].quantile(0.25)),
        "low_direction_shift_q25": float(candidates["direction_shift_deg"].quantile(0.25)),
    }

    frames: list[pd.DataFrame] = []
    definitions = {
        "strengthening": (
            candidates["speed_delta_ms"].ge(thresholds["trend_ms"])
            & candidates["direction_shift_deg"].lt(
                thresholds["strengthening_direction_deg"]
            ),
            candidates["speed_delta_ms"] + 0.15 * candidates["speed_range_ms"],
        ),
        "relief": (
            candidates["speed_delta_ms"].le(-thresholds["trend_ms"])
            & candidates["direction_shift_deg"].lt(thresholds["relief_direction_deg"]),
            -candidates["speed_delta_ms"] + 0.10 * candidates["speed_range_ms"],
        ),
        "reversal": (
            candidates["direction_shift_deg"].ge(
                thresholds["reversal_direction_deg"]
            )
            & candidates["vector_net_ms"].ge(thresholds["reversal_vector_net_ms"]),
            candidates["direction_shift_deg"] / 30.0
            + candidates["vector_net_ms"] / 3.0,
        ),
        "oscillation": (
            candidates["speed_delta_ms"].abs().lt(thresholds["trend_ms"])
            & candidates["path_excess"].ge(
                thresholds["oscillation_path_excess_q75"]
            )
            & candidates["speed_range_ms"].ge(
                thresholds["oscillation_speed_range_q60"]
            )
            & candidates["turn_count"].ge(3),
            candidates["path_excess"]
            + candidates["turn_count"] / 5.0
            + candidates["speed_range_ms"] / 5.0,
        ),
        "low_disturbance": (
            candidates["speed_delta_ms"].abs().lt(thresholds["low_trend_ms"])
            & candidates["speed_range_ms"].le(thresholds["low_speed_range_q25"])
            & candidates["vector_path_ms"].le(thresholds["low_vector_path_q25"])
            & candidates["direction_shift_deg"].le(
                thresholds["low_direction_shift_q25"]
            ),
            -(
                candidates["speed_range_ms"]
                + candidates["vector_path_ms"] / 5.0
                + candidates["direction_shift_deg"] / 15.0
            ),
        ),
    }
    for category, (mask, score) in definitions.items():
        frame = candidates.loc[mask].copy()
        frame["subtype"] = category
        frame["category"] = (
            "future_relief_or_reversal"
            if category in {"relief", "reversal"}
            else category
        )
        frame["category_score"] = score.loc[mask]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), thresholds


def stable_hash(*parts: object) -> str:
    text = "|".join(str(part) for part in (SEED, *parts))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_cases(
    pool: pd.DataFrame,
    stage: str,
    allocation: Allocation,
    excluded: list[tuple[pd.Timestamp, pd.Timestamp]],
    selected: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    subtype_counts = asdict(allocation)
    for subtype, count in subtype_counts.items():
        subset = pool.loc[pool["subtype"].eq(subtype)].copy()
        subset["intensity_band"] = pd.qcut(
            subset["category_score"].rank(method="first"),
            q=3,
            labels=["moderate", "marked", "strong"],
        )
        subset["selection_hash"] = subset.apply(
            lambda row: stable_hash(
                stage,
                subtype,
                row["season"],
                row["intensity_band"],
                row["timestamp"],
            ),
            axis=1,
        )
        subset = subset.sort_values(
            ["selection_hash", "timestamp"], kind="mergesort"
        )

        picked_for_subtype: list[pd.Series] = []
        used_seasons: dict[str, int] = {}
        used_bands: dict[str, int] = {}
        while len(picked_for_subtype) < count:
            eligible: list[tuple[tuple[int, int, str], pd.Series]] = []
            for _, row in subset.iterrows():
                start = row["timestamp"]
                end = row["interval_end"]
                if overlaps_any(start, end, excluded):
                    continue
                if any(
                    abs(start - other_start) < MIN_CASE_SEPARATION
                    for other_start, _ in selected
                ):
                    continue
                if any(
                    abs(start - other["timestamp"]) < MIN_CASE_SEPARATION
                    for other in picked_for_subtype
                ):
                    continue
                key = (
                    used_seasons.get(str(row["season"]), 0),
                    used_bands.get(str(row["intensity_band"]), 0),
                    str(row["selection_hash"]),
                )
                eligible.append((key, row))
            if not eligible:
                raise RuntimeError(f"Not enough independent {subtype} cases for {stage}")
            _, pick = min(eligible, key=lambda item: item[0])
            picked_for_subtype.append(pick)
            used_seasons[str(pick["season"])] = used_seasons.get(
                str(pick["season"]), 0
            ) + 1
            used_bands[str(pick["intensity_band"])] = used_bands.get(
                str(pick["intensity_band"]), 0
            ) + 1

        for pick in picked_for_subtype:
            selected.append((pick["timestamp"], pick["interval_end"]))
            chosen.append(pick)

    result = pd.DataFrame(chosen).sort_values("timestamp").reset_index(drop=True)
    result.insert(0, "stage", stage)
    result.insert(0, "sample_no", np.arange(1, len(result) + 1))
    abbreviations = {
        "strengthening": "enh",
        "relief": "rel",
        "reversal": "rev",
        "oscillation": "osc",
        "low_disturbance": "low",
    }
    result.insert(
        1,
        "case_id",
        [
            f"v2_{stage}_{abbreviations[subtype]}_{idx:02d}"
            for idx, subtype in zip(result["sample_no"], result["subtype"])
        ],
    )
    result["label"] = result.apply(
        lambda row: (
            f"v2_result_blind | stage={stage} | category={row['category']} | "
            f"subtype={row['subtype']} | intensity={row['intensity_band']} | "
            "split=test | selection_inputs=wind_only"
        ),
        axis=1,
    )
    return result


def interval_overlap_count(frame: pd.DataFrame) -> int:
    intervals = sorted(zip(frame["timestamp"], frame["interval_end"]))
    return sum(
        first_start < second_end and second_start < first_end
        for index, (first_start, first_end) in enumerate(intervals)
        for second_start, second_end in intervals[index + 1 :]
    )


def write_outputs(
    stages: dict[str, pd.DataFrame],
    thresholds: dict[str, float],
    out_dir: Path,
    excluded_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    candidate_counts: dict[str, int],
    observations_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "case_id",
        "timestamp",
        "label",
        "interval_end",
        "stage",
        "category",
        "subtype",
        "intensity_band",
        "season",
        "year",
        "speed_delta_ms",
        "speed_range_ms",
        "speed_std_ms",
        "vector_path_ms",
        "vector_net_ms",
        "path_excess",
        "direction_shift_deg",
        "turn_count",
        "selection_hash",
    ]
    all_frames: list[pd.DataFrame] = []
    for stage, frame in stages.items():
        output = frame[columns].copy()
        output.to_csv(out_dir / f"{stage}_cases.csv", index=False, float_format="%.6f")
        all_frames.append(output)
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(out_dir / "all_staged_cases.csv", index=False, float_format="%.6f")

    starts = sorted(pd.to_datetime(combined["timestamp"]).tolist())
    minimum_actual_separation_s = min(
        int((later - earlier).total_seconds())
        for earlier, later in zip(starts, starts[1:])
    )
    selected_intervals = list(
        zip(pd.to_datetime(combined["timestamp"]), pd.to_datetime(combined["interval_end"]))
    )
    historical_overlap_count = sum(
        overlaps_any(start, end, excluded_intervals)
        for start, end in selected_intervals
    )

    manifest = {
        "schema_version": 1,
        "selection_seed": SEED,
        "source_split": "test",
        "case_duration_s": int(CASE_DURATION.total_seconds()),
        "minimum_start_separation_s": int(MIN_CASE_SEPARATION.total_seconds()),
        "selection_inputs": [
            "wind_speed_ms",
            "wind_u_ms",
            "wind_v_ms",
            "timestamp",
            "split",
        ],
        "controller_outputs_used": False,
        "historical_intervals_excluded": len(excluded_intervals),
        "eligible_candidate_counts": candidate_counts,
        "thresholds": thresholds,
        "stage_counts": {stage: len(frame) for stage, frame in stages.items()},
        "category_counts": {
            stage: frame["category"].value_counts().sort_index().to_dict()
            for stage, frame in stages.items()
        },
        "subtype_counts": {
            stage: frame["subtype"].value_counts().sort_index().to_dict()
            for stage, frame in stages.items()
        },
        "within_and_cross_stage_overlap_count": interval_overlap_count(combined),
        "minimum_actual_start_separation_s": minimum_actual_separation_s,
        "historical_overlap_count": historical_overlap_count,
        "source_observations_sha256": hashlib.sha256(
            observations_path.read_bytes()
        ).hexdigest(),
        "combined_csv_sha256": hashlib.sha256(
            (out_dir / "all_staged_cases.csv").read_bytes()
        ).hexdigest(),
    }
    (out_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 科研版V2分层验证工况",
        "",
        "三组清单仅依据FINO1测试集中的风速和水平风矢量构造，未读取累计泵量、姿态响应或其他控制结果。每个工况持续6小时，三层工况互不重叠，且任意两个入选工况的起始时刻至少相隔48小时。旧版170组工况及现有烟雾测试时段未参与抽样。",
        "",
        "## 使用边界",
        "",
        "- `stage10`用于参数方向筛选，覆盖增强、回落或反转、持续振荡和低扰动。",
        "- `stage20`与`stage10`完全独立，用于检查改进是否依赖少量样本。其结果不回流调参。",
        "- `stage30`仅在参数冻结后运行，用于形成阶段性证据。",
        "- 三层均不是论文最终的大范围验证集合，不得根据节泵结果替换其中的工况。",
        "",
        "## 工况定义",
        "",
        "- 增强：工况末段平均风速较初段明显升高，且首尾平均来流方向未发生大幅转向。",
        "- 回落或反转：工况末段风速明显降低，或首尾平均风矢量发生较大方向变化。CSV中的`subtype`进一步区分`relief`和`reversal`。",
        "- 持续振荡：首尾风速净变化不大，但平滑风矢量的累计变化、风速范围和转折次数较高。",
        "- 低扰动：风速净变化、风速范围、风矢量累计变化和方向变化均处于测试集较低区间。",
        "",
        "具体阈值、固定随机种子、源数据哈希和清单哈希见`selection_manifest.json`。",
        "",
    ]
    for stage, frame in stages.items():
        lines.extend([f"## {stage}", ""])
        for category, group in frame.groupby("category", sort=True):
            case_ids = "、".join(group["case_id"].tolist())
            lines.append(f"- {category}：{case_ids}")
        lines.append("")
    lines.extend(
        [
            "## 重叠审计",
            "",
            f"- 三层内部及相互之间的时间重叠数：{interval_overlap_count(combined)}。",
            f"- 实际最小起始时间间隔：{minimum_actual_separation_s // 3600}小时。",
            "- 与旧版170组及既有烟雾测试时段的时间重叠数：0。",
            "",
            "当前数据足以形成三层清单。若后续增加新的类别或提高独立性要求，应从同一测试集的完整6小时记录中补充，并沿用相同的结果盲选、时间去重和参数冻结规则。",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    observations = pd.read_csv(
        args.observations,
        usecols=[
            "timestamp",
            "split",
            "wind_speed_ms",
            "wind_u_ms",
            "wind_v_ms",
        ],
    )
    candidates = candidate_metrics(observations)
    excluded = load_excluded_intervals([args.historical_cases, args.smoke_cases])
    candidates = candidates.loc[
        ~candidates.apply(
            lambda row: overlaps_any(row["timestamp"], row["interval_end"], excluded),
            axis=1,
        )
    ].copy()
    pool, thresholds = classify_candidates(candidates)

    stages: dict[str, pd.DataFrame] = {}
    selected: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for stage, allocation in ALLOCATIONS.items():
        stages[stage] = select_cases(
            pool,
            stage=stage,
            allocation=allocation,
            excluded=excluded,
            selected=selected,
        )
    candidate_counts = pool["subtype"].value_counts().sort_index().astype(int).to_dict()
    write_outputs(
        stages,
        thresholds,
        args.out_dir,
        excluded,
        candidate_counts,
        args.observations,
    )


if __name__ == "__main__":
    main()
