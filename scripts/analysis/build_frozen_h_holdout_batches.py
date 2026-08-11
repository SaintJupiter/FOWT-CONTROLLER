#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cases", type=Path, required=True)
    parser.add_argument("--development-cases", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    source = pd.read_csv(args.source_cases)
    development = pd.read_csv(args.development_cases)
    required = {"case_id", "timestamp", "label"}
    if not required.issubset(source.columns) or "timestamp" not in development.columns:
        raise ValueError("case files do not satisfy the expected schema")

    source = source.copy()
    source["_timestamp"] = pd.to_datetime(source["timestamp"])
    development_timestamps = set(pd.to_datetime(development["timestamp"]))
    overlap = source["_timestamp"].isin(development_timestamps)
    if int(overlap.sum()) != len(development_timestamps):
        raise ValueError("development cases are not an exact subset of source cases")

    holdout = source.loc[~overlap].copy()
    if len(holdout) != 150:
        raise ValueError(f"expected 150 holdout cases, found {len(holdout)}")
    if holdout["_timestamp"].duplicated().any():
        raise ValueError("holdout timestamps are not unique")

    holdout["selector_stratum"] = (
        holdout["label"]
        .str.extract(r"selector_stratum=([^|]+)")[0]
        .str.strip()
    )
    if holdout["selector_stratum"].isna().any():
        raise ValueError("holdout contains cases without selector_stratum")

    rng = np.random.default_rng(args.seed)
    queues: dict[str, list[int]] = {}
    for stratum, group in holdout.groupby("selector_stratum", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        queues[str(stratum)] = indices.tolist()

    batch_indices: list[list[int]] = [[], [], []]
    batch_strata: list[dict[str, int]] = [{}, {}, {}]
    while any(queues.values()):
        for stratum in sorted(queues):
            if not queues[stratum]:
                continue
            batch = min(
                range(3),
                key=lambda idx: (
                    len(batch_indices[idx]),
                    batch_strata[idx].get(stratum, 0),
                    idx,
                ),
            )
            batch_indices[batch].append(queues[stratum].pop())
            batch_strata[batch][stratum] = batch_strata[batch].get(stratum, 0) + 1

    if [len(indices) for indices in batch_indices] != [50, 50, 50]:
        raise ValueError("stratified assignment did not produce three 50-case batches")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_columns = ["case_id", "timestamp", "label"]
    master = holdout.sort_values("_timestamp")[output_columns]
    master_path = args.out_dir / "frozen_h_holdout150.csv"
    master.to_csv(master_path, index=False)

    batch_paths: list[Path] = []
    task_paths: list[Path] = []
    batch_counts: dict[str, dict[str, int]] = {}
    for batch_no, indices in enumerate(batch_indices, start=1):
        batch = holdout.loc[indices].sort_values("_timestamp")
        batch_path = args.out_dir / f"batch_{batch_no}_50.csv"
        batch[output_columns].to_csv(batch_path, index=False)
        batch_paths.append(batch_path)
        batch_counts[str(batch_no)] = {
            str(key): int(value)
            for key, value in batch["selector_stratum"].value_counts().items()
        }
        for task_no, start in enumerate(range(0, 50, 10), start=1):
            task_path = args.out_dir / f"batch_{batch_no}_task_{task_no}_10.csv"
            batch.iloc[start : start + 10][output_columns].to_csv(task_path, index=False)
            task_paths.append(task_path)

    manifest = {
        "schema_version": "frozen_h_holdout_batches.v1",
        "seed": int(args.seed),
        "source_cases": str(args.source_cases),
        "development_cases": str(args.development_cases),
        "source_case_count": int(len(source)),
        "development_case_count": int(overlap.sum()),
        "holdout_case_count": int(len(holdout)),
        "holdout_strata": {
            str(key): int(value)
            for key, value in holdout["selector_stratum"].value_counts().items()
        },
        "batch_strata": batch_counts,
        "files": {
            str(path.name): file_sha256(path)
            for path in [master_path, *batch_paths, *task_paths]
        },
    }
    manifest_path = args.out_dir / "frozen_h_holdout_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
