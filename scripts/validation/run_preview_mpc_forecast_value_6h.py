#!/usr/bin/env python3
"""CLI for the continuous preview-MPC comparison experiment."""

import argparse
import json
from pathlib import Path

from preview_mpc_continuous_experiment import (
    CYCLE_COUNT,
    FIXED_CASE_ORIGINS,
    WARMUP_BLOCK_COUNT,
    run_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--origin", action="append")
    parser.add_argument("--cycle-count", type=int, default=CYCLE_COUNT)
    parser.add_argument(
        "--warmup-block-count",
        type=int,
        default=WARMUP_BLOCK_COUNT,
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    origins = FIXED_CASE_ORIGINS if not args.origin else tuple(args.origin)
    result = run_comparison(
        origins=origins,
        device=args.device,
        cycle_count=args.cycle_count,
        warmup_block_count=args.warmup_block_count,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output_json": str(args.output_json),
                    "case_origins": result["case_origins"],
                    "pairwise_cross_case_summaries": result[
                        "pairwise_cross_case_summaries"
                    ],
                    "all_variants_complete_cross_case_summary": result[
                        "all_variants_complete_cross_case_summary"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(payload)


if __name__ == "__main__":
    main()
