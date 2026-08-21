import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data_preparation import prepare_ballast_wind_decision_dataset as dataset_builder


class ForecastTimeContractTests(unittest.TestCase):
    def _continuous_group(self, *, split: str, start: str, offset: float) -> pd.DataFrame:
        timestamps = pd.date_range(start=start, periods=4, freq="10min")
        rows = []
        for row_index, timestamp in enumerate(timestamps):
            row = {
                "series_id": "time-contract-test",
                "split": split,
                "timestamp": timestamp,
            }
            for column_index, column in enumerate(dataset_builder.FEATURE_COLUMNS):
                row[column] = float(offset + 10.0 * column_index + row_index)
            row["wind_u_ms"] = float(offset + row_index)
            row["wind_v_ms"] = float(offset + 100.0 + row_index)
            row["wind_speed_ms"] = float(offset + 200.0 + row_index)
            row["wind_dir_deg"] = float((offset + 10.0 * row_index) % 360.0)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_future_sequence_starts_one_dataset_period_after_history_end(self):
        frame = pd.concat(
            [
                self._continuous_group(split="train", start="2026-01-01", offset=0.0),
                self._continuous_group(split="validation", start="2026-02-01", offset=1000.0),
                self._continuous_group(split="test", start="2026-03-01", offset=2000.0),
            ],
            ignore_index=True,
        )
        history_steps = 2
        future_steps = 2
        counts, thresholds, _ = dataset_builder.collect_counts_and_thresholds(
            frame,
            history_steps=history_steps,
            future_steps=future_steps,
        )
        feature_stats = dataset_builder.standardizer(
            frame.loc[frame["split"] == "train", dataset_builder.FEATURE_COLUMNS]
        )
        target_stats = dataset_builder.standardizer(
            frame.loc[frame["split"] == "train", dataset_builder.TARGET_UV_COLUMNS]
        )

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            dataset_builder.build_dataset(
                frame,
                out_dir,
                history_steps=history_steps,
                future_steps=future_steps,
                feature_stats=feature_stats,
                target_stats=target_stats,
                thresholds=thresholds,
                counts=counts,
                selected_series_ids=["time-contract-test"],
            )

            targets = np.load(out_dir / "y_uv_raw_train.npy")
            np.testing.assert_allclose(
                targets[0],
                np.asarray([[2.0, 102.0], [3.0, 103.0]], dtype=np.float32),
            )
            with gzip.open(out_dir / "sample_index.csv.gz", "rt", encoding="utf-8") as handle:
                sample_index = pd.read_csv(handle, parse_dates=["history_end", "future_start", "future_end"])

        train_row = sample_index.loc[sample_index["split"] == "train"].iloc[0]
        self.assertEqual(train_row["future_start"] - train_row["history_end"], pd.Timedelta(minutes=10))
        self.assertEqual(train_row["future_end"] - train_row["history_end"], pd.Timedelta(minutes=20))


if __name__ == "__main__":
    unittest.main()
