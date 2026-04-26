# Codex Handoff

Last updated: 2026-04-26

This file is the first document another Codex session should read after cloning
or pulling this repository. It records the current project direction, what has
already been completed, where the important files are, and what should be done
next.

## Current Project Direction

The active project line is no longer the old FOWT controller replay and paper
artifact workflow. The current line is:

Build a short-term wind prediction module from real high-resolution wind speed
and wind direction data, then use the predicted wind-vector sequence and
wind-change event probabilities as preview inputs for later active-ballast
decision support.

The model should not directly output ballast commands. It should output:

- future wind-vector sequence;
- future wind speed and direction after inverse conversion;
- future wind-change event probabilities;
- risk signals that a later ballast decision layer can consume.

## Repository Orientation

Important paths:

- `README.md`: top-level current status and navigation.
- `CODEX_HANDOFF.md`: this handoff file.
- `data/README.md`: data layout.
- `data/processed/wind_ml_10min/README.md`: processed 10-minute wind ML dataset.
- `data/processed/wind_ml_10min/ballast_decision_dwd_helgoland/README.md`: current main LSTM dataset description.
- `scripts/data_preparation/prepare_wind_ml_dataset.py`: raw wind observations to 10-minute ML table.
- `scripts/data_preparation/prepare_lstm_sequences.py`: basic LSTM one-step sequence sets.
- `scripts/data_preparation/prepare_ballast_wind_decision_dataset.py`: current decision-oriented multi-step dataset.
- `scripts/modeling/evaluate_wind_decision_baselines.py`: persistence baseline.
- `scripts/modeling/train_ballast_lstm.py`: dual-head LSTM/GRU training.
- `scripts/modeling/train_lightgbm_wind_baseline.py`: LightGBM tabular strong baseline.
- `scripts/modeling/evaluate_regime_and_thresholds.py`: regime-level evaluation and threshold trade-off analysis.
- `风预测规划/面向主动压载提前调节的风预测技术路线.md`: method route.
- `风预测规划/风预测实现参考要点.md`: implementation rationale.
- `风预测规划/任务日志.md`: running task log and latest training/evaluation record.
- `4080_TRAINING_HANDOFF.md`: transfer and training instructions for the RTX 4080 laptop.
- `outputs/wind_prediction/gru_h128_b2048_e30_dir002/`: current best recurrent result.
- `outputs/wind_prediction/lightgbm_tabular_residual_e300/`: strong tabular baseline result.
- `outputs/wind_prediction/regime_threshold_analysis/`: regime and threshold analysis.

The handoff branch includes active code, planning docs, current processed
dataset metadata, and the core tensor dataset through Git LFS where available.
For the RTX 4080 laptop, prefer `git lfs pull` first; if LFS is slow or
incomplete, manually copy the files listed in `4080_TRAINING_HANDOFF.md`.

## Current Data State

Main processed dataset:

```text
data/processed/wind_ml_10min/ballast_decision_dwd_helgoland
```

Scope:

- source series: `DWD::02115_Helgoland::10min_wind`;
- raw station coverage: 1996-12-19 10:40 to 2025-12-31 23:50;
- input resolution: 10 minutes;
- input history: 12 steps, nominal 120 minutes;
- output horizon: 6 future steps, 60 minutes;
- input features: 31;
- regression target: future `u/v` wind-vector sequence;
- event target count: 5.

Current split counts:

| split | samples | time span |
| --- | ---: | --- |
| train | 986,913 | 1996-12-20 to 2016-07-30 |
| validation | 216,153 | 2016-07-30 to 2021-10-25 |
| test | 215,929 | 2021-10-25 to 2025-12-31 |

The chronological split is intentional. Windows do not cross split or series
boundaries, and scalers are fit on training rows only.

## Current Modeling State

Baseline:

- persistence baseline has been generated;
- it assumes future wind remains equal to the current wind state at the end of
  the history window;
- it is the required minimum comparison for any LSTM/GRU/TCN result.

Current best recurrent model:

```text
outputs/wind_prediction/gru_h128_b2048_e30_dir002
```

Model settings:

- architecture: dual-head GRU;
- input: `(batch, 12, 31)`;
- regression head: future 6-step `u/v` sequence;
- event head: 5 wind-change event logits;
- residual regression: enabled;
- hidden size: 128;
- GRU layers: 1;
- batch size: 2048;
- early-stopped after 23 epochs;
- event loss weight: 0.02;
- direction loss weight: 0.02;
- optimizer: AdamW;

Main test comparison:

| metric | persistence | residual GRU | LightGBM tabular |
| --- | ---: | ---: | ---: |
| all-step speed MAE | 0.6037 m/s | 0.5802 m/s | 0.5834 m/s |
| all-step vector MAE | 0.6674 m/s | 0.6481 m/s | 0.6476 m/s |
| all-step direction MAE | 7.7437 deg | 8.0956 deg | 7.9924 deg |
| t+60 speed MAE | 0.7587 m/s | 0.7286 m/s | 0.7340 m/s |
| t+60 vector MAE | 0.8570 m/s | 0.8243 m/s | 0.8255 m/s |
| t+60 direction MAE | 10.5484 deg | 10.7215 deg | 10.6687 deg |
| ballast attention recall | 0.0578 | 0.5678 | 0.5669 |
| ballast attention F1 | 0.1090 | 0.5162 | 0.4761 |
| ballast attention false alarm rate | 0.00024 | 0.06669 | 0.08588 |

Interpretation:

- ordinary speed/vector MAE improves over persistence;
- wind direction MAE is still not clearly better than persistence;
- LightGBM is a useful strong tabular baseline, but it does not beat the best
  residual GRU overall;
- event recognition improves strongly, especially for the aggregate
  `ballast_attention_event`;
- the current model is therefore better framed as a wind-change risk preview
  module, not as a finished high-accuracy wind direction forecaster.

## Hardware Judgment

The current GRU is still computationally light, but the next planned CNN-GRU or
TCN experiments will be heavier and are better moved to the RTX 4080 laptop.
Training is fast on the Mac mainly because the current model is small and the
sequence length is short, not because the dataset is useless.

Recommended split of work:

- M5 MacBook Air: data cleaning, document work, quick smoke tests.
- RTX 4080 and i9-13980HX laptop: formal training, repeated experiments,
  hyperparameter sweeps, GRU/TCN/CNN-LSTM comparisons.

For the current model scale, either machine can finish training. The 4080
laptop is still the better formal training machine because CUDA is more mature
and repeated experiments are much faster.

## What Is Already Done

Completed:

- collected and normalized multiple 10-minute wind data sources into a common
  observation schema;
- built circular wind-direction features with `sin/cos`;
- built meteorological `u/v` vector components;
- generated chronological train/validation/test splits;
- generated Markov-style wind-state transition probabilities;
- generated basic 60-minute and 120-minute LSTM sequence datasets;
- generated the current ballast-decision multi-step DWD Helgoland dataset;
- generated persistence baseline metrics;
- trained LSTM and GRU recurrent variants;
- selected `gru_h128_b2048_e30_dir002` as the current main recurrent model;
- trained LightGBM strong tabular baseline;
- generated regime-level and threshold trade-off analysis;
- documented the method route and current result.

Not done:

- no TCN/CNN-GRU comparison yet;
- no new short-term volatility feature set yet;
- no external-station generalization test yet;
- no probability interval or conformal uncertainty output yet;
- no final ballast decision integration yet.

## Next Work For Another Codex

Recommended next steps, in order:

1. Read `README.md`, this file, `4080_TRAINING_HANDOFF.md`, then `风预测规划/任务日志.md`.
2. Check whether the required data tensors exist locally:

```bash
ls data/processed/wind_ml_10min/ballast_decision_dwd_helgoland/X_train.npy
```

3. If tensors are missing or are tiny pointer files, run `git lfs pull`.
   If LFS download is unavailable, manually copy the dataset listed in
   `4080_TRAINING_HANDOFF.md`.
4. Keep `residual_gru` as the current main model and LightGBM as strong baseline.
5. Add short-term volatility features, then train CNN-GRU or TCN.
6. Keep persistence as the required minimum baseline in every comparison.
7. Report regime-level event recall, false alarm rate, and F1, not
   only ordinary speed MAE/RMSE.
8. Treat the DWD Helgoland result as single-station evidence. Do not claim
   generalization to all offshore sites until a held-out station or external
   source test is added.

## Reproduction Commands

Prepare the unified 10-minute ML table:

```bash
python scripts/data_preparation/prepare_wind_ml_dataset.py
```

Prepare the current ballast-decision dataset:

```bash
python scripts/data_preparation/prepare_ballast_wind_decision_dataset.py
```

Evaluate persistence baseline:

```bash
python scripts/modeling/evaluate_wind_decision_baselines.py
```

Train the current best GRU style:

```bash
python scripts/modeling/train_ballast_lstm.py \
  --epochs 30 \
  --batch-size 2048 \
  --hidden-size 128 \
  --num-layers 1 \
  --dropout 0.10 \
  --event-loss-weight 0.02 \
  --direction-loss-weight 0.02 \
  --patience 6 \
  --lr-patience 2 \
  --lr-factor 0.5 \
  --threads 8 \
  --residual-regression \
  --model-type gru \
  --output-dir outputs/wind_prediction/gru_h128_b2048_e30_dir002
```

On the RTX 4080 laptop, explicitly request CUDA if PyTorch CUDA is installed:

```bash
python scripts/modeling/train_ballast_lstm.py \
  --epochs 30 \
  --batch-size 2048 \
  --hidden-size 128 \
  --num-layers 1 \
  --dropout 0.10 \
  --event-loss-weight 0.02 \
  --direction-loss-weight 0.02 \
  --patience 6 \
  --lr-patience 2 \
  --lr-factor 0.5 \
  --threads 8 \
  --residual-regression \
  --model-type gru \
  --device cuda \
  --output-dir outputs/wind_prediction/gru_h128_b2048_e30_dir002
```

Train the LightGBM strong baseline:

```bash
python scripts/modeling/train_lightgbm_wind_baseline.py \
  --n-estimators 300 \
  --learning-rate 0.05 \
  --num-leaves 63 \
  --min-child-samples 100 \
  --threads 8 \
  --output-dir outputs/wind_prediction/lightgbm_tabular_residual_e300
```

Run regime and threshold analysis:

```bash
python scripts/modeling/evaluate_regime_and_thresholds.py \
  --split test \
  --device cuda \
  --batch-size 4096
```

## Cautions

- Do not random-split the sliding windows. Keep chronological splits.
- Do not fit scalers on full data. Fit scaling statistics on the train split
  only.
- Do not directly regress raw 0-360 degree wind direction as the main target.
  Use `sin/cos` or `u/v`.
- Do not present the million sliding windows as a million fully independent
  weather events. Adjacent windows overlap heavily.
- Do not write the current result as a general offshore-wind conclusion. It is
  a strong single-station short-term preview baseline.
- Large `.npy`, `.csv.gz`, raw NetCDF, zip, joblib, and model checkpoint files
  must use Git LFS. Do not add extra large derived datasets outside the current
  active scope without confirming that they are needed.

## Data Included In This Handoff

The current Git upload scope is intentionally narrower than the full local
workspace:

- included or manually transferable: current training tensor dataset
  `data/processed/wind_ml_10min/ballast_decision_dwd_helgoland/`;
- included or manually transferable: `data/processed/wind_ml_10min/supervised_learning_table_10min.csv.gz`;
- included or manually transferable: `data/processed/wind_ml_10min/canonical_observations_10min.csv.gz`;
- included: current code, planning docs, and compact output metrics;
- excluded: duplicate uncompressed
  `data/processed/wind_ml_10min/supervised_learning_table_10min.csv`;
- excluded: older derived basic sequence sets under
  `data/processed/wind_ml_10min/sequences*`;
- optional: PDF literature files under `风预测规划/文献/`;
- see `4080_TRAINING_HANDOFF.md` for the exact manual transfer list.

## Current Evidence Files

Best recurrent result files:

```text
outputs/wind_prediction/gru_h128_b2048_e30_dir002/lstm_config.json
outputs/wind_prediction/gru_h128_b2048_e30_dir002/lstm_run_summary.json
outputs/wind_prediction/gru_h128_b2048_e30_dir002/lstm_training_history.csv
outputs/wind_prediction/gru_h128_b2048_e30_dir002/lstm_regression_metrics.csv
outputs/wind_prediction/gru_h128_b2048_e30_dir002/lstm_event_metrics.csv
outputs/wind_prediction/model_comparison_with_lightgbm.csv
outputs/wind_prediction/regime_threshold_analysis/test_regression_by_regime.csv
outputs/wind_prediction/regime_threshold_analysis/test_attention_threshold_tradeoff.csv
```

Baseline files:

```text
data/processed/wind_ml_10min/ballast_decision_dwd_helgoland/baseline_persistence_regression_metrics.csv
data/processed/wind_ml_10min/ballast_decision_dwd_helgoland/baseline_persistence_event_metrics.csv
```

Main planning files:

```text
风预测规划/面向主动压载提前调节的风预测技术路线.md
风预测规划/风预测实现参考要点.md
风预测规划/LSTM训练执行记录.md
风预测规划/任务日志.md
4080_TRAINING_HANDOFF.md
```
