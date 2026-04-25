# FOWT Wind Prediction

This repository contains the wind-forecasting part of the floating offshore wind turbine active-ballast project.

The current scope is not direct ballast control. The model predicts future wind evolution from historical 10-minute wind speed and direction data, then provides wind-change event probabilities for downstream active-ballast preview decisions.

## Current Pipeline

1. Clean raw 10-minute wind observations and build a forecasting table.
2. Convert wind direction into circular/vector features.
3. Build a single-site DWD Helgoland sequence dataset:
   - input: past 120 minutes, 12 ten-minute steps;
   - output: next 60 minutes, 6 ten-minute steps;
   - regression target: future wind-vector `u/v` sequence;
   - classification target: future wind-change events.
4. Evaluate persistence baseline.
5. Train residual dual-head LSTM:
   - regression head predicts future `u/v` change relative to current wind vector;
   - classification head predicts wind-change event probabilities.

## Main Scripts

- `scripts/data_preparation/prepare_wind_ml_dataset.py`
- `scripts/data_preparation/prepare_ballast_wind_decision_dataset.py`
- `scripts/modeling/evaluate_wind_decision_baselines.py`
- `scripts/modeling/train_ballast_lstm.py`

## Current Result

Best current run:

`outputs/wind_prediction/lstm_ballast_decision_residual_w002`

Compared with persistence on the test split:

| Metric | Persistence | Residual LSTM |
| --- | ---: | ---: |
| All-step speed MAE | 0.6037 m/s | 0.5862 m/s |
| All-step vector MAE | 0.6674 m/s | 0.6496 m/s |
| t+60 speed MAE | 0.7587 m/s | 0.7375 m/s |
| t+60 vector MAE | 0.8570 m/s | 0.8275 m/s |
| Ballast-attention event recall | 0.0578 | 0.5277 |
| Ballast-attention event F1 | 0.1090 | 0.5099 |

Wind direction MAE is still weaker than persistence, so the next model iteration should add a circular direction loss or direction-change auxiliary task.

## Data Policy

Large raw and processed arrays are not committed:

- raw DWD/DTU/NDBC downloads;
- `.npy` training tensors;
- large `.csv.gz` tables;
- downloaded PDF papers.

The repository stores metadata, scripts, and compact result files. Rebuild the large arrays locally with the data-preparation scripts.

## Run

Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Prepare data after raw datasets are available locally:

```bash
python scripts/data_preparation/prepare_wind_ml_dataset.py
python scripts/data_preparation/prepare_ballast_wind_decision_dataset.py
python scripts/modeling/evaluate_wind_decision_baselines.py
```

Train LSTM:

```bash
python scripts/modeling/train_ballast_lstm.py \
  --epochs 8 \
  --batch-size 4096 \
  --hidden-size 64 \
  --event-loss-weight 0.02 \
  --residual-regression \
  --output-dir outputs/wind_prediction/lstm_ballast_decision_residual_w002
```

## Literature Notes

Technical notes and literature mapping are stored in `风预测规划/`.

