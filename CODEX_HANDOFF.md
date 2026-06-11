# Codex Handoff

Last updated: 2026-05-18

This is the first file a new Codex session should read in this repository.
It records the active project line, the current dataset and model state, what
has already been tried, and what should be done next.

## Current Project Direction

The active line is:

```text
historical 10-minute wind speed / wind direction
-> future 60-minute wind preview
-> segmented wind-change risk output
-> later active-ballast preview input
```

The wind model is not supposed to output ballast commands directly. It should
output:

- future wind-vector sequence;
- future wind speed / wind direction after inverse conversion;
- future wind-change event probabilities;
- segmented risk signals that a later ballast-control layer can consume.

## 2026-05-11 Control Integration Reset

The prediction-control line is being reset around three definitions before any
new controller patching:

1. Split the no-forecast baselines:
   - `persistence_60min`: repeat the latest 10-minute mean wind across the
     60-minute forecast horizon. This is a wind-forecast baseline.
   - `reactive_current`: the frozen `closed_baseline_v1` controller from
     `configs/closed_baseline_v1.json`. It is the engineering no-preview
     control baseline: PI/MIMO attitude feedback plus pump actuator constraints,
     with no forecast adapter, no planner horizon, and no current-wind-as-future
     assumption.

2. Do not treat the current `current_only` name as a true reactive controller.
   In the current code it repeats current wind into future blocks, so it is a
   persistence-style pseudo-preview. It must not be used as the engineering
   baseline against the learned preview controller.

3. Before more controller tuning, run diagnostics in this order:
   - Step 0: list which profile switches are actually active.
   - Step 1: forecast-layer horizon sweep on FINO1 test data, 10-120 minutes.
   - Step 2: five-layer control audit on small cases:
     forecast -> planner -> target -> execution -> safety.

4. Do not add new comfort/veto/credit gates while these diagnostics are open.
   Forecast value should enter through the same rollout cost equations for all
   forecast sources, not through learned-only privileges.

5. Keep failed or risky mechanisms out of the candidate mainline:
   `forecast credit`, source-specific comfort/evidence gates, `pi_release`, and
   1 Hz hold feedback. They may remain only as historical/diagnostic switches
   until a cleanup pass removes them safely.

## Read Order

When starting a new Codex session in this repo, read in this order:

1. `README.md`
2. `CODEX_HANDOFF.md`
3. `风预测规划/任务日志.md`
4. `4080_TRAINING_HANDOFF.md`

If the task is specifically about control integration, also read:

5. `archive/README.md`
6. `archive/legacy_fowt_control/controllers_extras.py`
7. `archive/legacy_fowt_control/run_validation.py`

## Execution Discipline For Codex

When running tasks in this repository, follow this checklist before improvising:

1. Use the project environment first:
   `./.venv312/bin/python` is the default Python for this repo. Do not casually
   use system `python3` or the old `./.venv/bin/python` for dependency checks,
   plotting, analysis, simulations, or model training.

2. Check dependencies inside `.venv312`:
   if a library appears missing, verify it with `./.venv312/bin/python` before
   changing approach. The old `.venv` may have some packages installed but is no
   longer the canonical environment.

3. Install missing project dependencies when needed:
   use `./.venv312/bin/python -m pip install <package>`. If install fails because
   of network or permission restrictions, ask the user for approval instead of
   silently switching to a weaker workaround.

4. Use writable cache paths for Python and plotting:

   ```bash
   MPLCONFIGDIR=/private/tmp/matplotlib \
   PYTHONPYCACHEPREFIX=/private/tmp/fowt_pycache \
   ./.venv312/bin/python ...
   ```

   This avoids false failures from non-writable macOS cache directories.

5. Do not downgrade outputs because of a bad environment guess:
   if the user asks for plots, first use the correct `.venv312` plotting stack.
   Only use SVG/manual fallbacks when the project environment truly cannot
   support the requested output or the user agrees.

6. GPU training discipline:
   model training must use GPU/MPS, not CPU. Before a training run, verify
   `.venv312` with a small PyTorch MPS sanity check. If MPS is unavailable or
   PyTorch import hangs, fix the environment first rather than silently training
   on CPU.

7. Pause for user judgment at major decision points:
   choosing or abandoning a main route, changing validation design, scaling up a
   run, or interpreting validation results as a direction change. Explain in
   plain language: what happened, why it matters, and what the proposed next
   step would change.

8. Continue autonomously for routine work:
   reading files, small sanity checks, compiling, fixing obvious environment
   setup, and generating explicitly requested artifacts can proceed without
   interrupting the user.

8. Control-result figures must expose action and ballast state:
   when plotting FOWT control comparisons, include bottom action/fallback bands
   and a cumulative pump-work / cumulative saved-ballast curve relative to the
   chosen baseline. Three-tank ballast mass curves are useful for diagnostics,
   but the user-facing advantage plot should show how much ballast pumping has
   been saved up to each time point. Do not rely on pitch/roll/pump-rate curves
   alone, because they hide whether savings come from prediction, coupled tank
   allocation, or safety fallback.

## Current Main Datasets

Current trained baseline dataset:

```text
data/processed/wind_ml_10min/ballast_decision_dwd_helgoland_segmented_v1
```

Scope:

- source series: `DWD::02115_Helgoland::10min_wind`
- input resolution: 10 minutes
- history window: 12 steps, 120 minutes
- future horizon: 6 steps, 60 minutes
- input features: 31
- regression target: future `u/v` sequence
- event target count: 8

Event targets:

- `speed_ramp_ge_3ms`
- `direction_shift_ge_45deg`
- `vector_change_ge_train_p90`
- `future_speed_ge_train_p95`
- `ballast_attention_event`
- `attention_event_0_20m`
- `attention_event_20_40m`
- `attention_event_40_60m`

Split counts:

| split | samples |
| --- | ---: |
| train | 986,913 |
| validation | 216,153 |
| test | 215,929 |

Important rules:

- chronological split only;
- train-only scaling;
- windows do not cross split boundaries;
- this is still single-station evidence, not offshore generalization evidence.

New offshore-height retraining base now available:

```text
data/processed/wind_ml_10min/fino1_platform_10min
data/processed/wind_ml_10min/ballast_decision_fino1_segmented_v1
```

FINO1 fused-series scope:

- source series: `BSH_FINO1::FINO1_Platform::fused_102m_speed_91m_dir_v1`
- wind speed: prefer `102m` mast-corrected cup speed, fallback to raw `102m`
  cup speed when corrected values fail QC or are missing
- wind direction: prefer `91m` vane, fallback to `82m` ultrasonic direction
- accepted QC flags: `2/3/4`
- retained auxiliary columns for later experiments:
  `air_temp_c`, `air_pressure_hpa`, `rel_humidity_pct`

FINO1 split counts:

| split | samples |
| --- | ---: |
| train | 657,747 |
| validation | 138,657 |
| test | 142,915 |

## Current Best Results

The current main comparison is on the `test` split.

### Key model folders

- segmented GRU:
  `outputs/wind_prediction/gru_segmented_head_v1`
- segmented LightGBM:
  `outputs/wind_prediction/lightgbm_segmented_residual_e300`
- residual LSTM comparison:
  `outputs/wind_prediction/lstm2_h128_b2048_e25_dir002`
- current result board and figures:
  `outputs/wind_prediction/ppt_figures`

### Current test summary

| model | speed MAE | direction MAE | ballast attention F1 | 0-20 F1 | 20-40 F1 | 40-60 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence baseline | 0.6037 | 7.7437 | 0.1090 | n/a | n/a | n/a |
| LSTM | 0.5822 | 8.0247 | 0.5131 | n/a | n/a | n/a |
| segmented GRU | 0.5816 | 8.1041 | 0.5209 | 0.4584 | 0.4634 | 0.4615 |
| segmented LightGBM | 0.5837 | 7.9858 | 0.4756 | 0.4054 | 0.4013 | 0.4156 |

Interpretation:

- ordinary speed MAE differences are small;
- direction MAE differences are also small, and persistence remains strong;
- the strongest argument for the current GRU is not speed MAE, but segmented
  risk output and ballast-attention event F1;
- even then, GRU vs LSTM is only a small margin, so model-side superiority is
  not yet a strong paper claim by itself.

## What Has Been Tried Already

Completed:

- persistence baseline;
- residual LSTM;
- segmented GRU;
- segmented LightGBM;
- CNN-GRU comparison;
- TCN comparison;
- enhanced short-term handcrafted features;
- focal-loss GRU event-loss attempt;
- result board and PPT-style comparison figure generation;
- control-chain preview interface insertion in legacy controller code.

Current conclusions from those trials:

- CNN-GRU and TCN did not beat `gru_segmented_head_v1`;
- enhanced handcrafted features did not produce a meaningful gain;
- focal loss slightly improved ordinary regression error but did not improve
  the target event F1 enough, so it is not the main version;
- LightGBM is a valid strong baseline, but segmented GRU remains better on the
  control-oriented event target.

## Control Integration State

Legacy control integration code is under:

```text
archive/legacy_fowt_control/
```

Relevant files:

- `archive/legacy_fowt_control/controllers_extras.py`
- `archive/legacy_fowt_control/run_validation.py`
- `archive/legacy_fowt_control/core_model.py`
- `archive/legacy_fowt_control/wind_env.py`

Already done:

- `ClosedLoopPolicy` now supports `preview_trim_provider`;
- `TrimGovernor.update()` now accepts `preview_trim_bias`;
- preview trim can be injected into the control chain without rewriting the
  whole controller;
- default mode keeps current-wind trim pressure-guarded while allowing preview
  trim to avoid being completely suppressed.

This means the next control experiment should not redesign the controller from
scratch. It should plug wind preview output into the existing preview interface.

## Current Planning Judgment

The method-route documents under `风预测规划/` are still directionally correct:

- the module is still positioned as a wind-preview front end, not a direct
  ballast-command model;
- vectorized wind direction, chronological splits, event labels, and multi-step
  output are still the right structure.

What changed since those documents were first written:

- the current active dataset is now `segmented_v1`, not the earlier 5-event
  dataset;
- the main comparison should now be framed around segmented risk and
  control-oriented event output;
- the next priority is no longer “try more structures until one wins clearly”.
  It is:
  1. get better offshore-height data such as FINO1;
  2. or integrate the current preview model into ballast-control evaluation.

## FINO1 Status

FINO1 download and first-pass processing are done.

Relevant artifacts:

- raw files:
  `data/raw/wind_10min/FINO1_platform_BSH/`
- processed observation table:
  `data/processed/wind_ml_10min/fino1_platform_10min/`
- model-ready segmented dataset:
  `data/processed/wind_ml_10min/ballast_decision_fino1_segmented_v1/`

Important note:

- FINO1 is no longer a pending data-access task.
- The next FINO1 action is retraining and comparing against the existing DWD
  result under the same model/metric setup.

## Recommended Next Steps

Choose one of these two lines and stay focused:

### Line A: better data

1. Reuse `ballast_decision_fino1_segmented_v1`.
2. Retrain GRU / LightGBM on FINO1 with the same settings first.
3. Compare DWD vs FINO1 under the same model and metric definition.
4. Only after that, decide whether extra exogenous FINO1 variables are worth
   introducing.

### Line B: control validation

1. Keep `gru_segmented_head_v1` as the preview model.
2. Convert predicted future wind into preview trim bias or preview risk signal.
3. Run closed-loop comparisons with and without preview.
4. Report pitch/roll RMS, threshold exceedance, pump throughput, pump starts,
   and whether preview reduces late reactions.

## 2026-05-17 Aggregate-Level Controller Tuning Session

This session attacked the **aggregate** (not single-case) optimization of
prediction-primary under the production overlay profile.

### Sessions outputs in `outputs/wind_prediction/manifest_pareto_eval/`

Read `findings_review.md` end-to-end — it has 3 in-line addenda
documenting the full trail. The bottom line:

**No production change committed.** Default
`rawenv_holdpause_barrier_reliefcap_adaptive_v1` (gain=0.40) remains the
recommended controller. All candidate mechanisms
(`active_intent_reproposal`, `objective_mode_v3`, `hold_relief_debt`)
either regress on guard10 or are inert under overlays.

### What was tried and rejected

1. `objective_mode_v3`: 5case b_high win + fr09 loss; mode-switch
   trajectory identical on both cases → predicate can't separate plateau
   from drift load.
2. `active_intent_reproposal`: 5case win on fr09; **guard10 strict
   Pareto regression on b_decay_strong** (+683 m³, +0.74° pitch, +8.15pp
   fb) — driven by `target_err` being huge (max 71t) on saturated load.
3. `hold_relief_debt`: bit-identical to overlays_only on all guard10
   cases — inert when combined with the overlay stack.
4. Cost-weight sweep (WATT/WTERM/GAIN/CLIP): only **GAIN** moves the
   planner's discrete sequence choice. WATT × 1.5 and CLIP 6→9 produce
   bit-identical output to baseline.
5. `posture_state_gain = 0.50` on guard10 oracle: **−18% pump,
   max_p95 unchanged, fb unchanged** (clean aggregate win).
6. Same `gain = 0.50` on guard10 **learned** forecast: **net loss**
   (pump +65 m³, fb +2.78 pp, b_signflip fb 3.18% → 6.08%).
   → Oracle-side tuning does NOT generalize to the deployment forecast
   protocol. Production default stays at gain=0.40.

### Code change committed (one file)

`scripts/analysis/run_prediction_primary_casebook.py`:
- added `import os`
- profile block (`rawenv_holdpause_barrier_reliefcap_adaptive_v1`) reads
  4 weights from env vars with the original constants as defaults:
  `FOWT_TUNE_GAIN` (0.40), `FOWT_TUNE_CLIP` (6.0), `FOWT_TUNE_WATT`
  (1.0), `FOWT_TUNE_WTERM` (1.0).

This is non-invasive: with no env var set, behavior is bit-identical to
the prior version. It makes future weight sweeps trivial.

### Methodological lesson (recorded in user memory)

Any prediction-primary controller change MUST be validated on
`--forecast-source learned` on guard10 before being committed as a
production default. Oracle-only wins repeatedly fail on learned
because gain amplifies forecast noise that the oracle protocol doesn't
expose. The oracle→learned gap is ~18pp of fallback induced by LSTM
forecast error alone — this is **larger than every controller-side
improvement found across the session combined**.

### What this implies for the next prediction-primary direction

The remaining integral improvement on production deployment must come
from one of:
- forecast quality (LSTM training, ensembling, uncertainty-aware
  features), or
- forecast-uncertainty-aware control (when LSTM confidence is low,
  default to reactive behavior, NOT predictive park).

Cost-weight tuning has been exhausted on the existing knobs at the
current operating point. No further sweeps on these dimensions are
worth pursuing without a structural change.

### Useful commands for re-validation

Reproduce the baseline (must match commit `7d11f9f` numbers exactly):
```bash
MPLCONFIGDIR=/private/tmp/matplotlib PYTHONPYCACHEPREFIX=/private/tmp/fowt_pycache \
./.venv312/bin/python scripts/analysis/run_prediction_primary_casebook.py \
  --cases-csv outputs/wind_prediction/broad_optimization_work/guard10_broad_cases.csv \
  --primary-control-profile rawenv_holdpause_barrier_reliefcap_adaptive_v1 \
  --primary-only --duration-s 7200 --skip-figures --forecast-source oracle \
  --out-dir outputs/wind_prediction/regression_check_baseline
```
Expected: sum_pump ≈ 3058, max p95 ≈ 6.49°, sum fb ≈ 12.64 pp.

Try the oracle-only gain=0.50 reproduction:
```bash
FOWT_TUNE_GAIN=0.50 [same command, different --out-dir]
```
Expected: sum_pump ≈ 2506, max p95 ≈ 6.50°, sum fb ≈ 12.61 pp.

Same on learned (shows the regression):
```bash
FOWT_TUNE_GAIN=0.50 [same command with --forecast-source learned]
```
Expected: sum_pump ≈ 3826, max p95 ≈ 6.97°, sum fb ≈ 33.29 pp (vs
baseline 30.51 — a +2.78 pp regression).

## 2026-05-17 Phase 6: Synth-Relief LSTM Training (breakthrough)

**Root-cause confirmed**: standard MSE regression on residual-uv targets
converges to ~persistence at long horizon. LSTM's b1/b0 ratios stay in
[0.95, 1.30] vs oracle's [0.07, 11.7] on guard10. Per-horizon MAE in
training is ~1 m/s at +60min, but median true 60-min wind change is
1.20 m/s — meaning LSTM is barely reducing persistence baseline error.

**Fix delivered**: new `EventEmphasisMSELoss` in
[`scripts/modeling/train_ballast_lstm.py`](scripts/modeling/train_ballast_lstm.py)
with two modes:

1. **Event-column emphasis**: upweight samples where `y_event[:, col]` is positive.
   Existing labels (col 0-7) are intensification-biased; helps catch sharp
   events but does NOT help with relief (de-escalate) signal.
2. **Synth-relief emphasis** (the real win): compute relief mask on-the-fly
   from y_uv: `relief = future_min_speed < (1-τ)·current_speed AND current_speed > 4 m/s`.
   Captures the 3.7% of samples with strong relief that existing labels miss
   (<15% overlap with any existing event flag).

**CLI**:
- `--regression-event-emphasis FLOAT` (per-sample weight = 1 + emphasis × mask)
- `--regression-event-column INT` (which y_event column for intensify mode)
- `--regression-synth-relief-threshold FLOAT` (if >0, use synth-relief mask)
- `--regression-synth-relief-current-min-ms FLOAT` (default 4.0)

All default-off; with no flags, behavior is bit-identical to prior MSE training.

**Best production tune found**:
```
--regression-event-emphasis 15.0
--regression-synth-relief-threshold 0.30
--regression-synth-relief-current-min-ms 4.0
```
Output: `outputs/wind_prediction/lstm_synth_relief_t030_e15_v1/lstm_best.pt`

**Result on guard10 (oracle vs learned forecast)**:

| controller | sum_pump | max_p95 | sum_fb | comment |
|---|---:|---:|---:|---|
| reactive (closed_baseline) | 9634 | 5.83 | 0.00 | safety upper bound |
| PP + oracle forecast | 3058 | 6.49 | 12.64 | best-case forecast |
| PP + baseline LSTM | 3761 | 7.14 | 30.51 | prior production |
| **PP + relief-e15 LSTM** | **3491** | **7.09** | **24.65** | strict Pareto over baseline |

Per-case improvements vs baseline LSTM:
- `fr_relief_09`: pump 863→757 (−12%), pitch 6.68→6.47°, fb 14.4%→12.9%
- `sf_holdout_02`: pump 480→354 (−26%), **fb 4.3%→0% (eliminated)**
- `b_decay_strong`: pump 957→920 (−4%)
- 7 other cases: zero change (no regression)

**Net delivery on the user's goal** ("实打实节约水泵 + 不松姿态"):
- pump savings: −270 m³ (−7.2%) **real**, comes from genuinely improved forecast
- max pitch: −0.05° (slightly safer, not less safe)
- fb: −5.86 pp (−19%, safer)
- Zero case regression

**Paper framing now supported by data**:
> Standard MSE-trained LSTM forecasts for FOWT ballast control collapse to ~persistence at the
> control-relevant horizon (20-60min), losing forecast signal that downstream MPC could use.
> We introduce a synth-relief emphasis loss that synthesizes a future-decrease mask from y_uv
> at training time and upweights those samples by 15×. On guard10, this delivers a strict
> Pareto improvement under learned forecast: pump −7.2%, max pitch −0.05°, fallback −19%,
> with zero regression on any of 10 test cases.

**Phase 1a-4 controller scaffolding kept**: bit-identical to prior production
when no env vars are set; provides safety_floor diagnostic infrastructure for
future iterations. See findings_review.md addenda 1-4 for full diagnostic trail.

**What this work does NOT yet deliver**:
- Oracle pump 3058 is still 433 m³ below e15's 3491; we closed ~38% of the
  oracle-vs-baseline gap. Further gains likely require combined intensify+relief
  emphasis (current loss supports one mask at a time).
- max p95 7.09° is still 1.26° above reactive's 5.83°; PP's "max-event" weakness
  vs reactive is reduced but not eliminated.

## 2026-05-18 Phase 7: sustained_active_recompute — candidate for stale-active execution chain (NOT mainline default)

**Problem context**: Even after `PP + relief_e15 LSTM` reached strict Pareto improvement
in Phase 6 (pump 3491, fb 24.65 on guard10 learned), several cases still spent long
continuous periods at 3-6° pitch. The user's narrow `active_effectiveness_refresh`
test (with --min-target-age-s 1800 --min-pressure-norm 0.6 --required-worsening-deg 0.5)
got fb down to 21.40 but **only fired on fr_relief_01 and fr_relief_09** and did
NOT break the longest sustained high-attitude episode. Codex diagnosed the
underlying mode as: planner picks active_small/medium, target_lifecycle reuses
old target, pump goes idle, posture stays high → "active execution chain idles".

**New mechanism**: `sustained_active_recompute`. Treats sustained-active +
stalled-pump + posture-above-band as a structural event_reset-equivalent signal
(same lifecycle priority as a discrete action change). Unconditional `_update_primary_target`
when ALL of the following hold:

1. action is active-like
2. consecutive active buckets ≥ `min_active_buckets` (default 4)
3. NOT fallback-dominated
4. |pitch| > `pitch_deg` (default 4.5°) OR |roll| > `roll_deg` (default 4.5°)
5. target age ≥ `min_target_age_s` (default 1800s)
6. target mean abs error ≤ `target_err_kg` (default 1500 kg)
7. total pump rate ≤ `pump_rate_m3_min` (default 0.5 m³/min)

Distinct from narrow `active_effectiveness_refresh`: this mechanism does NOT
require pressure-norm, worsening, proposal-delta, or episode-budget gates. It
fires whenever the active-execution chain is stalled with posture still high.

**Files touched**:
- `src/wind_prediction/ballast_planner.py` — unchanged (lifecycle elif extended in provider)
- `src/wind_prediction/ballast_planner_provider.py` — `_sustained_active_recompute_needed`
  method (lines ~2892-2940), telemetry fields (init + 2 live blocks), wired into
  lifecycle elif at line ~3651 alongside `event_reset or not _primary_target_initialized`
- `scripts/analysis/run_prediction_primary_casebook.py` — 7 new CLI flags + provider
  args + casebook_summary CSV schema (7 fields) + preview_sustained_active_recompute_ratio
  aggregation

**Defaults default-off; bit-identical with no flags** (verified twice on guard10
learned with relief_e15 LSTM).

**Result on guard10 (learned forecast, relief_e15 LSTM)**:

| controller | sum_pump | max_p95 | sum_fb | t_over_3 | t_over_5 |
|---|---:|---:|---:|---:|---:|
| PP+relief_e15 (no refresh) | 3491 | 7.085 | 24.65 | 28514 | 13895 |
| + narrow active_effectiveness_refresh | 3719 | 7.085 | 21.40 | 25576 | 13328 |
| **+ sustained_active_recompute (defaults)** | **3925** | **7.085** | **21.40** | **25500** | **12753** |

Per-case (sar_default vs no-refresh baseline):

| case | sar fires | Δpump | Δfb | Δt_over_3 | Δt_over_5 | Δmax_cont_3 |
|---|---:|---:|---:|---:|---:|---:|
| fr_relief_01 | 1 | +241 | 0 | **−2938s** | −542 | **−470s (−28%)** |
| fr_relief_09 | 1 | **−14** | **−3.2pp** | 0 | −25 | 0 |
| sf_holdout_02 | 1 | +57 | 0 | 0 | −14 | 0 |
| b_signflip_fallback | 1 | +49 | 0 | −76 | 0 | −4 |
| b_high_pressure_event | 2 | +101 | 0 | 0 | **−561s** | 0 |
| 5 others | 0 | 0 | 0 | 0 | 0 | 0 |

Total 6 true SAR firings out of 120 buckets after telemetry cleanup (an earlier
draft counted 9 because normal event-reset refreshes inherited stale SAR telemetry).
Still sparse selective intervention.

**Key qualitative wins** over narrow active_effectiveness_refresh:
- Fires on 5 cases vs narrow's 2 — broader structural coverage
- **2× larger t_over_5 reduction** (−1142s vs −567s) — directly addresses the
  longest-sustained-high-attitude problem the narrow refresh failed on
- **fr_relief_09 strict Pareto on critical case** (−14 m³ pump AND −3.2pp fb)
- max_continuous_over_3 on fr_relief_01: 1674s → 1204s (−28%) — first mechanism
  to meaningfully reduce this metric

**Trade-off vs narrow refresh**: +206 m³ pump for ~2× safety improvement on
t_over_5. Trade is structural, not case-specific (gates do not reference case
identity, only physical signals).

**Status: candidate, NOT a main-line default.**

This mechanism is the cleanest default-off candidate currently on the table.
It addresses a real target-lifecycle stall (stale active target + idle pump +
sustained high posture) and produces measurable, attribution-clear improvement
on guard10 over both the no-refresh baseline and the narrow
`active_effectiveness_refresh`. But the data does NOT yet justify promoting it
to a production default:

- max_p95 is unchanged (7.085°). The single worst pitch peak (b_decay_strong)
  is not reduced.
- Aggregate pump cost rises by +434 m³ vs the no-refresh baseline. The
  improvement is "trade pump for safety", not strict Pareto.
- guard10 is 10 cases of one site. Generalization to a broader/holdout set is
  unverified.

Appropriate next step: **carry sar_default forward as a candidate into the next
holdout-set evaluation alongside the no-refresh baseline and the narrow
refresh**. Decide on production default only after holdout numbers are in. The
old narrow `active_effectiveness_refresh` should stay in the code as a parallel
default-off candidate (in this guard10 run its fired cases were covered by sar,
but the two gates are not equivalent and should be compared independently).

**Reproducible command**:
```bash
./.venv312/bin/python scripts/analysis/run_prediction_primary_casebook.py \
  --cases-csv outputs/wind_prediction/broad_optimization_work/guard10_broad_cases.csv \
  --primary-control-profile rawenv_holdpause_barrier_reliefcap_adaptive_v1 \
  --primary-only --duration-s 7200 --skip-figures \
  --forecast-source learned \
  --model-dir outputs/wind_prediction/lstm_synth_relief_t030_e15_v1 \
  --sustained-active-recompute \
  --out-dir outputs/wind_prediction/pp_relief_e15_sar_default_learned_guard10
```

Expected: sum_pump≈3925, max_p95≈7.085, sum_fb≈21.40, t_over_3≈25500, t_over_5≈12753.

**What this does NOT yet deliver**:
- max_p95 unchanged at 7.085° — single worst pitch peak (b_decay_strong) not
  reduced. sar does not fire on b_decay_strong because its target_err mostly
  stays > 1500 kg (pump never fully reaches target before next event).
- Combined pump cost is higher than the no-refresh baseline (+434 m³). Net
  Pareto trade is: extra pump for substantially better safety. Whether this
  trade fits depends on the user's pump-vs-safety weighting; the alternative
  is to keep the no-refresh baseline as-is (3491 m³, 24.65 fb).

## 2026-05-18 Phase 8: combined intensify+relief emphasis LSTM — NEGATIVE RESULT

**Hypothesis**: relief_e15 LSTM handles relief-class cases (fr_relief_09) but
does not help intensification-class cases (b_decay_strong, b_*-family). Training
a single LSTM where the emphasis mask is `relief_mask OR ballast_attention_event`
might handle BOTH failure modes in one model.

**Implementation**: extended `EventEmphasisMSELoss` with a `combine_masks`
opt-in (CLI: `--regression-emphasis-combine-modes`). When both
`--regression-synth-relief-threshold` AND `--regression-event-column` are
configured, OR the masks instead of mutually-exclusive selection.
**Default behavior unchanged** (relief-only when threshold set, column-only
otherwise) so prior runs reproduce bit-identically.

**Result on guard10 learned**:

| LSTM | pump | max_p95 | sum_fb | t_over_3 | t_over_5 |
|---|---:|---:|---:|---:|---:|
| relief_e15 (current best) | 3491 | 7.085 | 24.65 | 28514 | 13895 |
| combine_e15 (relief OR col4, emph=15) | 4010 (+519) | 7.144 | 28.98 (+4.3) | 26699 | 13374 |
| combine_e20 (emph=20) | 3764 (+273) | 7.143 | 30.55 (+5.9) | 29756 | 14984 |

**Both combined variants are clearly worse** than relief_e15 on pump AND fb AND
max_p95. The "combined" emphasis dilutes the relief-specific gradient signal
(20% mask coverage instead of 3.7%) and pulls the LSTM toward an averaged
prediction that helps neither class.

**Lesson**: the bimodal failure modes (relief case vs intensification case)
cannot be jointly handled by a single emphasis-MSE loss. Possible future
directions (NOT pursued):
- Two separate LSTMs + ensemble at inference (compute cost 2×; complex)
- Multi-head architecture (one head for relief, one for intensification)
- Different loss class entirely (e.g., focal regression with horizon-aware
  bins)

These are all big-architecture moves, not loss-tuning. None are recommended
without clear hypothesis.

**Production status**: `relief_e15` remains the best learned LSTM. Combined-mode
flag stays in code as opt-in (no impact on default training) for future
experimentation, but no recommendation to use it.

**Files**: extension in `scripts/modeling/train_ballast_lstm.py`
(`EventEmphasisMSELoss.combine_masks`, CLI flag `--regression-emphasis-combine-modes`).

## 2026-05-19 Phase 9: broader 20-case validation REVERSES the relief_e15 verdict

**Hypothesis under test**: that Phase 6's `relief_e15` is the best learned LSTM
for production. Until now this was based only on guard10 (10 cases skewed
toward relief/signflip events).

**Run**: `outputs/wind_prediction/f60_relief_e15_holdout_validation_v1/` —
20 broader cases (pilot30 minus guard10 overlap), same controller, only
`--forecast-source` and `--model-dir` differ across runs.

**Aggregate result (20-case holdout)**:

| LSTM | sum_pump | max_p95 | sum_fb |
|---|---:|---:|---:|
| baseline LSTM (`lstm_segmented_fino1_meteo_aux_v1`) | **5731** | 6.274 | **0.105** |
| **relief_e15** (`lstm_synth_relief_t030_e15_v1`) | **6665 (+934)** | 6.274 | **0.242 (+13.7 pp)** |
| oracle | 5989 | 6.416 | 0.157 |

`relief_e15` is **net worse than baseline LSTM** on the broader set:
+934 m³ pump (+16 %), +13.7 pp fallback, no improvement on max_p95.

**Failure attribution (9 worst cases)**: see
`f60_relief_e15_holdout_validation_v1/forecast_attribution/relief_bias_table.txt`.

`relief_e15` has a **systemic downward bias on b1/b2 forecast blocks** across
ALL these cases:

| metric | baseline | e15 | oracle |
|---|---:|---:|---:|
| mean b1/b0 across 9 failure cases | 1.074 | **0.910** | 1.319 |

e15 sits **below 1.0** (predicting future relief) on cases where oracle
and baseline both predict **above 1.0** (intensification). On the worst
case `onset_strong`, oracle says b1/b0=2.55 (strong upcoming intensification),
e15 says 0.95 (mild relief) — a 2.6× directional error. On
`fr_relief_06` (the worst single failure: +2.79° max p95, +10.35 pp fb),
oracle predicts mild rise (b1/b0=1.08), e15 predicts persistence-flat
(0.98) — direction missed.

**Training loss confirms the trade-off**: baseline validation loss ≈ 0.0361,
relief_e15 ≈ 0.0511 (+41 % MSE). The synth-relief emphasis amplified the
relief signal at the cost of overall forecast accuracy on non-relief
samples. On guard10 this trade happened to align with the 2-3 relief
cases driving Phase 6's win. On a broader case mix it inverts: more
non-relief cases lose than relief cases win.

### Verdict

`relief_e15` is **NOT a global LSTM upgrade**. It is a **relief-biased model
that wins on a narrow regime and loses on the rest**. Specifically:

- It is correct as a research artifact (demonstrating the emphasis-loss
  mechanism works in its intended regime).
- It is wrong as a production default. Reverting to baseline LSTM:
  - −934 m³ pump on the broader set
  - −13.7 pp fallback
  - no max_p95 regression

### Required actions

1. **No production code change is needed** — `scripts/analysis/run_prediction_primary_casebook.py`
   already defaults `--model-dir` to `outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1`
   (the baseline LSTM). `relief_e15` is only used when explicitly passed.
2. **Reclassify `relief_e15`** in this handoff and elsewhere as a
   *default-off research candidate*, not "current best".
3. **Phases 6-8 conclusions are partially superseded**:
   - Phase 6's "strict Pareto on guard10" is real on those 10 cases but
     does NOT generalize to broader validation.
   - SAR (Phase 7) was evaluated on top of `relief_e15`; its result
     numbers (3925 pump, 21.40 fb, 12753 t_over_5) are conditional on
     using `relief_e15` as the LSTM. Whether SAR still adds value on top
     of the baseline LSTM on the broader 20-case set is **unverified**
     and should be re-tested before SAR is treated as a holdout candidate.
4. **Do NOT** build gated / mixture / regime-aware versions of `relief_e15`
   as the next step. That would commit more engineering to a single
   biased model. The simpler conclusion — baseline LSTM is the production
   default — already gives the broader-set best result among learned
   models (5731 m³, 0.105 fb).
5. **If forecast-side work continues**, the right next experiment is
   **NOT** another emphasis variant. It is the cost-margin attribution
   from the prior session: determine whether better LSTM training can
   ever cross the planner's discrete action thresholds on the broader
   set, BEFORE training another model. See
   `outputs/wind_prediction/forecast_control_attribution_v1/review.md`
   for the framing.

### What this does NOT change

- h240/path research branch is still summarized as
  `outputs/wind_prediction/h240_path_research_branch_summary_v1/` —
  insufficient evidence, not a main line.
- Phase 1a–4 controller scaffolding (safety_floor, sustained_active_recompute)
  is still default-off and bit-identical to baseline LSTM behavior when
  not enabled. No revert needed there.

### Single-line summary

`relief_e15` is a relief-biased LSTM that beat baseline LSTM on a relief-
heavy 10-case subset and lost on a 20-case broader holdout. Production
default is the baseline LSTM. e15 stays in the repo as a research
candidate only.

**Artifact**:
`outputs/wind_prediction/f60_relief_e15_holdout_validation_v1/forecast_attribution/relief_bias_table.txt`

## Commands That Still Matter

Prepare the segmented dataset:

```bash
python scripts/data_preparation/prepare_ballast_wind_decision_dataset.py
```

Train the current segmented GRU:

```bash
python scripts/modeling/train_ballast_lstm.py \
  --dataset-dir data/processed/wind_ml_10min/ballast_decision_dwd_helgoland_segmented_v1 \
  --output-dir outputs/wind_prediction/gru_segmented_head_v1 \
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
  --model-type gru
```

Train the segmented LightGBM baseline:

```bash
python scripts/modeling/train_lightgbm_wind_baseline.py \
  --dataset-dir data/processed/wind_ml_10min/ballast_decision_dwd_helgoland_segmented_v1 \
  --output-dir outputs/wind_prediction/lightgbm_segmented_residual_e300 \
  --n-estimators 300 \
  --learning-rate 0.04 \
  --threads 8 \
  --threshold-mode best_f1
```

## Current Risks

- the present evidence is still single-station and partly low-height;
- GRU superiority over LSTM is real but small, not enough to be the sole paper
  contribution;
- the current control preview interface exists, but end-to-end control benefit
  has not been demonstrated yet;
- the most valuable next improvement may come from better offshore-height data,
  not from another small model tweak.
