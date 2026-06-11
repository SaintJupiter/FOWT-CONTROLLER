# Planner Overlay Audit

Date: 2026-05-11
Scope: PlannerConfig + BallastPlannerPreviewProvider default-off mechanisms
Trigger: 5 months of patches accumulated to compensate for two missing defaults
(`w_attitude_residual=0`, `posture_state_residual_active=False`). Both fixed
2026-05-11 (commits in `src/wind_prediction/ballast_planner.py`).

## Audit method

For each default-off field/mechanism, classify by:

- **REDUNDANT**: now subsumed by new defaults (w_attitude_residual=1.0,
  posture_state_residual_active=True, posture_hold_forecast_credit=0.0).
  Action: delete code or freeze and never enable.
- **DIAGNOSTIC**: independent purpose, keep default-off, document as diagnostic.
  Never enable in production profile.
- **CANDIDATE**: independent purpose, may be useful in main profile after
  further validation.
- **PRODUCTION**: already used in main profile, keep.

## A. PlannerConfig structural switches

| field | default | category | rationale |
|---|---|---|---|
| `envelope_use_discount` | True | PRODUCTION | Default behavior. Raw envelope is an alternative cost shape, switch as PROFILE option only. |
| `envelope_barrier_active` | False | PRODUCTION (enabled in main profile) | Penalize "envelope violated + no strong action" sequences. Independent from new defaults. KEEP. |
| `envelope_barrier_const` | 50.0 | PRODUCTION | Validated insensitive (5/20/50/100/500 identical). Keep at 50. |
| `posture_hold_barrier_active` | False | REDUNDANT | Old mechanism to penalize hold under high posture via *barrier const*. Now superseded by attitude_residual_cost × w_attitude_residual flowing through every block. Freeze; do not enable. |
| `posture_hold_action_uses_posture_vec` | False | REDUNDANT | Sub-option for posture_hold_barrier. Useless without barrier active. Freeze. |
| `posture_hold_low_risk_norm_max` | 0.0 | REDUNDANT | Sub-option. Freeze. |
| `posture_hold_forecast_credit` | 0.0 | PRODUCTION | Set to 0.0 (2026-05-11) to remove forecast_has_future asymmetry. Keep at 0. Setting >0 reintroduces learned-only credit and breaks fairness contract. |
| `posture_state_residual_active` | True | PRODUCTION | Core state-feedback input. New default 2026-05-11. |
| `posture_state_gain` | 0.40 | PRODUCTION | Verified by `single_bucket_posture_cost_check.py`. |

## B. Provider feedforward channels

| field | default | category | rationale |
|---|---|---|---|
| `ff_channel_enabled` | False | DIAGNOSTIC | Direct mass FF (bypass deadband). Earlier diagnostics showed +15-39% pump cost. Never enable as production. Keep code for future ablation. |
| `setpoint_channel_enabled` | True | PRODUCTION | Standard setpoint shift channel. Keep. |
| `setpoint_bias_sign` | 1.0 | PRODUCTION | Pump-first sign convention (see configs/planner_sign_contract.json). Keep. |

## C. Prediction-primary target lifecycle

| field | default | category | rationale |
|---|---|---|---|
| `prediction_primary_enabled` | False | PRODUCTION (enabled in main profile) | Gates whole primary mode. Keep. |
| `prediction_primary_scale` | 1.0 | PRODUCTION | Keep. |
| `primary_hold_target_mode` | "current" | PRODUCTION (set to "pause" in main profile) | Pause mode prevents active→hold→active target reset. Keep "pause". "pi_release" and "forecast_pause" are DIAGNOSTIC. |
| `event_reset_mode` | "action" | PRODUCTION | Standard. "target_change" and "active_bucket" are DIAGNOSTIC. |

## D. Pump suppression / preview overlays — REDUNDANT BLOCK

All of these were attempts to make learned look different from persistence
without going through cost function. New defaults remove the need.

| field | default | category | rationale |
|---|---|---|---|
| `pump_suppression_enabled` | False | REDUNDANT | Overlay to prevent pump restart based on forecast. Used as workaround for cost function ignoring forecast relief. Now: forecast naturally affects rollout cost; planner picks hold when future relief lowers terminal cost. Freeze, never enable. |
| `pump_suppression_restart_err_kg` | 2000 | REDUNDANT | Sub-param. Freeze. |
| `pump_suppression_relief_margin_norm` | 0.25 | REDUNDANT | Sub-param. Freeze. |
| `pump_suppression_low_risk_norm` | 0.75 | REDUNDANT | Sub-param. Freeze. |
| `pump_suppression_event_risk_guard_enabled` | False | REDUNDANT | Sub-overlay. Freeze. |
| `event_risk_pressure_boost_enabled` | False | REDUNDANT + HARMFUL | Tested: single-window benefit but +17-23% pump on 30-case. Never enable. Delete code candidate. |
| `event_risk_pressure_boost_gain/threshold/max` | 0.35/0.5/0.35 | REDUNDANT | Delete with parent. |
| `event_risk_pressure_floor_enabled` | False | REDUNDANT | Same problem as boost. Freeze, delete candidate. |
| `event_risk_pressure_floor_threshold/norm/max_lift` | 0.70/0.85/0.50 | REDUNDANT | Delete with parent. |
| `preview_lead_action_enabled` | False | REDUNDANT | 8-gate mechanism to advance future actions. Tested: overfit to case 01 onset, fires <5% buckets, +pump for marginal pitch improvement (not pump-first). Freeze, delete candidate. |
| `preview_lead_*` (8 sub-params) | various | REDUNDANT | Delete with parent. |
| `sequence_lead_action_enabled` | False | REDUNDANT | Similar idea to preview_lead. Tested: marginal effect on case 01 only, fragile. Freeze, delete candidate. |
| `sequence_lead_*` (4 sub-params) | various | REDUNDANT | Delete with parent. |

## E. Relief / posture overlays

| field | default | category | rationale |
|---|---|---|---|
| `relief_medium_cap_enabled` | False | PRODUCTION (enabled in main profile) | Caps active_medium magnitude when learned event_risk > threshold. KEEP. But re-evaluate after new defaults: cost function now penalizes attitude error, so medium might already be self-limited. Sensitivity test needed. |
| `relief_medium_cap_event_threshold` | 0.70 | PRODUCTION | Keep. |
| `relief_medium_cap_ratio` | 0.25 | PRODUCTION (0.30 in main profile) | Keep. |
| `relief_medium_cap_adaptive` | False | PRODUCTION (True in main profile) | Adaptive variant. Keep. |
| `hold_comfort_release_enabled` | False | REDUNDANT | Output-layer attitude correction. Tested: compresses learned vs reactive_current differentiation. Freeze, do not enable in any production profile. |
| `hold_comfort_*` (12+ sub-params) | various | REDUNDANT | Freeze with parent. Code retained as diagnostic but not in main. |
| `hold_comfort_forecast_veto_enabled` | False | REDUNDANT | Asymmetric (learned-only). Freeze. |
| `hold_comfort_veto_*` (7 sub-params) | various | REDUNDANT | Freeze with parent. |
| `hold_risk_micro_action_enabled` | False | REDUNDANT | Single-case fix attempt. Tested: case-specific, ungeneralizable. Freeze. |
| `hold_risk_micro_*` (6 sub-params) | various | REDUNDANT | Freeze. |

## F. Posture-aware action overlays — MOSTLY REDUNDANT

| field | default | category | rationale |
|---|---|---|---|
| `posture_action_refresh_enabled` | False | REDUNDANT | Refresh target when posture changes. Now: posture residual in cost function naturally drives target updates via active action selection. Freeze. |
| `quiet_posture_action_enabled` | False | REDUNDANT | Quiet-wind posture correction. Now: cost function active under quiet wind too because posture residual independent of wind. Freeze. |
| `quiet_posture_action_name` | "active_small" | REDUNDANT | Sub-param. Freeze. |
| `no_preview_myopic_horizon_enabled` | False | DIAGNOSTIC | When reactive_current forecast has no future, restrict planner to horizon=1. Useful for fairness experiment. Keep default-off but document. |
| `primary_stall_refresh_enabled` | False | **CANDIDATE** | Refreshes target when posture > stall AND target reached AND pump idle. Independent purpose: handles "target met but attitude not recovered" stall. Not subsumed by new defaults. Re-evaluate after new defaults run; may be unnecessary, may still be useful for severe cases like fr_relief_09. |
| `primary_stall_pitch_deg` | 5.0 | CANDIDATE | Tune after stall mechanism evaluation. |
| `primary_stall_roll_deg` | 4.0 | CANDIDATE | Same. |
| `primary_stall_target_err_kg` | 1500 | CANDIDATE | Same. |
| `primary_stall_min_age_s` | 300 | CANDIDATE | Same. |
| `primary_stall_pitch_axis_bias` | False | DIAGNOSTIC | Pitch-only refresh when roll low. Risk of overfitting fr_relief_09. Default-off; never auto-enable. |
| `primary_stall_pitch_axis_roll_max_deg` | 2.0 | DIAGNOSTIC | Same. |

## G. Hold→active escape overlays — REDUNDANT

| field | default | category | rationale |
|---|---|---|---|
| `no_unexplained_hold_action_enabled` | False | REDUNDANT | "Force active when posture > 3° for 2 buckets without explanation". Now: posture residual in cost makes hold cost grow with attitude error; planner naturally picks active. Freeze. |
| `no_unexplained_hold_*` (4 sub-params) | various | REDUNDANT | Freeze. |
| `active_intent_reproposal_enabled` | False | REDUNDANT | Re-propose active when previous active didn't take. Was workaround for target lifecycle bug. holdpause + target lifecycle pause already covers normal case; stall_refresh covers severe case. Freeze. |
| `active_intent_reproposal_*` (5 sub-params) | various | REDUNDANT | Freeze. |
| `medium_escalation_enabled` | False | REDUNDANT | Escalate small → medium when small insufficient. Now: cost function compares all 5 actions × 3 blocks (125 sequences) and selects globally optimal. Manual escalation is redundant. Freeze. |
| `recovery_mode_enabled` | False | REDUNDANT | Same general concept as no_unexplained_hold. Freeze. |

## H. Casebook profile names — STALE PROFILES TO PRUNE

40 profile choices declared in argparse, 22 with code branches. 18 profile
names exist only as argparse choices with no implementation.

### Production profiles (keep):
- `rawenv_holdpause_barrier_reliefcap_adaptive_v1` ← current main line

### Diagnostic profiles (keep, document):
- `rawenv_holdpause_barrier_v1` — A-class fix demonstration
- `rawenv_holdpause_barrier_reliefcap030_v1` — fixed-cap variant
- `rawenv_forecastpause_barrier_reliefcap_adaptive_v1` — forecast_pause hold mode test
- `rawenv_pirelease_barrier_reliefcap_adaptive_v1` — PI release test

### Stale (no code branch, delete from argparse choices):
- `rawenv_holdpause_barrier_reliefcap_state_v1`
- `rawenv_holdpause_barrier_reliefcap_adaptive_postureveto055_seqlead_v1`
- `rawenv_holdpause_barrier_reliefcap_adaptive_postureveto_nogate_v1`
- ...and ~15 more

### Failed/abandoned profiles (DELETE both name and code):
- `rawenv_holdpause_barrier_reliefcap_adaptive_micro_v1` — hold_risk_micro experiment
- `rawenv_holdpause_barrier_reliefcap_adaptive_nopreviewcomfort_v1` — comfort experiment
- `rawenv_holdpause_barrier_reliefcap_adaptive_evidencecomfort_v1` — forecast veto experiment
- `rawenv_holdpause_barrier_reliefcap_adaptive_currentcomfort_v1` — comfort experiment
- `rawenv_holdpause_barrier_reliefcap_adaptive_postureveto_v1` — posture veto
- `rawenv_holdpause_barrier_reliefcap_adaptive_postureveto055_v1` — posture veto variant
- `rawenv_holdpause_barrier_reliefcap_adaptive_evidence_holdfb_lite_v1`
- `rawenv_holdpause_barrier_reliefcap_adaptive_holdfb_lite_v1`
- `rawenv_holdpause_barrier_reliefcap_adaptive_postureaction_v1`
- `rawenv_holdpause_barrier_reliefcap_adaptive_postureaction_refresh_v1`
- `rawenv_holdpause_barrier_reliefcap_adaptive_quietposture_v1`
- `rawenv_holdpause_barrier_reliefcap_adaptive_quietposture_output_v1`
- `rawenv_holdpause_barrier_reliefcap_adaptive_holdcomfort_bucket_v1`
- `rawenv_holdpause_barrier_reliefcap_adaptive_boundedposture_v1`

## Summary of action items

### Immediate (Phase 2a — safe cleanups, no behavior change):
1. **Prune 18 stale argparse profile names** without code branches
2. **Delete or freeze** 14 failed profile branches in `if primary_control_profile == ...` blocks
3. Net code reduction: ~300-500 lines, no production behavior change

### Validation (Phase 2b — verify new defaults cover REDUNDANT mechanisms):
1. Run new defaults (post 2026-05-11) on 3-case smoke: fr_relief_09 + lowrisk_clean + fr_relief_01
2. Compare against current main profile output
3. If new defaults produce equivalent or better results on these 3 cases:
   - lowrisk_clean pitch < 3°
   - fr_relief_09 pump not worse than current main
   - fr_relief_01 unchanged
4. Then proceed to Phase 3

### Phase 3 — REDUNDANT mechanism deletion (after Phase 2b passes):
1. Delete `event_risk_pressure_boost_*` (8 fields + handler code)
2. Delete `event_risk_pressure_floor_*` (4 fields + handler code)
3. Delete `preview_lead_*` (9 fields + handler code)
4. Delete `sequence_lead_*` (5 fields + handler code)
5. Delete `hold_comfort_*` (20 fields + handler code)
6. Delete `hold_risk_micro_*` (7 fields + handler code)
7. Delete `pump_suppression_*` (5 fields + handler code)
8. Delete `posture_action_refresh_*`, `quiet_posture_action_*`,
   `no_unexplained_hold_*`, `active_intent_reproposal_*`,
   `medium_escalation_*`, `recovery_mode_*` (and sub-params)
9. Delete `posture_hold_barrier_*` from PlannerConfig (now subsumed)

Expected line count reduction: ~1500-2000 lines from
`src/wind_prediction/ballast_planner_provider.py` (currently ~2700 lines) and
~150 lines from `src/wind_prediction/ballast_planner.py`.

### Phase 4 — CANDIDATE re-evaluation:
1. `primary_stall_refresh_enabled` — only kept CANDIDATE mechanism
2. Run with new defaults: does fr_relief_09 stall still occur?
3. If yes: enable stall_refresh with default thresholds
4. If no: freeze stall_refresh as DIAGNOSTIC

## I. 2026-05-16 addendum: attitude_zone_form prototype (DIAGNOSTIC, do not enable)

Implemented `attitude_zone_form: str = "quadratic" | "smooth_huber"` in
PlannerConfig. Default remains "quadratic" so all current 5-case results stay
valid. The smooth_huber form was investigated to address an observed
"bistable" objective:

  - quadratic + w_attitude=0 (legacy baseline): hold always wins, long-hold
    parked at 4-5deg posture.
  - quadratic + w_attitude=1 (current main): attitude residual squared cost
    dominates pump cost (~30x at residual_norm=2), active over-fires,
    lowrisk_clean over-pumps without measurable benefit.

smooth_huber attempts to introduce a deadzone:

  cost = max(0, residual_norm - delta)**2

Cost-replay validation (scripts/analysis/cost_replay_attitude_zone_v4.py):

| scenario | quadratic | smooth_huber (delta=1.5, w=3) | desired |
|---|---|---|---|
| S1 lowrisk_clean static high posture | active_medium (over-pump) | hold | hold |
| S2 fr_relief_09 high wind | active_small | active_small | active_small |
| S3 fr01 weak benefit | active_medium | hold | hold |
| S4 onset rising future risk | active_medium | **hold** | active_medium |

smooth_huber FIXES lowrisk over-pump and weak-benefit (S1, S3) but LOSES
onset preemption (S4). This is a **structural property of the deadzone
form**: at the zone boundary (residual = delta), the cost slope is exactly
zero, so the marginal cost of an approaching-but-not-yet-violating residual
is zero. Quadratic has positive slope everywhere, which is what drives
preemptive response.

**Decision**: smooth_huber NOT enabled in production. Paper's core claim
("preview-driven preemptive action") requires preemption sensitivity, which
the deadzone destroys.

The mechanism is kept as default-off PlannerConfig fields for two reasons:
1. Reproducibility — the 5-case cost-replay table is part of the paper's
   limitations discussion ("we considered a deadzone form; here is why we
   rejected it").
2. Future hybrid work — a form that preserves boundary slope while having
   reduced amplitude inside a soft zone (e.g., linear-floor smooth_huber)
   may be developed later. The plumbing supports it.

**Never** set `attitude_zone_form="smooth_huber"` in a production profile.
This is purely diagnostic.

### Result: clean main profile
After Phase 3, the production controller should be defined by:
- `prediction_primary_enabled=True`
- `primary_hold_target_mode="pause"`
- `envelope_use_discount=False` (raw envelope)
- `envelope_barrier_active=True`
- `relief_medium_cap_enabled=True`
- `relief_medium_cap_adaptive=True`
- PlannerConfig defaults (which now include posture_state_residual_active=True,
  w_attitude_residual=1.0, w_terminal_residual=1.0,
  posture_hold_forecast_credit=0.0)

That's it. ~6 production switches, all with mechanism explanations.
Profile name can be simplified to `posture_aware_v1` or `state_feedback_v1`.

## What NOT to do

- Do not write a new `planner_objective_v2` prototype. The 4 default flips on
  2026-05-11 already implemented "unified objective" Codex was asking for.
- Do not re-enable any REDUNDANT mechanism during Phase 3 validation. If new
  defaults fail a case, that's information; don't paper it over with overlay.
- Do not delete CANDIDATE (stall_refresh) before Phase 4 evaluation.
- Do not change main profile name until Phase 3 complete; renaming mid-stream
  breaks comparability with prior casebook results.
