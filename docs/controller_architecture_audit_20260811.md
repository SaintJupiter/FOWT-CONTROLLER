# Controller architecture audit (2026-08-11)

> Status note (2026-08-12): the Provider decomposition described below is now
> treated as the submitted-paper V1 compatibility path. The post-paper V2
> controller uses `wind_prediction.controller`; no new decision policy should
> be added to the Provider mixins. See `docs/controller_v2_framework.md`.

> Status note (2026-08-13): the V2 path now has an explicit framework-only
> platform profile, complete runtime platform identity, signed ballast mass
> properties, a direct controller-to-plant smoke runner and separate hashes for
> forecast payload, decision state and candidate ranking. Current evidence is
> framework connectivity evidence, not controller-performance validation.

## Scope

This audit covers the maintained path from forecast input to candidate action,
ballast target, pump execution, platform update, and validation evidence. The
paper-submission snapshot remains frozen separately. The work described here is
behavior-preserving architecture repair for the post-paper mainline.

## Maintained execution path

The normal validation entry is:

```text
versioned protocol JSON
  -> scripts/validation/run_control_protocol.py
  -> casebook compatibility runner
  -> forecast-assisted provider
  -> target transaction and pump execution
  -> platform update and protocol checker
```

The provider public entry is
`src/wind_prediction/ballast_planner_provider.py`. It is now a 69-line facade
assembled from responsibility-specific mixins. New validation work should not
call the historical casebook CLI directly.

## Repairs completed

### Provider decomposition

The original provider exceeded 11,000 lines and combined construction,
forecast acquisition, candidate evaluation, target lifecycle, safety rules,
stateful experimental policies, cycle execution, and telemetry. It is now
split into modules for:

- grouped configuration and defaults;
- immutable setting validation;
- mutable runtime state and reset;
- forecast admission and forecast features;
- candidate planning;
- target lifecycle and safety adjustments;
- economy and long-horizon policies;
- cycle orchestration;
- telemetry construction.

No behavior module exceeds 2,000 lines. The cycle orchestrator contains no
method longer than 250 lines. A structural regression test enforces these
limits so the provider cannot silently grow back into a monolith.

### Configuration and state ownership

The former constructor exposed roughly 290 control and diagnostic arguments.
The public provider now accepts `ProviderRuntimeInputs` and grouped
`ProviderConfig`; legacy keyword construction remains only as a compatibility
path. Literal defaults are centralized in `provider_defaults.py`, while
validation and normalization live in `provider_settings.py`.

Runtime state now has one initialization owner. Construction calls `reset()`
once, and `reset()` covers the complete forecast, target, safety, economy, and
long-horizon state. The previous duplicate initialization and incomplete reset
paths have been removed.

### Cycle and target execution

The per-bucket path is divided into explicit stages:

1. reset bucket-local diagnostics;
2. handle missing forecast samples conservatively;
3. generate and filter the candidate action;
4. commit the selected action to the primary target;
5. apply hold, event-refresh, or target-reuse policy;
6. record the resulting decision and execution evidence.

The pump-facing command path also records target ownership and ordered target
adjustments. Plant telemetry distinguishes commanded pump magnitude, signed
water transfer, inflow, outflow, per-tank mass change, and total ballast mass.

The legacy policy cycle is now separated into setpoint resolution, posture
feedback target calculation, prediction-target arbitration, and final target
commit. The posture controller intentionally remains warm while prediction
owns the target so that safety fallback always has a current feedback command.
The distinction between an enabled prediction target and an active target
change is covered by regression tests: a zero-change hold still belongs to the
prediction target channel.

Target-water shaping is now represented by `TargetSlewLimiter` in
`archive/legacy_fowt_control/target_execution.py`. It limits how quickly the
requested target moves but does not own pump state. Pump latching, flow-stage
selection, flow ramping, capacity clipping, and tank-mass integration remain
in the platform model. Legacy target-rate configuration keys are normalized at
the validation entry, and unsupported non-plant actuator authority now fails
fast.

### Validation and historical experiments

The formal entry accepts a versioned protocol rather than hundreds of command
line switches. The maintained production profile is registered separately from
diagnostic and historical profiles. Seventy-four isolated experiment profiles
remain available for reproduction but require an explicit opt-in and are not
part of the normal validation surface.

The three-case control-chain smoke gate fixes input files, forecast source,
planner configuration, cases, and expected outputs. After the current refactor:

- 80 unit and architecture tests pass;
- the smoke protocol passes without metric drift;
- the two active cases retain cumulative pump volumes of 751.34 and 560.94
  cubic metres;
- the low-disturbance case retains zero pump volume.

Architecture-v2 development runs now use a bounded validation protocol. Every
case lasts exactly six hours, one experiment contains no more than 30 cases,
and evidence is expanded through 1-, 3-, 10-, 20-, and 30-case stages. The
submitted 19.85% reduction remains historical evidence only; it is not encoded
as a regression target, tuning objective, pass threshold, or lower bound for
the new controller.

## Remaining work

The main control path is now structurally usable, but four burdens remain:

1. `casebook_profile_resolver.py` still contains the historical profile
   catalogue. It is isolated, not yet deleted, because reproduction evidence
   must be retained before pruning.
2. Several default-off overlays remain in compatibility configuration and
   telemetry schemas. The overlay registry identifies deletion candidates, but
   behavior-affecting removal still requires the smoke gate.
3. The legacy policy and validation runner still carry broad diagnostic output
   schemas. Their control decisions are separated into bounded stages, but the
   field catalogues should eventually move into dedicated telemetry modules.
4. The three-tank plant is a low-order model. Independent sea exchange is now
   auditable, and signed tank increments update total mass, centre of gravity
   and inertia. Exact attitude-dependent gravity, variable-mass momentum,
   tank fill/free-surface geometry and reference-platform calibration remain
   future model-development work.

## V2 framework evidence on 2026-08-13

The current V2 evidence uses all four configured six-hour framework cases. It
verifies 216,000 plant updates for each of eight variant trajectories, exact
target transfer, finite states, tank capacity, mass-property telemetry and
numerical mass balance. A same-state
probe holds all measured and controller state fixed while changing only the
actual model's future wind vectors. The future payload changes candidate
ranking and the executable three-tank target, providing direct evidence that
forecast content reaches decision logic.

The chain evidence is stored in
`outputs/controller_v2_chain_smoke_reference_incremental_v9_20260813/`. The
open-loop plant evidence is stored in
`outputs/platform_open_loop_audit_reference_incremental_v6_20260813/`. Both
identify their use as structural or open-loop evidence, not performance. The
chain manifest records the formal four-case framework suite as complete. The
repository suite contains 389 passing tests at this stage.

The connected runner now rejects controller/plant actuator-parameter
mismatches before simulation. The connected adapter also rejects target-slew
configurations because that limiter is not yet owned by the real command path;
isolated rollout studies may still use it. This closes the path where a
candidate could be scored against a shaped target but the plant could receive
the unshaped request.

## Current assessment

The architecture problem has moved from an uncontrolled 11,000-line provider
to a bounded research-code structure with one public entry, explicit runtime
stages, centralized state ownership, explicit target-versus-pump responsibility,
and repeatable validation. The remaining code volume is distributed behavior
and historical evidence rather than one executable monolith. Further work is
mainly historical-profile pruning, telemetry extraction, and model-fidelity
development rather than another controller rewrite.
