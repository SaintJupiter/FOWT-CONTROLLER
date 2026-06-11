# Control Chain Rectification Plan

Date: 2026-06-05

## Decision

Do architecture cleanup before any new forecast-signal tuning. The recent
forecast-pressure trust work produced useful telemetry but no closed-loop
material improvement, and `6.4opus.md` confirms that the main blocker is the
prediction-primary control chain, not another gate threshold.

The legacy closed-loop controller remains the stable baseline. The cleanup must
therefore preserve the existing reactive path while making the prediction layer
auditable, smaller, and easier to test.

## Current Chain

```text
wind replay / forecast adapter
  -> ForecastResult / ForecastContract
  -> ballast planner pressure blocks and sequence evaluation
  -> BallastPlannerPreviewProvider
  -> ClosedLoopPolicy.preview_trim_provider
  -> setpoint bias or prediction-primary mass target
  -> PI/MIMO controller, rate limiter, pump latch
  -> plant
  -> telemetry and casebook reports
```

## Diagnosed Problems

1. `preview_trim_provider` is no longer only a trim-bias provider. It carries
   forecast evidence, planner output, primary target commands, safety state,
   fallback reasons, and hundreds of telemetry fields.
2. `BallastPlannerPreviewProvider` has become a large decision bundle instead
   of a thin adapter. It mixes forecast trust, planner selection, overlays,
   target lifecycle, safety fallback, and telemetry assembly.
3. The forecast contract checks numeric shape, but not control semantics such
   as staleness, confidence, horizon, allowed control effect, or fail-closed
   behavior.
4. The casebook script contains both experiment harness logic and production
   controller profile decisions.
5. Many default-off overlays were already classified as redundant in
   `configs/planner_overlay_audit.md`, but they remain in the runtime surface.
6. Long experiments have been used where a static contract audit or three-case
   smoke would have stopped the branch earlier.

## Execution Phases

### Phase 0: Freeze Direction

Status: in progress.

- Stop adding new default-off overlays until the control chain has a canonical
  production configuration.
- Treat forecast-pressure trust, trusted-event gate, forecast-advised economy,
  h120 scheduling, and similar mechanisms as diagnostic unless explicitly
  promoted through a config review.
- Keep the legacy closed-loop baseline untouched.

Acceptance:

- A canonical config exists.
- A task list exists.
- A static audit can identify God-object pressure, redundant overlays, and
  profile sprawl without running simulations.

### Phase 1: Canonical Production Controller

Status: in progress.

- Create `configs/production_controller_v1.json`.
- Define the current clean main profile as the only production prediction-primary
  profile.
- Mark redundant and diagnostic mechanisms as forbidden for production use.
- Require every future experiment to declare a delta from the production config.

Acceptance:

- JSON validates.
- The config names the baseline, production profile, production switches, and
  forbidden production switches.

### Phase 2: Control-Facing Contracts

Status: in progress.

- Add a small set of typed contracts for:
  - `ForecastSignal`
  - `SupervisorDecision`
  - `ControlCommand`
- These are not wired into runtime yet. They are the target surface for future
  extraction and tests.

Acceptance:

- Contracts compile.
- Contracts include staleness, horizon, confidence, allowed effects, authority,
  and decision trace fields.

### Phase 3: Safe Pruning

Status: pending.

- First prune stale profile names and abandoned profile branches from the
  casebook script.
- Then remove redundant overlay code only after static audit and short smoke
  checks confirm no production behavior dependency.

Do not start by deleting the provider internals. The worktree is already dirty
and the provider is too entangled for blind removal.

Acceptance:

- The production profile remains runnable.
- Static audit shows fewer profile names and redundant-token references.
- No broad simulation is run unless the smoke gate passes.

### Phase 4: Module Extraction

Status: pending.

Extract from `BallastPlannerPreviewProvider` in this order:

1. `TelemetryAssembler`
2. `PrimaryTargetLifecycle`
3. `SafetySupervisor`
4. `ForecastTrustEvaluator`
5. Thin provider orchestrator

Acceptance:

- Each extracted module has a smaller interface than the implementation it
  hides.
- Safety decisions can be tested without running the planner.
- Telemetry fields have a schema version and no longer define the control
  interface.

### Phase 5: Forecast Signal Work

Status: pending.

Only after Phases 1-4:

- Add forecast smoothing or curve processing inside a `ForecastSignal` adapter.
- Compare learned forecast against persistence before allowing actuator-facing
  authority.
- Require fail-closed behavior for stale, low-confidence, or contradictory
  forecasts.

## Stop Rules

- Stop any experiment branch if a three-case smoke does not show a control
  effect.
- Stop any gate branch if it changes telemetry only.
- Stop any promotion if learned forecast is not materially better than
  persistence for the target metric.
- Stop any new overlay unless an older diagnostic overlay is removed or frozen.

## Immediate Deliverables

- `.taskmaster/docs/control_chain_rectification_20260605_prd.md`
- `.taskmaster/tasks/control_chain_rectification_20260605_tasks.json`
- `configs/production_controller_v1.json`
- `configs/casebook_profile_registry_v1.json`
- `configs/overlay_freeze_registry_v1.json`
- `configs/control_chain_smoke_gate_v1.json`
- `src/wind_prediction/control_contracts.py`
- `scripts/analysis/audit_control_chain_structure.py`
