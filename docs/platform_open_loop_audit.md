# Platform open-loop internal-consistency audit

## Purpose

This audit isolates the existing platform plant from the forecast controller.
It checks the implementation's declared load directions, load composition,
incremental-reference settling and pitch/roll free decay before any controller
parameter is tuned.

The audit is an internal mathematical-consistency check of the low-order model.
It is not an OpenFAST comparison, physical validation or controller-performance
test.

## Active load channels

`FloatingPlatform.generalized_load_components(...)` reports the exact terms
used by the six-degree-of-freedom integrator:

- rotor wind load;
- irregular-wave load;
- quadratic hydrodynamic drag;
- horizontal mooring restoring load and fairlead moment;
- hydrostatic restoring load;
- dry-platform gravity moment;
- ballast gravity force and moment;
- linear damping;
- their total generalized load.

All vectors follow `[surge, sway, heave, roll, pitch, yaw]`. Forces use newtons
and moments use newton-metres.

## Baseline command

```bash
PYTHONPATH=src .venv312/bin/python3.12 \
  scripts/validation/run_platform_open_loop_audit.py \
  --output-dir outputs/platform_open_loop_audit/default \
  --platform-profile default
```

The formal audit requires explicit, readable mooring and thrust-curve files.
It does not silently replace either input with a fallback relation.

## Outputs

- `audit_summary.json`: input provenance, incremental-reference residuals, free-decay
  periods, model scope and directional checks;
- `load_channel_scan.csv`: every load component for cardinal wind directions
  and positive/negative displacement perturbations;
- `zero_load_settle.csv`: historical filename for the incremental-reference
  zero-residual settling trace; it is not an absolute static-equilibrium result;
- `free_decay_pitch.csv` and `free_decay_roll.csv`: small-angle open-loop decay
  traces about the settled state;
- `ballast_increment_scan.csv`: signed inflow, outflow and balanced two-port
  changes in mass, centre of mass, inertia and ballast gravity load.

`audit_summary.json` also records the complete runtime platform identity and
SHA-256 hashes for input files, runtime sources and generated outputs.

## Deferred by design

- exact attitude-dependent gravity generalized forces and variable-mass
  momentum-flux terms;
- tank fill geometry and free-surface effects;
- recalibrating hydrostatic, damping or wave-RAO parameters;
- changing forecast admission, candidate actions or controller costs;
- using pump reduction or posture metrics as an acceptance target.

## Baseline result on 2026-08-12

The default-profile audit completed with the repository's mooring workbook and
hub-height thrust curve. All cardinal wind, restoring-direction and load-sum
checks passed. At an audit wind speed of 12 m/s, the thrust lookup returned
1.74 MN and the 127 m aerodynamic arm produced a 220.98 MN-m roll or pitch
moment depending on wind direction.

After 600 s with zero wind, zero waves and no controller action, the model
settled near:

- surge/sway: 0.099 m / 0.099 m;
- heave: -8.095 m relative to the model's zero coordinate;
- roll/pitch: -0.00023 deg / -0.01188 deg;
- pitch and roll free-decay period: approximately 38 s.

The remaining force and moment residual norms at the sampled settled state were
1.26 kN and 1.25 kN-m, respectively. The result is sufficient as a numerical
baseline, but the -8.095 m heave offset shows that the model's
zero coordinate and displaced-volume reference need to be made explicit before
the posture response is treated as physical evidence. The fixed rotational
inertia and deferred ballast mass-property integration also need to be examined
before the 38 s period is used for model calibration.

The generated evidence is stored in
`outputs/platform_open_loop_audit/default_20260812/`.

## Research incremental audit on 2026-08-13

The `research_incremental_v1` profile passed twelve open-loop checks covering
load directions, load composition, signed ballast mass and first moment,
inertia, ballast gravity load, fixed added mass, independent inflow/outflow and
balanced two-port exchange. Its initial working ballast is the incremental
reference, and pump execution updates total mass, centre of mass and rigid-body
inertia.

This audit remains a low-order framework check. It does not validate the tank
geometry, added-mass coefficients, damping, waves or mooring response against
OpenFAST. It also does not establish an absolute static equilibrium: the
reference load is removed by construction, and the present state derivative
uses the small-angle approximation in which generalized attitude rates are
identified with angular-velocity channels. The traceable evidence is stored in
`outputs/platform_open_loop_audit_reference_incremental_v6_20260813/`.
