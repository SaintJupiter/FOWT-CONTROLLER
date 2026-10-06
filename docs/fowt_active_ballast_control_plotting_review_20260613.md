# FOWT Active-Ballast / Control Papers: Metric and Plotting Review

Date: 2026-06-13

Purpose: record how related floating-wind active-ballast and platform-control
papers present control effects. This note is for the paper result-section
design, especially pump saving, attitude change, threshold exposure, and
prediction-supervised decision logic.

## Main Reading Judgment

The related literature almost never relies on a single average improvement
number. The usual pattern is:

1. define one or two primary benefit metrics;
2. report the actuator or energy cost separately;
3. report motion/load/safety metrics as guardrails;
4. compare controller variants or operating conditions;
5. show both representative time histories and aggregate statistics.

For this project, the most defensible result logic is therefore not "the
prediction controller simply makes everything better." It is:

> The prediction-supervised ballast layer reduces redundant ballast pumping in
> forecast-actionable windows. This saving is evaluated together with pump
> start/stop frequency, average dominant attitude change, and severe-attitude
> exposure. The attitude metric is a posture-margin cost or guardrail, not a
> hidden secondary objective.

## Papers Read and What They Show

### Stansby 2021: Pumping Between Floats for Pitch Reduction

Source: https://link.springer.com/article/10.1007/s40722-021-00194-y

This is the closest methodological analogue for "ballast transfer as a motion
control actuator." The paper studies pitch-motion reduction of a semi-sub wind
platform by pumping water internally between floats to generate a differential
head and balance heave-excitation moments.

Metric style:

- motion benefit: RMS and maximum hub acceleration;
- posture benefit: RMS and maximum pitch angle;
- actuator cost: average and maximum pump power as a proportion of wind power;
- operating envelope: wind speed and wave period/height;
- safety framing: operational hub acceleration and pitch limits are discussed.

Plot style:

- with/without pumping curves over wave period for hub acceleration, base
  acceleration, pitch angle, and pump power;
- with/without pumping curves over wind speed for 5, 10, and 20 MW scales;
- separate plots for RMS and maximum responses;
- pump power is not hidden: it is plotted as a cost metric.

Transfer to this paper:

- It is normal to present pump work as an explicit control cost rather than only
  as an internal variable.
- It is also normal to present motion improvement or motion penalty through both
  average/RMS and maximum/threshold-style quantities.
- For our study, the closest equivalents are cumulative pump volume, start/stop
  frequency, average dominant attitude delta, and T above attitude thresholds.

### Mahfouz et al. 2021: IEA 15 MW WindCrete and Activefloat

Source: https://wes.copernicus.org/articles/6/867/2021/

This paper is useful because Activefloat uses an active ballast system to keep
the static mean pitch around zero over operational wind speeds. The ballast
schedule is not presented as a "smart predictive controller"; it is an
operating-condition-based ballast arrangement linked to mean thrust.

Metric style:

- model definition tables: turbine, platform, ballast schedule, mooring, load
  cases;
- static equilibrium: surge, heave, pitch;
- natural frequencies: surge, heave, pitch, yaw, tower;
- dynamic response: surge, heave, and pitch time/frequency responses;
- controller sanity: step-wind response used to check that the tuned controller
  does not induce negative damping or pitch instability.

Plot style:

- geometry figures;
- table of active ballast schedule by turbulence model and wind speed;
- step-wind response curves;
- regular/irregular wave response figures with and without second-order wave
  forcing;
- operational wind-wave response figures for NTM/ETM/EWM load cases.

Transfer to this paper:

- A serious active-ballast paper documents the ballast schedule/control setup,
  load cases, and platform response before claiming performance.
- Our results should therefore clearly state what the 170 windows represent,
  what P/C/W regimes mean, what the closed-loop baseline is, and what the
  prediction-supervised layer changes.
- The paper should not make active ballast look like a high-frequency actuator.
  It is better described as a slow supervisory ballast regulation layer.

### Nanos et al. 2022: Differential Ballast for Vertical Wake Deflection

Source: https://wes.copernicus.org/articles/7/1641/2022/

This paper uses differential ballast to pitch the floater intentionally. Its
objective is wake steering rather than attitude minimization, but it is very
important for how to present trade-offs: desired platform tilt, water movement,
energy cost, fatigue loads, ultimate loads, and operating-envelope restrictions
are all reported.

Metric style:

- aerodynamic benefit: upstream power loss, downstream/cluster power gain;
- ballast feasibility: water volume moved, maneuver time, and energy
  expenditure;
- structural guardrails: fatigue DEL and ultimate load tables;
- operating policy: ballast steering is assumed inactive in extreme conditions;
- sensitivity: platform geometry and turbine-platform orientation are discussed.

Plot style:

- wake velocity contours and wake recovery curves;
- power drop/gain curves versus tilt angle and turbine spacing;
- load tables for lifetime DEL and ultimate loads;
- hydrostatic force/moment schematic for ballast calculation;
- ballast movement and energy estimates tied to target pitch angle.

Transfer to this paper:

- If our strategy releases attitude margin to save pumping, the right response
  is not to hide that release; it is to show the trade-off and safety
  guardrails.
- T>5, T>7.5, and T>10 seconds are analogous to their load/operating-envelope
  checks: these metrics define whether the saved pump effort is acceptable.
- It is legitimate to define "deactivate / tighten" behavior under severe risk,
  but it must be described as a controller guardrail, not post-hoc result
  cleaning.

### Meng et al. 2026: Active Ballasting System With Python PID

Source: https://www.sciencedirect.com/science/article/pii/S0029801825028252

Only the ScienceDirect article page and abstract/highlights were available in
this pass. Still, it is highly relevant because it is a recent Ocean Engineering
active-ballasting paper.

Confirmed from the page:

- a coupled FOWT and active-ballasting model is built with SESAM and TCP
  communication;
- the active-ballasting program is implemented in Python;
- the control algorithm uses pitch and roll angles with pump-flow constraints;
- the reported effect is mean roll/pitch reduction with little impact on other
  motions and improved average power under the same environment.

Transfer to this paper:

- This is a direct reference for the baseline idea: active ballast can be
  pitch/roll feedback plus pump-flow constraints.
- Our contribution should be distinguished from that: the proposed layer uses
  short-term wind prediction/risk information to supervise whether and when the
  ballast target is refreshed, held, or constrained.
- If possible later, retrieve the full PDF through institutional access for
  exact figure/table structure.

### Stockhouse et al. 2021/2022: Novel Actuated Platform

Source: https://arxiv.org/abs/2110.14169

This paper separates low-bandwidth variable ballast from high-bandwidth
platform actuation. That distinction is useful for our wording: ballast belongs
more naturally to slow trim/supervisory regulation than instantaneous motion
cancellation.

Metric style:

- controller variants are compared across wind speeds and turbulence seeds;
- generator speed and tower-base fore-aft bending moment are evaluated by
  standard deviation and absolute maximum;
- safety threshold lines are plotted for overspeed;
- mean power is also checked for economic relevance.

Plot style:

- controller-component matrix;
- gain schedules vs wind speed;
- standard deviation and maximum values in the same wind-speed plot, often
  solid/dashed;
- safety threshold shown as a reference line;
- combined controller variants compared against baseline.

Transfer to this paper:

- Use "mean/RMS/STD for ordinary variation" and "max/threshold exposure for
  safety boundary" as separate result channels.
- For our results, mean attitude delta should not be treated as equivalent to
  T>10 s. They answer different questions.

### Wakui et al. 2021: MPC With Previewed Wind/Wave Disturbances

Source: https://www.sciencedirect.com/science/article/abs/pii/S0960148121004687

The accessible preview is enough to establish why prediction belongs inside the
control chain. The paper develops model predictive control using previewed
spatial mean wind speed and wave height to stabilize power output and platform
motion and reduce dynamic loads.

Metric style from the preview:

- power-output stabilization;
- platform-motion reduction;
- dynamic-load reduction;
- comparison with gain-scheduled feedback and no-wave-preview variants;
- sensitivity to preview errors and internal model choices.

Transfer to this paper:

- This supports our chain: prediction is not an appendix metric, but a decision
  input to the supervisory control layer.
- We should include ablations such as learned prediction vs current-only vs
  persistence/shuffled prediction, because that is how the "preview is useful"
  claim becomes credible.

### Capaldo and Mella 2022: Platform Damping Strategy

Source: https://arxiv.org/abs/2211.10362

This is not active ballast, but it is a good example of control-result
presentation for platform motion and fatigue.

Metric and plot style:

- platform pitch time series under wave cases;
- tower-base moment density distributions;
- rotor speed time series;
- compensation gain evolution;
- multi-panel comparison across wind speeds for platform pitch, blade pitch,
  tower bending moment maximum/DEL, rotor speed, and generator power;
- min/mean/max/std table for a representative case;
- fatigue damage from rainflow counting and Miner rule.

Transfer to this paper:

- We should combine representative time series with aggregate tables.
- We should explicitly mention side effects: pump use decreases, but some
  attitude margin may be released; switch-frequency improvement may differ from
  pump-volume improvement.

### Gong et al. 2024: Robust Nonlinear Control

Source: https://arxiv.org/abs/2406.11158

This is useful mainly for its normalized comparison format.

Metric and plot style:

- model validation against FAST using average and RMS discrepancies;
- time histories for rotor-speed error, platform pitch rate, and blade pitch;
- normalized RMS values for rotor-speed error and platform roll/pitch/yaw rates;
- normalized DEL table for tower-base, blade-root, fairlead, and anchor loads;
- side effects are stated when one load metric does not improve.

Transfer to this paper:

- A normalized bar/line comparison can be used for learned vs current-only vs
  persistence/shuffled policies.
- Not every metric must improve, but the paper must state which metric is the
  primary objective and which is a guardrail.

### Sundarrajan et al. 2023: Pitch Constraints in Control Co-Design

Source: https://arxiv.org/abs/2310.13647

This paper is helpful for the attitude-constraint language. Platform pitch is
treated as an explicit constraint, and tighter constraints can reduce power or
make some designs infeasible.

Metric and plot style:

- optimal trajectories for generator speed, platform pitch, blade pitch,
  generator torque, power, and structural outputs;
- constraint satisfaction shown directly in time-history figures;
- average power, AEP, and LCOE heatmaps over design variables under different
  pitch constraints;
- infeasible cases are not hidden.

Transfer to this paper:

- If the advisor wants mean attitude increase near 0.4 deg, that should be
  handled as a stricter conservative-control variant, not by deleting many bad
  windows.
- A 0.4 deg variant should be presented as a trade-off curve: lower pump saving
  but tighter posture margin.

## Recommended Figure System for This Paper

### Figure A: Framework / Data Flow

Show the chain:

wind-history window -> LSTM wind-speed/wind-direction prediction -> risk/trend
features -> supervisory ballast decision -> target lifecycle / pump execution
-> attitude and pump metrics.

Purpose:

- proves prediction is inside the control chain;
- avoids the weak impression that prediction is only an offline label;
- separates prediction output from direct pump command.

### Figure B: Aggregate Regime Results

Use P/C/W regimes plus total. Avoid relying on one total table only.

Panels:

- cumulative pump-volume reduction;
- pump saving percentage;
- start/stop frequency reduction;
- average dominant attitude-angle delta;
- T>5, T>7.5, T>10 deltas.

Purpose:

- gives the same information as the current table, but makes the trade-off
  visually visible;
- prevents the 0.54-0.6 deg average attitude increase from being buried.

### Figure C: Window-Level Trade-Off

Use one point per validation window.

Suggested axes:

- x: pump saving percentage or saved volume;
- y: mean dominant attitude-angle delta;
- color: P/C/W regime;
- marker outline: whether T>7.5 or T>10 increased.

Purpose:

- shows whether bad attitude deltas are isolated or distributed;
- directly supports the conclusion that removing only the top 10 worst attitude
  windows cannot reduce the mean to 0.4 deg.

### Figure D: Worst-Mean-Attitude Diagnostic

Sort windows by mean dominant attitude-angle delta and show the top cases.

Columns:

- case id / regime;
- dominant axis or pitch/roll sign;
- saved pump volume and percent;
- start/stop change;
- mean attitude delta;
- T>5, T>7.5, T>10 deltas.

Purpose:

- this is the right place to analyze the "倾角变化最大的样本";
- it is diagnostic evidence, not a formal exclusion rule.

### Figure E: Representative Time Series

For several selected windows, show separated subplots:

- wind speed and wind direction, with prediction/risk markers if available;
- pitch and roll separately;
- pump flow or pump activity;
- cumulative pump volume;
- ballast mass and target mass.

Important:

- do not only plot a generic "axis" line for case diagnosis;
- use pitch and roll directly so the user/advisor can see which physical angle
  changed.

Purpose:

- helps readers understand whether the saving is caused by smarter target hold,
  delayed correction, early correction, or risk-triggered constraint.

### Figure F: Conservative Variant / Pareto Plot

If a stricter attitude version is run, show variants on a trade-off curve.

Possible variants:

- current main result;
- attitude-conservative supervisor;
- stricter severe-risk guard;
- current-only baseline;
- persistence/shuffled forecast ablation.

Metrics:

- x: pump saving or saved pump volume;
- y: mean attitude delta;
- bubble/label: T>7.5 or T>10 exposure.

Purpose:

- makes the 0.4 deg target a controller-design trade-off, not a post-hoc
  reporting trick.

## How to Explain the Current 0.5-0.6 deg Issue

The literature gives a reasonable way to handle this, but not a free pass.

Reasonable:

- say that pump saving is obtained by releasing limited posture margin in
  low-risk or forecast-actionable windows;
- report severe-attitude exposure separately;
- run a conservative variant if the advisor requires a smaller mean attitude
  delta;
- analyze the worst mean-attitude windows instead of hiding them.

Not reasonable:

- claim the posture is "basically unchanged" when the mean dominant attitude
  delta is around 0.54 deg;
- delete many windows just to force the mean near 0.4 deg;
- describe a deadband-only adjustment as the main algorithmic innovation;
- report only pump saving without the attitude and threshold guardrails.

## Recommended Wording Logic

Use this hierarchy:

1. Primary contribution:
   short-term wind prediction is converted into trend/risk information and used
   by the supervisory ballast decision layer.

2. Primary benefit:
   redundant pump actions are reduced in forecast-actionable windows.

3. Actuator-side evidence:
   cumulative pump volume and pump start/stop frequency decrease.

4. Posture-side guardrail:
   mean dominant attitude change and severe-threshold exposure are reported.

5. Conservative option:
   if a tighter mean-attitude requirement is imposed, a conservative variant is
   evaluated and reported as a pump-saving vs attitude-margin trade-off.

Suggested sentence:

> The proposed strategy should be interpreted as prediction-supervised
> ballast-economy control under attitude guardrails, rather than as an
> unconstrained attitude-minimization controller.

## Practical Decision for the Next Experiment

Do not spend time trying to "fix" the 0.54 deg number by sample deletion. The
previous top-N exclusion check already suggests that the attitude increase is
distributed across the validation set rather than caused by only 10 outliers.

Recommended next action:

1. extract the worst mean-dominant-attitude windows and inspect pitch/roll,
   pump, cumulative pump, and target lifecycle curves;
2. identify whether the problem comes from target hold, delayed refresh,
   too-loose low-risk suppression, wind-direction change, or decay windows;
3. if needed, run a small conservative canary variant on those windows first;
4. only after the worst-case mechanism is understood, rerun the full 170-window
   result.

## Source Boundary

Full/near-full text reviewed in this pass:

- Stansby 2021, Springer open access.
- Mahfouz et al. 2021, Wind Energy Science open access.
- Nanos et al. 2022, Wind Energy Science open access.
- Stockhouse et al. 2021/2022, arXiv.
- Capaldo and Mella 2022, arXiv.
- Gong et al. 2024, arXiv.
- Sundarrajan et al. 2023, arXiv.

Preview/abstract-level source checked:

- Wakui et al. 2021, ScienceDirect article preview.
- Meng et al. 2026, ScienceDirect article page with abstract/highlights.

