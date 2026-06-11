# Paper Validation Strategy

Last updated: 2026-06-11

Purpose: freeze the paper-facing validation strategy after the 2026-06-05
deep-research review. This is a writing and experiment-planning contract. It
does not change controller behavior.

## Accepted Methodology Reading

The deep-research conclusion is accepted with one project-specific refinement:

- A 6 h horizon is acceptable as the **main paper horizon** only when it is
  framed as short-horizon, episode-level, forecast-assisted control evidence.
- A 6 h result must not be written as day-scale, deployment-scale, or universal
  long-duration pump saving.
- A transparent 12 h mixed-regime robustness check should be included, because
  the current project already has 12 h evidence showing modest but useful
  aggregate saving and mild service-band trade-offs.
- A 24 h result is not required for the current short-paper claim. It remains a
  production-freeze or deployment-scale evidence requirement.

The practical reason is claim alignment. The controller consumes rolling
10 min wind buckets and acts as a short-term forecast supervisory layer. A
6 h broad episode gives many forecast/control buckets while still matching the
short-horizon control question. A 12 h run is useful as a robustness disclosure.
A 24 h run answers a stronger lifecycle-debt and deployment question that is not
the current paper's headline.

## Current Paper Claim

The current paper should claim:

> Short-term wind-condition prediction can support active-ballast supervisory
> regulation by reducing unnecessary ballast pumping in forecast-actionable
> episodes while keeping severe attitude-exposure metrics bounded in the tested
> cases.

The current paper should not claim:

> The final controller provides universal long-duration pump saving across all
> wind regimes.

It should also not claim:

> A 30% saving is the general 12 h or 24 h production result.

## Current Controller Evidence Boundary

Do not mix deadband evidence into the current production-near controller claim.

The current production profile is:

```text
rawenv_holdpause_barrier_reliefcap_adaptive_v1
```

The current 12 h production-near candidate is the same profile with:

```text
FOWT_TUNE_GAIN=0.45
```

This is a near-neighbor tuning of the production profile, not a deadband branch,
not P2 guarded economy, and not C3 gusty hold-current. It should be reported as
a mixed-regime robustness result on the guard10 set.

## Fixed 12 h Robustness Facts

The current 12 h guard10 mixed-regime evidence is:

| variant | pump work | saving vs current-only | `t>5 deg` | `t>7.5 deg` | `t>10 deg` |
|---|---:|---:|---:|---:|---:|
| current-only | 12800.43 m3 | 0.00% | 11.0310% | 0.0785% | 0.0023% |
| production, gain 0.40 | 12406.56 m3 | 3.08% | 11.3125% | 0.0785% | 0.0023% |
| production-near, gain 0.45 | 11582.94 m3 | 9.51% | 11.4741% | 0.0683% | 0.0023% |

Reading:

- gain 0.45 improves the 12 h mixed-regime pump saving from 3.08% to 9.51%.
- gain 0.45 increases service-band exposure slightly (`t>5 deg`) and should be
  described as an economy/safety-margin trade-off.
- gain 0.45 does not increase the operating-limit tail in this run (`t>10 deg`
  unchanged) and reduces `t>7.5 deg` slightly.
- gain 0.45 is a candidate for paper-facing robustness evidence, not yet a
  frozen production default.

Output directories:

```text
outputs/wind_prediction/guard10_12h_current_only_baseline_20260605
outputs/wind_prediction/guard10_12h_production_learned_candidate_20260605
outputs/wind_prediction/guard10_12h_production_learned_gain045_20260605
```

## Required Main Paper Evidence

The main result should be a broad 6 h casebook or a predeclared 6 h window set.
It must avoid hand-picked showcase-only evidence.

## Current Sample-Size Support

The current broad closed-loop validation basis is documented in
`docs/validation_sample_size_support_20260606.md`.

Short reading:

- FINO1 processed wind record: 978,280 ten-minute rows from 2005-01-01 to
  2025-01-01 23:50, about 18.6 effective years of valid 10 min data.
- Forecast dataset split: 638,631 train samples, 133,949 validation samples,
  and 140,173 test samples, split chronologically.
- Closed-loop control validation: 221 predeclared 6 h evaluation windows in the
  held-out test period, totaling 1326 h.
- The closed-loop validation duration is about 5.7% of the test split by 10 min
  time-equivalent scale.
- The 221 windows include 101 mixed cases and a 120-case positive-regime
  expansion. They should be described as evaluation windows, not all as fully
  independent meteorological events, because some starts are adjacent.

Paper implication:

- The sample volume is sufficient for a short-horizon, episode-level,
  regime-stratified control claim.
- It is not sufficient for a 20-year deployment-scale or certification-level
  load/fatigue claim.

Minimum execution principles:

1. Use predeclared cases/windows before reading final performance.
2. Include current-only or equivalent no-future-information comparison.
3. Use the same control profile when claiming forecast-information value.
4. Report all cases, including weak or negative cases.
5. Report pump work together with attitude exposure.
6. Keep deadband, P2, and C3 evidence separate unless the paper explicitly
   changes to a regime-specialist paper.
7. Treat fixed conservative deadband runs as baselines, not as the forecast
   algorithm contribution.
8. For any claim about forecast-supervised decision value, follow the
   diagnosis -> shadow policy -> canary replay -> learned/current-only/
   persistence/shuffled ablation -> full casebook gate in
   `docs/control_validation_protocol.md`.
9. Do not add one-off controller gates or low-level threshold patches directly
   into the main paper result path before they pass as logged shadow policies.

Preferred 6 h result table:

- pump work and pump saving percentage;
- pitch/roll p95;
- `time/max_continuous/area` above 3, 4, 5, 7.5, and 10 deg;
- fallback ratio;
- latch switches or target lifecycle diagnostics where useful;
- per-case paired deltas, not only aggregate averages.

## Role of 12 h and 24 h

For the current short paper:

- 6 h broad evidence is the main mechanism/effectiveness evidence.
- 12 h mixed-regime guard evidence is the robustness and honesty check.
- 24 h evidence is optional and should not be forced unless the claim becomes a
  production/deployment or day-scale operational claim.

For production freeze:

- 6 h paper evidence is insufficient.
- 12 h guard evidence is useful but still not enough by itself.
- 24 h or broader lifecycle-debt testing remains required before changing the
  production default.

## Safe Manuscript Language

Safe:

```text
The proposed short-term prediction-supervised ballast strategy reduces pump
work in forecast-actionable episodes and maintains severe attitude-exposure
metrics within the tested benchmark.
```

Safe:

```text
In a 12 h mixed-regime guard check, the production-near gain-0.45 candidate
reduced total pump work by 9.51% relative to current-only. The result is a
robustness trade-off rather than a universal long-duration saving claim:
service-band exposure increased slightly, while the severe and operating-limit
tails did not increase.
```

Avoid:

```text
The proposed controller achieves 30% long-duration pump saving.
```

Avoid:

```text
The controller is uniformly better in all regimes and all attitude metrics.
```

Avoid:

```text
Deadband/P2/C3 savings prove the current production profile saving.
```

## Next Execution Checklist

Before drafting final results:

1. Freeze the exact 6 h casebook/window selection rule.
2. Run current-only and learned/candidate variants under the same casebook.
3. Summarize per-case paired deltas and aggregate metrics.
4. Include the 12 h gain-0.45 robustness table as appendix or robustness
   subsection.
5. State explicitly that 24 h deployment-scale validation is future work unless
   a 24 h campaign is actually run and stable.
