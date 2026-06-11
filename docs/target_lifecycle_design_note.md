# Target Lifecycle Design Note

Date: 2026-05-13

This note records the target/execution design decision after the `active_intent_reproposal` 2-case check. It is intentionally not a tuning note. No new controller rule is proposed here, and no case-specific fix is accepted as mainline.

## Current Conclusion

The current problem is no longer simply "the planner does not want to adjust." In `fr_relief_09`, the planner can keep outputting active actions, and a default-off reproposal prototype can force target updates and pump execution. That prototype reduced target reuse and fallback, but pitch still stayed high in parts of the window. In `lowrisk_clean`, the same prototype spent extra ballast water for a posture improvement that was not safety-critical.

So the useful conclusion is:

> Target lifecycle has a real interface problem, but "intent changed, therefore update target" is too broad for mainline.

The target manager needs a cleaner state lifecycle, not another comfort/veto/credit patch.

## Design Goal

The target manager should decide target reuse or target update in two distinct steps:

1. Did the planner's control intent actually change?
2. If it changed, is it necessary and safe to execute that change now?

The old behavior effectively collapsed this into action-name reuse:

```text
same action name -> reuse old target
```

That is too coarse. `active_small` is only an action class. It does not fully describe whether the intended pitch/roll correction direction or three-tank target proposal changed.

The reproposal prototype then went too far in the other direction:

```text
intent/proposal changed -> update target
```

That is also too coarse, because it can spend water in low-risk states.

The clean lifecycle should be:

```text
planner action and action_vec
-> intent_signature_changed
-> reproposal_eligibility
-> target proposal accepted/rejected
-> target completion by mass tracking only
```

## Layer 1: intent_signature_changed

This layer only answers:

> Is the planner asking for a materially different control intent than the previous bucket?

It may use:

- action name changes, such as `hold` to `active_small`;
- action vector changes in pitch/roll components;
- target proposal changes after mapping action vector to three tank masses;
- proposal direction changes, such as pitch-dominant, roll-dominant, or mixed;
- a small numerical deadband to ignore floating-point noise.

It must not decide whether the action is worth executing.

It should not use:

- current posture risk level;
- water cost;
- fallback state;
- whether the posture is already comfortable;
- whether the platform is naturally recovering.

Those are execution-eligibility questions, not intent-signature questions.

The output should be observable in logs:

```text
intent_signature_changed: true/false
intent_signature_reason
proposal_delta_mean_kg
proposal_axis
proposal_pitch_component
proposal_roll_component
```

## Layer 2: reproposal_eligibility

This layer answers:

> Given that intent changed, should the target manager actually accept a new target now?

It should protect the system from two failure modes:

- false stall: planner intent changed but target manager silently reuses old target;
- unnecessary correction: intent changed, but executing it only polishes an already-safe posture at high water cost.

It may consider:

- whether pitch/roll are outside the safety or engineering band;
- whether posture is already improving without a new target;
- whether the previous target has been reached by actual tank mass;
- whether pump activity has already stopped;
- estimated target change size and implied water cost;
- whether fallback is active and may be overriding the normal target;
- whether the new proposal mostly improves the main residual axis or only creates mixed/coupled motion.

It should produce explicit reasons:

```text
reproposal_eligible: true/false
reproposal_reason
reproposal_reject_reason
```

This layer should be conservative. A new target should be accepted because it is needed, not merely because the planner proposal is different.

## Target Completion Must Stay Clean

Target completion should continue to mean:

> Actual tank masses are close to the current target masses.

It should not be redefined as:

```text
target complete only if posture has recovered
```

That would mix actuator tracking with platform-state success and can create a deadlock where a target is never considered complete.

If the masses reach target but posture is still poor, the right response is not to mark the target incomplete forever. The right response is for `reproposal_eligibility` to decide whether a new target should be generated.

## What Can Stay Default-Off

These are useful as diagnostic or prototype mechanisms, but should not be enabled in the mainline until the lifecycle is redesigned and validated:

- `active_intent_reproposal`;
- intent signature trace fields;
- reproposal eligibility trace fields;
- action vector axis classification;
- proposal-to-target delta diagnostics;
- target/actual mass bucket trace;
- command-chain trace from planner intent to target, pump, mass, posture, and fallback.

Keeping these default-off is useful because they expose the control chain without claiming a final control law.

## What Should Not Enter Mainline

The following should not be accepted into the main controller as mainline logic:

- a rule that refreshes target only because pitch is high;
- a fr_relief_09-specific pitch bias;
- a rule that updates target whenever action name is unchanged but action vector changes;
- a rule that updates target whenever proposal delta exceeds a threshold;
- learned-only forecast credit, learned-only veto, or learned-only comfort release;
- changing target completion to include posture recovery;
- stacking comfort/veto/seqlead/evidence gates to hide lifecycle problems.

These rules may improve one plot, but they make the controller harder to explain and easier to overfit.

## Interpretation of the 2-Case Reproposal Result

The reproposal prototype proved one useful thing:

> Action name is not a sufficient target reuse key.

But it also proved the missing second layer:

> Proposal change is not sufficient execution justification.

In `fr_relief_09`, target updates and actual tank mass changes occurred, and fallback almost disappeared, but pitch still had high or reverse-response sections. That points downstream toward physical response, coupling, or action-to-response authority.

In `lowrisk_clean`, only one reproposal trigger added about 122.6 m3 of pump work. Pitch improved, but the state was not fallback-driven or clearly unsafe. That points to missing execution necessity, not missing intent detection.

## Minimal Future Implementation Path

If this design is implemented later, the validation should remain small and staged.

### Step 1: Unit or Dry-Run Intent Signature

Use fixed bucket states to verify:

- same action name plus changed action vector gives `intent_signature_changed = true`;
- same action name plus negligible proposal difference gives `intent_signature_changed = false`;
- hold-to-active and active-to-hold transitions are detected cleanly.

Exit condition:

> The intent signature layer correctly separates action class from actual control intent.

### Step 2: Eligibility Dry-Run

Use the same intent change under different posture states:

- high-risk posture and stalled target should be eligible;
- low-risk posture with no fallback and no urgent safety need should be rejected;
- fallback-dominated buckets should be labeled separately, not mixed with target stall.

Exit condition:

> Eligibility accepts necessary target updates and rejects low-value posture polishing.

### Step 3: Two-Case Closed Loop

Only then run:

- `fr_relief_09`;
- `lowrisk_clean`.

Required checks:

- target reuse stall is reduced in `fr_relief_09`;
- `lowrisk_clean` pump work does not increase just to polish a safe posture;
- fallback does not worsen;
- target updates have explicit, readable reasons.

### Step 4: Add One Guard Case

Add `fr_relief_01` only after the two cases pass.

Required checks:

- no new meaningless active behavior;
- no new pump burst;
- no fallback increase;
- pitch/roll response remains explainable.

### Step 5: Small Casebook Before Large Casebook

Only after the above passes should a 5-case check be considered. A 30-case run should be treated as a later robustness check, not as a way to discover the design.

## Reporting Standard

Any future target lifecycle experiment should report the full chain:

```text
planner action
action vector
intent signature changed or not
eligibility accepted or rejected
target before/after
actual mass before/after
pump work
fallback ratio
pitch/roll current-bucket response
pitch/roll next 1-2 bucket response
```

If one of these links is missing, the result should be treated as not diagnosable.

## Final Position

The target manager should be redesigned around a clear lifecycle:

```text
intent detection first,
execution eligibility second,
target update third,
mass tracking completion kept separate.
```

`active_intent_reproposal` should remain default-off as evidence and a prototype. It should not enter the mainline in its current form.

