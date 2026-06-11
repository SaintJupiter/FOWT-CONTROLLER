#!/usr/bin/env python3
"""Audit controller constraints before prediction-primary redesign.

The goal is not to prove the true hardware spec. It is to separate the current
control stack into:
  - constraints that look physical / hardware-like and should remain in any
    safety-realistic controller;
  - heuristic damping layers that may be removed in a research-minimal profile
    to expose the forecast controller's upper-bound potential.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "wind_prediction" / "control_constraints_audit"


def item(
    key: str,
    value: Any,
    unit: str,
    subsystem: str,
    classification: str,
    safety_realistic: str,
    research_minimal: str,
    source_refs: list[str],
    rationale: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "unit": unit,
        "subsystem": subsystem,
        "classification": classification,
        "safety_realistic": safety_realistic,
        "research_minimal": research_minimal,
        "source_refs": source_refs,
        "rationale": rationale,
    }


def build_audit() -> dict[str, Any]:
    items = [
        item(
            key="tank_capacity_kg",
            value=1850.0 * 1025.0,
            unit="kg/tank",
            subsystem="plant_physics",
            classification="physics",
            safety_realistic="keep_hard",
            research_minimal="keep_hard",
            source_refs=["archive/legacy_fowt_control/core_model.py:59"],
            rationale="Tank capacity is a physical storage bound; removing it would make mass commands impossible.",
        ),
        item(
            key="pump_rate_schedule_m3_min",
            value=[
                [3000.0, 15.0],
                [2000.0, 14.0],
                [1000.0, 12.0],
                [700.0, 10.0],
                [500.0, 8.0],
                [300.0, 6.0],
                [200.0, 4.0],
                [0.0, 0.0],
            ],
            unit="[kg_error, m3/min]",
            subsystem="pump_actuator",
            classification="hardware_like_model",
            safety_realistic="keep_hard",
            research_minimal="keep_hard_or_single_max_rate_variant",
            source_refs=["archive/legacy_fowt_control/core_model.py:65-74"],
            rationale="This encodes actuator flow authority. A single max-rate variant can test upper-bound forecast value, but physical validation should keep it.",
        ),
        item(
            key="pump_stop_err_kg",
            value=300.0,
            unit="kg",
            subsystem="pump_latch",
            classification="validated_model_hysteresis_tuning",
            safety_realistic="keep_base",
            research_minimal="replace_with_prediction_decision_or_lower_to_physical_min",
            source_refs=[
                "archive/legacy_fowt_control/defaults.py:88",
                "archive/legacy_fowt_control/core_model.py:517-524",
                "archive/legacy_fowt_control/data/FOWT验证终审评估.md:22-25",
            ],
            rationale="The project has validated hysteresis behavior, but the current default value is an engineering baseline, not an external hardware spec.",
        ),
        item(
            key="pump_restart_err_kg",
            value=500.0,
            unit="kg",
            subsystem="pump_latch",
            classification="validated_model_hysteresis_tuning",
            safety_realistic="keep_base",
            research_minimal="replace_with_prediction_decision_or_raise_dynamically",
            source_refs=[
                "archive/legacy_fowt_control/defaults.py:89",
                "archive/legacy_fowt_control/core_model.py:532-534",
                "archive/legacy_fowt_control/data/FOWT验证终审评估.md:22-25",
            ],
            rationale="Useful anti-chatter hysteresis, but fixed context-blind restart threshold is exactly the layer a prediction-primary controller should replace.",
        ),
        item(
            key="pump_min_on_s",
            value=20.0,
            unit="s",
            subsystem="pump_latch",
            classification="hardware_like_model",
            safety_realistic="keep_hard",
            research_minimal="keep_hard",
            source_refs=[
                "archive/legacy_fowt_control/defaults.py:90",
                "archive/legacy_fowt_control/core_model.py:517-520",
                "archive/legacy_fowt_control/data/FOWT验证终审评估.md:22-25",
            ],
            rationale="Minimum on-time protects against rapid cycling; treat as hardware-like until a manual proves otherwise.",
        ),
        item(
            key="pump_min_off_s",
            value=12.0,
            unit="s",
            subsystem="pump_latch",
            classification="hardware_like_model",
            safety_realistic="keep_hard",
            research_minimal="keep_hard",
            source_refs=[
                "archive/legacy_fowt_control/defaults.py:91",
                "archive/legacy_fowt_control/core_model.py:532-534",
                "archive/legacy_fowt_control/data/FOWT验证终审评估.md:22-25",
            ],
            rationale="Minimum off-time protects against rapid restarts; keep in both profiles.",
        ),
        item(
            key="pump_hold_before_stop_s",
            value=10.0,
            unit="s",
            subsystem="pump_latch",
            classification="anti_chatter_tuning",
            safety_realistic="keep_base",
            research_minimal="optional_disable",
            source_refs=["archive/legacy_fowt_control/defaults.py:92", "archive/legacy_fowt_control/core_model.py:517-520"],
            rationale="A dwell before stop reduces tail chatter but is not documented as a physical hard constraint.",
        ),
        item(
            key="pump_ramp_up_m3_min_per_s",
            value=2.0,
            unit="m3/min/s",
            subsystem="pump_actuator",
            classification="hardware_like_model",
            safety_realistic="keep_hard",
            research_minimal="keep_hard_or_infinite_upper_bound_variant",
            source_refs=["archive/legacy_fowt_control/defaults.py:99", "archive/legacy_fowt_control/core_model.py:572-579"],
            rationale="Rate-release dynamics model actuator rate changes. Keep for realistic runs; infinite-ramp variant is only an upper-bound test.",
        ),
        item(
            key="pump_ramp_down_m3_min_per_s",
            value=3.0,
            unit="m3/min/s",
            subsystem="pump_actuator",
            classification="hardware_like_model",
            safety_realistic="keep_hard",
            research_minimal="keep_hard_or_infinite_upper_bound_variant",
            source_refs=["archive/legacy_fowt_control/defaults.py:100", "archive/legacy_fowt_control/core_model.py:572-579"],
            rationale="Same as ramp-up; this should not be removed in safety-realistic mode.",
        ),
        item(
            key="command_rate_limiter.rate_limit_m3_min",
            value=10.0,
            unit="m3/min",
            subsystem="command_chain",
            classification="authority_shaping_tuning",
            safety_realistic="keep_base",
            research_minimal="remove_if_plant_rate_schedule_kept",
            source_refs=["archive/legacy_fowt_control/defaults.py:80", "archive/legacy_fowt_control/controllers_extras.py:59-89"],
            rationale="This is an upper-layer target limiter in addition to the plant pump schedule. For prediction-primary research it can duplicate physical rate limits.",
        ),
        item(
            key="deadband_pitch",
            value=1.0,
            unit="deg",
            subsystem="pi_controller",
            classification="control_tuning",
            safety_realistic="keep_base_or_soften_after_sensitivity",
            research_minimal="remove_from_prediction_primary_path",
            source_refs=["archive/legacy_fowt_control/defaults.py:14", "archive/legacy_fowt_control/controllers.py:192-205"],
            rationale="PI deadband is useful for reactive anti-chatter, but it filters forecast signals and should not gate prediction-primary pump decisions.",
        ),
        item(
            key="deadband_roll",
            value=0.8,
            unit="deg",
            subsystem="pi_controller",
            classification="control_tuning",
            safety_realistic="keep_base_or_soften_after_sensitivity",
            research_minimal="remove_from_prediction_primary_path",
            source_refs=["archive/legacy_fowt_control/defaults.py:15", "archive/legacy_fowt_control/controllers.py:192-205"],
            rationale="Same as pitch deadband.",
        ),
        item(
            key="deadband_exit_ratio",
            value=0.5,
            unit="ratio",
            subsystem="pi_controller",
            classification="control_tuning",
            safety_realistic="keep_base",
            research_minimal="remove_from_prediction_primary_path",
            source_refs=["archive/legacy_fowt_control/defaults.py:17", "archive/legacy_fowt_control/controllers.py:192-205"],
            rationale="Reactive hysteresis should not be the main prediction-primary economic gate.",
        ),
        item(
            key="controller_filter_tau",
            value=25.0,
            unit="s",
            subsystem="pi_controller",
            classification="signal_filter_tuning",
            safety_realistic="keep_base",
            research_minimal="remove_or_reduce_in_prediction_path",
            source_refs=["archive/legacy_fowt_control/defaults.py:18", "archive/legacy_fowt_control/controllers.py:142-150"],
            rationale="Sensor filtering may be needed for PI safety, but prediction-primary should optimize from forecast/state estimates directly.",
        ),
        item(
            key="controller_update_interval",
            value=10.0,
            unit="s",
            subsystem="pi_controller",
            classification="control_tuning",
            safety_realistic="keep_base",
            research_minimal="decouple_prediction_bucket_from_pi_update",
            source_refs=["archive/legacy_fowt_control/defaults.py:19", "archive/legacy_fowt_control/controllers.py:160-166"],
            rationale="Reactive PI cadence should not define the prediction-primary planning cadence.",
        ),
        item(
            key="setpoint_shaper_alpha",
            value=0.98,
            unit="ratio",
            subsystem="target_shaping",
            classification="anti_chatter_tuning",
            safety_realistic="keep_for_pi_setpoints",
            research_minimal="bypass_for_mass_schedule",
            source_refs=["archive/legacy_fowt_control/defaults.py:82", "archive/legacy_fowt_control/controllers_extras.py:36-54"],
            rationale="Useful for smooth setpoints, but it is another layer that dilutes forecast action if used as the main channel.",
        ),
        item(
            key="setpoint_rate_deg_s",
            value=0.02,
            unit="deg/s",
            subsystem="target_shaping",
            classification="anti_chatter_tuning",
            safety_realistic="keep_for_pi_setpoints",
            research_minimal="bypass_for_mass_schedule",
            source_refs=["archive/legacy_fowt_control/defaults.py:83", "archive/legacy_fowt_control/controllers_extras.py:36-54"],
            rationale="Same as shaper alpha.",
        ),
        item(
            key="trim_freeze_and_pressure_gates",
            value={
                "sat_gate_on": 0.95,
                "sat_gate_off": 0.85,
                "cmd_gap_decay_start_kg": 5000.0,
                "cmd_gap_decay_full_kg": 12000.0,
                "simple_pressure_hold_th": 0.35,
                "simple_pressure_decay_th": 0.70,
            },
            unit="mixed",
            subsystem="trim_supervisor",
            classification="control_supervision_tuning",
            safety_realistic="keep_for_legacy_trim",
            research_minimal="disable_if_prediction_primary_owns_mass_schedule",
            source_refs=["archive/legacy_fowt_control/defaults.py:32-58", "archive/legacy_fowt_control/defaults.py:70-74"],
            rationale="These gates manage old trim behavior under pressure/backlog. A prediction-primary controller should handle these costs internally or bypass trim.",
        ),
        item(
            key="planner_capacity_guard_ratio",
            value={"upper": 0.92, "lower": 0.08},
            unit="tank_capacity_ratio",
            subsystem="forecast_planner",
            classification="soft_safety_margin",
            safety_realistic="keep_soft_or_hard_margin",
            research_minimal="keep_soft_margin",
            source_refs=["src/wind_prediction/ballast_planner.py:37-39", "src/wind_prediction/ballast_planner.py:160-166"],
            rationale="Not a physical bound, but a useful margin away from hard capacity. Keep as soft margin in research.",
        ),
        item(
            key="planner_fullspeed_guard",
            value=True,
            unit="bool",
            subsystem="forecast_planner",
            classification="safety_guard",
            safety_realistic="keep_hard",
            research_minimal="keep_hard",
            source_refs=["src/wind_prediction/ballast_planner.py:39", "src/wind_prediction/ballast_planner.py:165-166"],
            rationale="Avoids adding forecast action while plant already reports full-speed pumping.",
        ),
    ]

    return {
        "schema_version": 1,
        "purpose": "Separate hard constraints from legacy anti-chatter/tuning layers before prediction-primary redesign.",
        "profiles": {
            "safety_realistic": {
                "description": "Conservative mode for validation: keep physical, hardware-like, unknown, and validated hysteresis constraints unless explicitly proven redundant.",
            },
            "research_minimal": {
                "description": "Upper-bound mode for forecast value: bypass reactive tuning layers and keep only physical/safety constraints plus clearly hardware-like actuator limits.",
            },
        },
        "items": items,
    }


def write_report(audit: dict[str, Any], out_dir: Path) -> None:
    items = audit["items"]
    counts: dict[str, int] = {}
    for x in items:
        counts[x["classification"]] = counts.get(x["classification"], 0) + 1

    lines = [
        "# Control Constraints Audit",
        "",
        "Step 0a for prediction-primary redesign. This is an evidence map, not a hardware certification.",
        "",
        "## Summary",
        "",
        f"- audited constraints: `{len(items)}`",
        "- profiles: `safety_realistic`, `research_minimal`",
        "- key finding: the stack mixes physical actuator bounds with legacy anti-chatter/control-tuning layers; prediction-primary should bypass the latter, not blindly delete everything.",
        "",
        "## Classification Counts",
        "",
        "| classification | count |",
        "|---|---:|",
    ]
    for k, v in sorted(counts.items()):
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Profile Guidance",
        "",
        "| subsystem/key | class | safety_realistic | research_minimal |",
        "|---|---|---|---|",
    ]
    for x in items:
        lines.append(
            f"| {x['subsystem']}/{x['key']} | {x['classification']} | "
            f"{x['safety_realistic']} | {x['research_minimal']} |"
        )

    lines += [
        "",
        "## Next Gate",
        "",
        "Do not implement prediction-primary yet. Next step is Step 0b: calibrate a minimal forward model against existing closed_only timeseries. If that model cannot reproduce pump_work and major attitude metrics within 10-15%, prediction-primary should be stopped or reframed as a qualitative upper-bound study.",
    ]
    (out_dir / "control_constraints_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    (OUT_DIR / "control_constraints_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(audit, OUT_DIR)
    print(f"Wrote {OUT_DIR / 'control_constraints_audit.json'}")
    print(f"Wrote {OUT_DIR / 'control_constraints_audit.md'}")


if __name__ == "__main__":
    main()
