"""Control-facing contracts for forecast-assisted ballast control.

These dataclasses are intentionally behavior-neutral. They define the target
interfaces for future cleanup of the prediction-primary chain without changing
the existing provider or closed-loop runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


Authority = str
ControlEffect = str


@dataclass(frozen=True)
class ForecastSignal:
    """Forecast evidence after conversion into control-facing semantics."""

    source: str
    model_version: str
    horizon_minutes: float
    sample_period_s: float
    staleness_s: float
    is_fresh: bool
    confidence: float
    persistence_agreement_norm: float
    current_support_norm: float
    event_support: Mapping[str, float] = field(default_factory=dict)
    allowed_effects: tuple[ControlEffect, ...] = ()
    fail_closed_action: str = "reactive_only"
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionTraceStep:
    """One explainable step in the planner-to-command decision path."""

    name: str
    input_action: str
    output_action: str
    changed: bool
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupervisorDecision:
    """Safety and authority decision between reactive and forecast-assisted paths."""

    authority: Authority
    accepted: bool
    requested_action: str
    accepted_action: str
    veto_reason: str = ""
    safety_state: str = "nominal"
    forecast_signal_reason: str = ""
    decision_trace: tuple[DecisionTraceStep, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlCommand:
    """Final command surface exposed to the closed-loop policy."""

    authority: Authority
    source: str
    pitch_bias_deg: float = 0.0
    roll_bias_deg: float = 0.0
    target_masses_kg: tuple[float, ...] | None = None
    mass_delta_kg: tuple[float, ...] | None = None
    reason: str = ""
    telemetry: Mapping[str, Any] = field(default_factory=dict)


def validate_forecast_signal(signal: ForecastSignal) -> ForecastSignal:
    """Validate control semantics that numeric forecast contracts cannot cover."""

    if signal.horizon_minutes <= 0.0:
        raise ValueError("ForecastSignal.horizon_minutes must be positive")
    if signal.sample_period_s <= 0.0:
        raise ValueError("ForecastSignal.sample_period_s must be positive")
    if signal.staleness_s < 0.0:
        raise ValueError("ForecastSignal.staleness_s must be non-negative")
    if not 0.0 <= signal.confidence <= 1.0:
        raise ValueError("ForecastSignal.confidence must be in [0, 1]")
    if signal.persistence_agreement_norm < 0.0:
        raise ValueError("ForecastSignal.persistence_agreement_norm must be non-negative")
    if signal.current_support_norm < 0.0:
        raise ValueError("ForecastSignal.current_support_norm must be non-negative")
    if signal.fail_closed_action == "":
        raise ValueError("ForecastSignal.fail_closed_action must be non-empty")
    return signal


def validate_control_command(command: ControlCommand) -> ControlCommand:
    """Validate the command shape before it reaches actuator-facing code."""

    if command.authority == "":
        raise ValueError("ControlCommand.authority must be non-empty")
    if command.source == "":
        raise ValueError("ControlCommand.source must be non-empty")
    if command.target_masses_kg is not None and len(command.target_masses_kg) == 0:
        raise ValueError("ControlCommand.target_masses_kg cannot be empty")
    if command.mass_delta_kg is not None and len(command.mass_delta_kg) == 0:
        raise ValueError("ControlCommand.mass_delta_kg cannot be empty")
    return command
