"""Grouped configuration for the forecast-assisted ballast provider."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from .ballast_planner import PlannerConfig
from .forecast_adapter import ForecastModelAdapter
from .replay_dataset import Fino1ReplayDataset


@dataclass(frozen=True)
class ProviderRuntimeInputs:
    """Objects that vary with a run or case rather than controller identity."""

    replay_dataset: Fino1ReplayDataset
    start_timestamp: datetime
    planner: PlannerConfig
    block_discounts: tuple[float, ...]
    plant_info: Mapping[str, Any] | None = None
    forecast_adapter: ForecastModelAdapter | None = None


@dataclass(frozen=True)
class ProviderConfig:
    """Controller options grouped by their role in the control chain."""

    planning: Mapping[str, Any] = field(default_factory=dict)
    forecast: Mapping[str, Any] = field(default_factory=dict)
    target: Mapping[str, Any] = field(default_factory=dict)
    safety: Mapping[str, Any] = field(default_factory=dict)
    experimental: Mapping[str, Any] = field(default_factory=dict)

    def to_legacy_kwargs(self) -> dict[str, Any]:
        """Flatten sections for the behavior-preserving legacy initializer."""

        merged: dict[str, Any] = {}
        for section_name, section in self.sections().items():
            for name, value in section.items():
                if name in merged:
                    raise ValueError(
                        f"provider option {name!r} appears in multiple sections; "
                        f"duplicate found in {section_name}"
                    )
                merged[str(name)] = value
        return merged

    def sections(self) -> dict[str, Mapping[str, Any]]:
        return {
            "planning": self.planning,
            "forecast": self.forecast,
            "target": self.target,
            "safety": self.safety,
            "experimental": self.experimental,
        }

    def fingerprint(self) -> str:
        """Stable identity for primitive-valued experiment configurations."""

        payload = json.dumps(
            {name: dict(values) for name, values in self.sections().items()},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
