"""Forecast-assisted ballast planner provider.

The public provider is assembled from behavior groups so callers retain the
original interface while forecast analysis, planning, target management, and
cycle orchestration remain locally maintainable.
"""

from __future__ import annotations

from .provider_common import complete_event_probabilities_available
from .provider_config import ProviderConfig, ProviderRuntimeInputs
from .provider_economy import ProviderEconomyMixin
from .provider_forecast import ProviderForecastMixin
from .provider_horizon import ProviderHorizonMixin
from .provider_planning import ProviderPlanningMixin
from .provider_runtime import ProviderRuntimeMixin
from .provider_state import ProviderStateMixin
from .provider_target import ProviderTargetMixin
from .provider_telemetry import ProviderTelemetryMixin


class BallastPlannerPreviewProvider(
    ProviderRuntimeMixin,
    ProviderTelemetryMixin,
    ProviderTargetMixin,
    ProviderPlanningMixin,
    ProviderForecastMixin,
    ProviderHorizonMixin,
    ProviderEconomyMixin,
    ProviderStateMixin,
):
    """Convert forecast evidence into a pump-facing ballast target preview."""

    def __init__(
        self,
        runtime: ProviderRuntimeInputs | None = None,
        config: ProviderConfig | None = None,
        *legacy_args,
        **legacy_kwargs,
    ) -> None:
        """Build from grouped config while accepting old research scripts."""

        if isinstance(runtime, ProviderRuntimeInputs):
            if legacy_args or legacy_kwargs:
                raise TypeError(
                    "grouped provider construction cannot be mixed with legacy arguments"
                )
            options = (config or ProviderConfig()).to_legacy_kwargs()
            ProviderStateMixin.__init__(
                self,
                replay_dataset=runtime.replay_dataset,
                start_timestamp=runtime.start_timestamp,
                cfg=runtime.planner,
                block_discounts=list(runtime.block_discounts),
                plant_info=(
                    dict(runtime.plant_info) if runtime.plant_info is not None else None
                ),
                forecast_adapter=runtime.forecast_adapter,
                **options,
            )
            return

        positional = []
        if runtime is not None:
            positional.append(runtime)
        if config is not None:
            positional.append(config)
        positional.extend(legacy_args)
        ProviderStateMixin.__init__(self, *positional, **legacy_kwargs)
