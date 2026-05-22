"""Adaptive process-noise heuristics from innovation statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Mapping

from pyrecest.filters.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig as _PyRecEstAdaptiveProcessNoiseConfig,
    RollingNISProcessNoiseAdapter,
    adaptive_scale_from_ratio as _pyrecest_adaptive_scale_from_ratio,
)

ENV_ADAPTIVE_PROCESS_NOISE = "RAFT_UAV_ADAPTIVE_PROCESS_NOISE"
ENV_ADAPTIVE_PROCESS_NOISE_MIN_SCALE = "RAFT_UAV_ADAPTIVE_PROCESS_NOISE_MIN_SCALE"
ENV_ADAPTIVE_PROCESS_NOISE_MAX_SCALE = "RAFT_UAV_ADAPTIVE_PROCESS_NOISE_MAX_SCALE"
ENV_ADAPTIVE_PROCESS_NOISE_EWMA_ALPHA = "RAFT_UAV_ADAPTIVE_PROCESS_NOISE_EWMA_ALPHA"
ENV_ADAPTIVE_PROCESS_NOISE_HIGH_NIS_RATIO = (
    "RAFT_UAV_ADAPTIVE_PROCESS_NOISE_HIGH_NIS_RATIO"
)
ENV_ADAPTIVE_PROCESS_NOISE_LOW_NIS_RATIO = (
    "RAFT_UAV_ADAPTIVE_PROCESS_NOISE_LOW_NIS_RATIO"
)
ENV_ADAPTIVE_PROCESS_NOISE_SCALE_GAIN = "RAFT_UAV_ADAPTIVE_PROCESS_NOISE_SCALE_GAIN"


@dataclass(frozen=True)
class AdaptiveProcessNoiseConfig:
    """Parameters for NIS-driven acceleration-noise adaptation."""

    base_acceleration_std_mps2: float = 4.0
    min_scale: float = 0.35
    max_scale: float = 4.0
    ewma_alpha: float = 0.05
    high_nis_ratio: float = 1.5
    low_nis_ratio: float = 0.6
    scale_gain: float = 0.5

    def __post_init__(self) -> None:
        if self.base_acceleration_std_mps2 <= 0.0:
            raise ValueError("base_acceleration_std_mps2 must be positive")
        if self.min_scale <= 0.0 or self.max_scale < self.min_scale:
            raise ValueError("scale bounds must be positive and ordered")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")


@dataclass
class RollingNISAdaptiveAcceleration:
    """Maintain per-source EWMA NIS ratios and return an acceleration scale."""

    config: AdaptiveProcessNoiseConfig = field(default_factory=AdaptiveProcessNoiseConfig)
    adapter: RollingNISProcessNoiseAdapter = field(init=False)

    def __post_init__(self) -> None:
        self.adapter = RollingNISProcessNoiseAdapter(
            _PyRecEstAdaptiveProcessNoiseConfig(
                base_scale=1.0,
                min_scale=self.config.min_scale,
                max_scale=self.config.max_scale,
                ewma_alpha=self.config.ewma_alpha,
                high_nis_ratio=self.config.high_nis_ratio,
                low_nis_ratio=self.config.low_nis_ratio,
                scale_gain=self.config.scale_gain,
            )
        )

    @property
    def ratios_by_source(self) -> dict[str, float]:
        """Expose the upstream adapter state under the historical RaFT-UAV name."""

        return self.adapter.ratios_by_source

    @property
    def updates_by_source(self) -> dict[str, int]:
        """Expose the upstream adapter counts under the historical RaFT-UAV name."""

        return self.adapter.updates_by_source

    def observe(self, *, source: str, measurement_dim: int, nis: float, accepted: bool = True) -> float:
        """Ingest one innovation and return the updated source ratio."""

        return self.adapter.observe(
            source=source,
            measurement_dim=measurement_dim,
            nis=nis,
            accepted=accepted,
        )

    def acceleration_std_mps2(self, source_weights: Mapping[str, float] | None = None) -> float:
        """Return the adapted acceleration standard deviation."""

        return float(self.config.base_acceleration_std_mps2 * self.adapter.scale(source_weights))


def adaptive_scale_from_ratio(ratio: float, config: AdaptiveProcessNoiseConfig) -> float:
    """Map a normalized NIS ratio to a bounded process-noise scale."""

    upstream_config = _PyRecEstAdaptiveProcessNoiseConfig(
        base_scale=1.0,
        min_scale=config.min_scale,
        max_scale=config.max_scale,
        ewma_alpha=config.ewma_alpha,
        high_nis_ratio=config.high_nis_ratio,
        low_nis_ratio=config.low_nis_ratio,
        scale_gain=config.scale_gain,
    )
    return _pyrecest_adaptive_scale_from_ratio(ratio, upstream_config)


def adaptive_process_noise_from_environment(
    *,
    base_acceleration_std_mps2: float,
) -> RollingNISAdaptiveAcceleration | None:
    """Return an online process-noise adapter when the env flag is enabled.

    The adapter is deliberately environment-driven so existing CLIs and scripts
    remain bit-for-bit unchanged unless experiments opt in with
    ``RAFT_UAV_ADAPTIVE_PROCESS_NOISE=1``.  All values are leakage-safe because
    they are derived from the tracker's own accepted innovations.
    """

    if not _env_flag(ENV_ADAPTIVE_PROCESS_NOISE):
        return None
    return RollingNISAdaptiveAcceleration(
        AdaptiveProcessNoiseConfig(
            base_acceleration_std_mps2=float(base_acceleration_std_mps2),
            min_scale=_env_float(ENV_ADAPTIVE_PROCESS_NOISE_MIN_SCALE, 0.35),
            max_scale=_env_float(ENV_ADAPTIVE_PROCESS_NOISE_MAX_SCALE, 4.0),
            ewma_alpha=_env_float(ENV_ADAPTIVE_PROCESS_NOISE_EWMA_ALPHA, 0.05),
            high_nis_ratio=_env_float(ENV_ADAPTIVE_PROCESS_NOISE_HIGH_NIS_RATIO, 1.5),
            low_nis_ratio=_env_float(ENV_ADAPTIVE_PROCESS_NOISE_LOW_NIS_RATIO, 0.6),
            scale_gain=_env_float(ENV_ADAPTIVE_PROCESS_NOISE_SCALE_GAIN, 0.5),
        )
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)
