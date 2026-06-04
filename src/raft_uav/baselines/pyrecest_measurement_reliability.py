"""RaFT-UAV adapter for PyRecEst measurement reliability utilities.

RaFT-UAV owns RF/radar-specific reliability *scores*.  PyRecEst owns the generic
conversion from those scores to covariance scaling or hard accept/reject
decisions.  This module keeps the existing RaFT-UAV soft/hard/off vocabulary
while delegating the generic math to :mod:`pyrecest.tracking`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pyrecest.tracking import (
    MeasurementReliabilityConfig,
    MeasurementReliabilityResult,
    apply_measurement_reliability,
    reliability_to_covariance_scale,
    scale_covariance_by_reliability,
)

RaftReliabilityMode = Literal["off", "soft", "hard"] | str


@dataclass(frozen=True)
class RaftMeasurementReliabilityDecision:
    """RaFT-facing reliability decision with legacy field names."""

    accepted: bool
    update_action: str
    covariance_scale: float
    covariance: np.ndarray
    reliability: float

    @classmethod
    def from_pyrecest(
        cls,
        result: MeasurementReliabilityResult,
        *,
        mode: str,
    ) -> "RaftMeasurementReliabilityDecision":
        action = result.action
        if mode == "soft" and result.accepted:
            action = "rf_reliability_scaled" if result.covariance_scale > 1.0 else "rf_reliability_accepted"
        elif mode == "hard":
            action = "rf_reliability_accepted" if result.accepted else "rf_reliability_rejected"
        elif mode == "off":
            action = "rf_reliability_off"
        return cls(
            accepted=bool(result.accepted),
            update_action=action,
            covariance_scale=float(result.covariance_scale),
            covariance=np.asarray(result.covariance, dtype=float),
            reliability=float(result.reliability),
        )


def rf_reliability_covariance_scale(
    reliability: float,
    *,
    min_probability: float = 0.05,
    exponent: float = 1.0,
    max_scale: float | None = None,
) -> float:
    """Return PyRecEst's covariance scale for a RaFT-UAV RF reliability score."""

    return reliability_to_covariance_scale(
        reliability,
        floor=min_probability,
        exponent=exponent,
        max_scale=max_scale,
    )


def scale_rf_covariance_by_reliability(
    covariance: np.ndarray,
    reliability: float,
    *,
    min_probability: float = 0.05,
    exponent: float = 1.0,
    max_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return RF covariance inflated by inverse reliability."""

    return scale_covariance_by_reliability(
        covariance,
        reliability,
        floor=min_probability,
        exponent=exponent,
        max_scale=max_scale,
    )


def apply_raft_measurement_reliability(
    covariance: np.ndarray,
    reliability: float,
    *,
    mode: RaftReliabilityMode = "off",
    threshold: float = 0.5,
    min_probability: float = 0.05,
    exponent: float = 1.0,
    max_scale: float | None = None,
) -> RaftMeasurementReliabilityDecision:
    """Apply RaFT-UAV RF reliability mode using PyRecEst primitives.

    Modes map to PyRecEst as follows:

    ``off``
        Accept and keep the nominal covariance.
    ``soft``
        Accept and inflate covariance by inverse reliability.
    ``hard``
        Reject rows below ``threshold`` and keep accepted-row covariance nominal.
    """

    mode = str(mode)
    if mode not in {"off", "soft", "hard"}:
        raise ValueError("mode must be one of 'off', 'soft', or 'hard'")
    if mode == "off":
        config = MeasurementReliabilityConfig(mode="off")
    elif mode == "soft":
        config = MeasurementReliabilityConfig(
            mode="inflate",
            threshold=None,
            floor=min_probability,
            exponent=exponent,
            max_scale=max_scale,
        )
    else:
        config = MeasurementReliabilityConfig(
            mode="hard",
            threshold=threshold,
            floor=min_probability,
            exponent=exponent,
            max_scale=max_scale,
        )
    result = apply_measurement_reliability(
        covariance,
        reliability=reliability,
        config=config,
    )
    return RaftMeasurementReliabilityDecision.from_pyrecest(result, mode=mode)


__all__ = [
    "RaftMeasurementReliabilityDecision",
    "apply_raft_measurement_reliability",
    "rf_reliability_covariance_scale",
    "scale_rf_covariance_by_reliability",
]
