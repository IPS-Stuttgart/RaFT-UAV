"""RaFT-UAV adapter around PyRecEst robust linear update planning.

PyRecEst owns the generic robust linear-Gaussian update policy: NIS
calculation, hard NIS rejection, safety/residual gates, and robust covariance
inflation modes such as ``nis-inflate``, ``student-t``, and ``huber``.  This
module keeps RaFT-UAV's public diagnostics schema and source-specific option
lookup while delegating the estimation math to PyRecEst.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pyrecest.filters.linear_update_planning import (
    DEFAULT_HUBER_THRESHOLD,
    DEFAULT_STUDENT_T_DOF,
    huber_covariance_scale as _pyrecest_huber_covariance_scale,
    normalized_innovation_squared as _pyrecest_normalized_innovation_squared,
    plan_linear_measurement_update as _pyrecest_plan_linear_measurement_update,
    robust_update_covariance_scale as _pyrecest_robust_update_covariance_scale,
    robust_update_for_measurement as _pyrecest_robust_update_for_measurement,
    source_float_value as _pyrecest_source_float_value,
    student_t_covariance_scale as _pyrecest_student_t_covariance_scale,
)


class TrackingMeasurementLike(Protocol):
    """Protocol for measurement objects used by baseline trackers."""

    source: str
    vector: np.ndarray


@dataclass(frozen=True)
class LinearUpdatePlan:
    """RaFT-facing robust update plan produced by PyRecEst."""

    vector: np.ndarray
    covariance: np.ndarray
    observation: np.ndarray
    residual: np.ndarray
    innovation_covariance: np.ndarray
    nis: float
    residual_norm: float
    threshold: float | None
    safety_threshold: float | None
    residual_threshold: float | None
    covariance_scale: float
    update_action: str
    accepted: bool
    inflation_alpha: float


def _backend_scalar_to_float(value: object) -> float:
    """Convert a backend scalar to a Python float."""

    return float(np.asarray(value, dtype=float))


def normalized_innovation_squared(
    residual: np.ndarray,
    innovation_covariance: np.ndarray,
) -> float:
    """Return the squared Mahalanobis innovation distance via PyRecEst."""

    return _backend_scalar_to_float(
        _pyrecest_normalized_innovation_squared(residual, innovation_covariance)
    )


def student_t_covariance_scale(
    nis: float,
    measurement_dim: int,
    degrees_of_freedom: float = DEFAULT_STUDENT_T_DOF,
) -> float:
    """Return PyRecEst's Student-t robust covariance inflation factor."""

    return _backend_scalar_to_float(
        _pyrecest_student_t_covariance_scale(
            nis,
            measurement_dim,
            degrees_of_freedom=degrees_of_freedom,
        )
    )


def huber_covariance_scale(
    nis: float,
    threshold: float = DEFAULT_HUBER_THRESHOLD,
) -> float:
    """Return PyRecEst's multivariate Huber covariance inflation factor."""

    return _backend_scalar_to_float(
        _pyrecest_huber_covariance_scale(nis, threshold=threshold)
    )


def robust_update_covariance_scale(
    robust_update: str | None,
    *,
    nis: float,
    measurement_dim: int,
    gate_threshold: float | None,
    inflation_alpha: float = 1.0,
    student_t_dof: float = DEFAULT_STUDENT_T_DOF,
    huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
) -> tuple[float, str | None]:
    """Return covariance scale and diagnostic action for a robust update mode."""

    return _pyrecest_robust_update_covariance_scale(
        robust_update,
        nis=nis,
        measurement_dim=measurement_dim,
        gate_threshold=gate_threshold,
        inflation_alpha=inflation_alpha,
        student_t_dof=student_t_dof,
        huber_threshold=huber_threshold,
    )


def _raft_update_action(action: str) -> str:
    """Map generic PyRecEst planning actions into RaFT-UAV record labels."""

    if action in {"residual_rejected", "safety_rejected"}:
        return "missed_detection"
    return action


def _as_symmetric_float_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a float covariance matrix with numerical asymmetry removed."""

    return symmetrized(np.asarray(matrix, dtype=float))


def plan_linear_measurement_update(
    *,
    mean: np.ndarray,
    covariance_matrix: np.ndarray,
    measurement_vector: np.ndarray,
    measurement_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    gate_threshold: float | None = None,
    safety_gate_threshold: float | None = None,
    max_residual_norm: float | None = None,
    robust_update: str | None = None,
    inflation_alpha: float = 1.0,
    student_t_dof: float = DEFAULT_STUDENT_T_DOF,
    huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
) -> LinearUpdatePlan:
    """Plan one robust linear update with PyRecEst and RaFT diagnostics names."""

    plan = _pyrecest_plan_linear_measurement_update(
        mean=mean,
        covariance_matrix=_as_symmetric_float_matrix(covariance_matrix),
        measurement_vector=measurement_vector,
        measurement_covariance=_as_symmetric_float_matrix(measurement_covariance),
        observation_matrix=observation_matrix,
        gate_threshold=gate_threshold,
        safety_gate_threshold=safety_gate_threshold,
        max_residual_norm=max_residual_norm,
        robust_update=robust_update,
        inflation_alpha=inflation_alpha,
        student_t_dof=student_t_dof,
        huber_threshold=huber_threshold,
    )
    return LinearUpdatePlan(
        vector=np.asarray(plan.vector, dtype=float),
        covariance=_as_symmetric_float_matrix(plan.covariance),
        observation=np.asarray(plan.observation, dtype=float),
        residual=np.asarray(plan.residual, dtype=float),
        innovation_covariance=_as_symmetric_float_matrix(plan.innovation_covariance),
        nis=float(plan.nis),
        residual_norm=float(plan.residual_norm),
        threshold=plan.gate_threshold,
        safety_threshold=plan.safety_gate_threshold,
        residual_threshold=plan.residual_threshold,
        covariance_scale=float(plan.covariance_scale),
        update_action=_raft_update_action(str(plan.action)),
        accepted=bool(plan.accepted),
        inflation_alpha=float(plan.inflation_alpha),
    )


def gate_threshold_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    probability_to_threshold,
) -> float | None:
    """Resolve a source-specific NIS threshold for one measurement."""

    if gate_thresholds_by_source and measurement.source in gate_thresholds_by_source:
        threshold = gate_thresholds_by_source[measurement.source]
        return None if threshold is None else float(threshold)
    if gate_probabilities_by_source and measurement.source in gate_probabilities_by_source:
        return probability_to_threshold(
            gate_probabilities_by_source[measurement.source],
            measurement.vector.size,
        )
    return None


def robust_update_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    robust_update_by_source: Mapping[str, str | None] | None,
) -> str | None:
    """Resolve a source-specific robust update mode for one measurement."""

    return _pyrecest_robust_update_for_measurement(
        measurement,
        robust_update_by_source=robust_update_by_source,
    )


def inflation_alpha_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    inflation_alpha_by_source: Mapping[str, float] | None,
) -> float:
    """Resolve a source-specific NIS-inflation exponent for one measurement."""

    value = _pyrecest_source_float_value(
        measurement,
        inflation_alpha_by_source,
        default=1.0,
    )
    assert value is not None
    return float(value)


def max_residual_norm_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
) -> float | None:
    """Resolve a source-specific Euclidean residual cap for one measurement."""

    return _pyrecest_source_float_value(
        measurement,
        max_residual_norms_by_source,
        default=None,
    )


def student_t_dof_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    student_t_dof_by_source: Mapping[str, float] | None,
) -> float:
    """Resolve a source-specific Student-t degrees-of-freedom value."""

    return float(
        _pyrecest_source_float_value(
            measurement,
            student_t_dof_by_source,
            default=DEFAULT_STUDENT_T_DOF,
        )
    )


def huber_threshold_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    huber_threshold_by_source: Mapping[str, float] | None,
) -> float:
    """Resolve a source-specific Huber innovation-radius threshold."""

    return float(
        _pyrecest_source_float_value(
            measurement,
            huber_threshold_by_source,
            default=DEFAULT_HUBER_THRESHOLD,
        )
    )


def symmetrized(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    return 0.5 * (matrix + matrix.T)
