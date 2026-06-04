"""Compatibility exports for RaFT-UAV linear-update planning utilities."""

from __future__ import annotations

from raft_uav.baselines.pyrecest_update_logic import (
    DEFAULT_HUBER_THRESHOLD,
    DEFAULT_STUDENT_T_DOF,
    ROBUST_UPDATE_MODES,
    LinearUpdatePlan,
    TrackingMeasurementLike,
    gate_threshold_for_measurement,
    huber_covariance_scale,
    huber_threshold_for_measurement,
    inflation_alpha_for_measurement,
    max_residual_norm_for_measurement,
    normalized_innovation_squared,
    plan_linear_measurement_update,
    robust_update_covariance_scale,
    robust_update_for_measurement,
    student_t_covariance_scale,
    student_t_dof_for_measurement,
    symmetrized,
)

__all__ = [
    "DEFAULT_HUBER_THRESHOLD",
    "DEFAULT_STUDENT_T_DOF",
    "ROBUST_UPDATE_MODES",
    "LinearUpdatePlan",
    "TrackingMeasurementLike",
    "gate_threshold_for_measurement",
    "huber_covariance_scale",
    "huber_threshold_for_measurement",
    "inflation_alpha_for_measurement",
    "max_residual_norm_for_measurement",
    "normalized_innovation_squared",
    "plan_linear_measurement_update",
    "robust_update_covariance_scale",
    "robust_update_for_measurement",
    "student_t_covariance_scale",
    "student_t_dof_for_measurement",
    "symmetrized",
]
