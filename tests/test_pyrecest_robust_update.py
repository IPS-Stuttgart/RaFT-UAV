from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from raft_uav.baselines.pyrecest_robust_update import (
    gate_threshold_for_measurement,
    huber_covariance_scale,
    inflation_alpha_for_measurement,
    normalized_innovation_squared,
    plan_linear_measurement_update,
    robust_update_for_measurement,
    student_t_covariance_scale,
)


@dataclass(frozen=True)
class DummyMeasurement:
    source: str
    vector: np.ndarray


def test_normalized_innovation_squared_delegates_to_pyrecest_math() -> None:
    residual = np.array([2.0, 0.0])
    innovation_covariance = np.diag([4.0, 1.0])

    assert np.isclose(normalized_innovation_squared(residual, innovation_covariance), 1.0)


def test_pyrecest_student_t_and_huber_scales_are_inflating_for_outliers() -> None:
    assert student_t_covariance_scale(25.0, measurement_dim=2, degrees_of_freedom=4.0) > 1.0
    assert huber_covariance_scale(25.0, threshold=2.0) > 1.0


def test_plan_linear_measurement_update_rejects_hard_gated_outlier() -> None:
    mean = np.zeros(2)
    covariance = np.eye(2)
    measurement = np.array([10.0, 0.0])
    measurement_covariance = np.eye(2)
    observation = np.eye(2)

    plan = plan_linear_measurement_update(
        mean=mean,
        covariance_matrix=covariance,
        measurement_vector=measurement,
        measurement_covariance=measurement_covariance,
        observation_matrix=observation,
        gate_threshold=5.991,
        robust_update=None,
    )

    assert not plan.accepted
    assert plan.update_action == "rejected"
    assert plan.nis > 5.991
    assert np.allclose(plan.covariance, measurement_covariance)


def test_plan_linear_measurement_update_inflates_nis_outlier_without_rejection() -> None:
    mean = np.zeros(2)
    covariance = np.eye(2)
    measurement = np.array([10.0, 0.0])
    measurement_covariance = np.eye(2)
    observation = np.eye(2)

    plan = plan_linear_measurement_update(
        mean=mean,
        covariance_matrix=covariance,
        measurement_vector=measurement,
        measurement_covariance=measurement_covariance,
        observation_matrix=observation,
        gate_threshold=5.991,
        robust_update="nis-inflate",
        inflation_alpha=1.0,
    )

    assert plan.accepted
    assert plan.update_action == "inflated"
    assert plan.covariance_scale > 1.0
    assert np.allclose(plan.covariance, measurement_covariance * plan.covariance_scale)


def test_source_specific_policy_helpers_remain_raft_compatible() -> None:
    measurement = DummyMeasurement(source="radar", vector=np.zeros(3))

    threshold = gate_threshold_for_measurement(
        measurement,
        gate_probabilities_by_source={"radar": 0.95},
        gate_thresholds_by_source=None,
        probability_to_threshold=lambda probability, dim: probability + dim,
    )

    assert threshold == 3.95
    assert robust_update_for_measurement(
        measurement,
        robust_update_by_source={"radar": "student-t"},
    ) == "student-t"
    assert inflation_alpha_for_measurement(
        measurement,
        inflation_alpha_by_source={"radar": 1.5},
    ) == 1.5
