from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyrecest.filters.linear_update_planning import (
    plan_linear_measurement_update as pyrecest_plan_linear_measurement_update,
)

from raft_uav.baselines import update_logic
from raft_uav.baselines.update_logic import (
    gate_threshold_for_measurement,
    huber_covariance_scale,
    inflation_alpha_for_measurement,
    max_residual_norm_for_measurement,
    normalized_innovation_squared,
    plan_linear_measurement_update,
    robust_update_covariance_scale,
    robust_update_for_measurement,
    student_t_covariance_scale,
)


@dataclass(frozen=True)
class _Measurement:
    source: str
    vector: np.ndarray


def _linear_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.array([0.0, 0.0])
    covariance = np.eye(2)
    measurement = np.array([10.0, 0.0])
    measurement_covariance = np.eye(2)
    observation = np.eye(2)
    return mean, covariance, measurement, measurement_covariance, observation


def test_update_logic_public_plan_is_pyrecest_backed() -> None:
    assert update_logic.plan_linear_measurement_update.__module__.endswith(
        "pyrecest_update_logic"
    )


def test_normalized_innovation_squared_matches_pyrecest_plan() -> None:
    mean, covariance, measurement, measurement_covariance, observation = _linear_case()
    residual = measurement - observation @ mean
    innovation_covariance = observation @ covariance @ observation.T + measurement_covariance

    raft_nis = normalized_innovation_squared(residual, innovation_covariance)
    pyrecest_plan = pyrecest_plan_linear_measurement_update(
        mean=mean,
        covariance_matrix=covariance,
        measurement_vector=measurement,
        measurement_covariance=measurement_covariance,
        observation_matrix=observation,
    )

    assert np.isclose(raft_nis, pyrecest_plan.nis)


def test_nis_inflate_plan_delegates_scale_and_action_to_pyrecest() -> None:
    mean, covariance, measurement, measurement_covariance, observation = _linear_case()

    raft_plan = plan_linear_measurement_update(
        mean=mean,
        covariance_matrix=covariance,
        measurement_vector=measurement,
        measurement_covariance=measurement_covariance,
        observation_matrix=observation,
        gate_threshold=5.0,
        robust_update="nis-inflate",
        inflation_alpha=1.0,
    )
    pyrecest_plan = pyrecest_plan_linear_measurement_update(
        mean=mean,
        covariance_matrix=covariance,
        measurement_vector=measurement,
        measurement_covariance=measurement_covariance,
        observation_matrix=observation,
        gate_threshold=5.0,
        robust_update="nis-inflate",
        inflation_alpha=1.0,
    )

    assert raft_plan.accepted is True
    assert raft_plan.update_action == pyrecest_plan.action == "inflated"
    assert np.isclose(raft_plan.covariance_scale, pyrecest_plan.covariance_scale)
    assert np.allclose(raft_plan.covariance, pyrecest_plan.covariance)


def test_plain_gate_rejection_uses_pyrecest_decision() -> None:
    mean, covariance, measurement, measurement_covariance, observation = _linear_case()

    plan = plan_linear_measurement_update(
        mean=mean,
        covariance_matrix=covariance,
        measurement_vector=measurement,
        measurement_covariance=measurement_covariance,
        observation_matrix=observation,
        gate_threshold=5.0,
        robust_update=None,
    )

    assert plan.accepted is False
    assert plan.update_action == "rejected"
    assert plan.nis > 5.0
    assert np.allclose(plan.covariance, measurement_covariance)


def test_safety_gate_action_is_mapped_to_raft_missed_detection() -> None:
    mean, covariance, measurement, measurement_covariance, observation = _linear_case()

    plan = plan_linear_measurement_update(
        mean=mean,
        covariance_matrix=covariance,
        measurement_vector=measurement,
        measurement_covariance=measurement_covariance,
        observation_matrix=observation,
        robust_update="nis-inflate",
        gate_threshold=100.0,
        safety_gate_threshold=5.0,
    )

    assert plan.accepted is False
    assert plan.update_action == "missed_detection"


def test_scale_helpers_are_pyrecest_compatible() -> None:
    student_scale, student_action = robust_update_covariance_scale(
        "student-t",
        nis=50.0,
        measurement_dim=2,
        gate_threshold=None,
        student_t_dof=4.0,
    )
    huber_scale, huber_action = robust_update_covariance_scale(
        "huber",
        nis=50.0,
        measurement_dim=2,
        gate_threshold=None,
        huber_threshold=2.0,
    )

    assert student_scale == student_t_covariance_scale(50.0, 2, 4.0)
    assert student_action == "student_t"
    assert huber_scale == huber_covariance_scale(50.0, 2.0)
    assert huber_action == "huberized"


def test_source_specific_policy_helpers_stay_in_raft() -> None:
    measurement = _Measurement(source="radar", vector=np.zeros(3))

    assert (
        robust_update_for_measurement(
            measurement,
            robust_update_by_source={"radar": "student-t"},
        )
        == "student-t"
    )
    assert inflation_alpha_for_measurement(
        measurement,
        inflation_alpha_by_source={"radar": 1.5},
    ) == 1.5
    assert max_residual_norm_for_measurement(
        measurement,
        max_residual_norms_by_source={"radar": 250.0},
    ) == 250.0
    assert gate_threshold_for_measurement(
        measurement,
        gate_probabilities_by_source=None,
        gate_thresholds_by_source={"radar": 7.0},
        probability_to_threshold=lambda *_args: 0.0,
    ) == 7.0
