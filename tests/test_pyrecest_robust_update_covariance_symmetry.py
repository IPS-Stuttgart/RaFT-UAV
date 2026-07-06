import numpy as np

from raft_uav.baselines.pyrecest_robust_update import plan_linear_measurement_update


def test_plan_linear_measurement_update_symmetrizes_covariances():
    plan = plan_linear_measurement_update(
        mean=np.zeros(2),
        covariance_matrix=np.array([[2.0, 0.3], [0.3000000001, 1.0]]),
        measurement_vector=np.array([1.0]),
        measurement_covariance=np.array([[4.0]]),
        observation_matrix=np.array([[1.0, 0.0]]),
    )

    assert np.allclose(plan.covariance, plan.covariance.T)
    assert np.allclose(plan.innovation_covariance, plan.innovation_covariance.T)


def test_plan_linear_measurement_update_removes_input_measurement_covariance_asymmetry():
    plan = plan_linear_measurement_update(
        mean=np.zeros(2),
        covariance_matrix=np.eye(2),
        measurement_vector=np.array([1.0, -1.0]),
        measurement_covariance=np.array([[3.0, 0.25], [0.2500000001, 4.0]]),
        observation_matrix=np.eye(2),
    )

    assert np.allclose(plan.covariance, plan.covariance.T)
    assert np.allclose(plan.innovation_covariance, plan.innovation_covariance.T)
