import numpy as np

from raft_uav.baselines import radar_association as assoc
from raft_uav.baselines.kalman import TrackingMeasurement


def test_track_bank_bootstrap_preserves_position_velocity_state(monkeypatch):
    vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    measurement = TrackingMeasurement(
        time_s=10.0,
        vector=vector,
        covariance=np.eye(6),
        source="radar",
    )
    captured: dict[str, object] = {}

    def fake_kalman_filter(initial_state):
        captured["initial_state"] = initial_state
        return object()

    class FakeMultiHypothesisTracker:
        def __init__(self, *, initial_prior, association_param, **kwargs):
            captured["initial_prior"] = initial_prior
            captured["association_param"] = association_param
            captured["tracker_kwargs"] = kwargs

    monkeypatch.setattr(assoc, "KalmanFilter", fake_kalman_filter)
    monkeypatch.setattr(
        assoc,
        "MultiHypothesisTracker",
        FakeMultiHypothesisTracker,
    )

    assoc._initial_mht_tracker(
        measurement,
        max_global_hypotheses=4,
        max_assignments_per_hypothesis=5,
        max_candidates_per_track=6,
        gate_probability=0.99,
        detection_probability=0.9,
        clutter_intensity=1.0e-6,
        prune_log_weight_delta=20.0,
    )

    initial_mean, initial_covariance = captured["initial_state"]
    np.testing.assert_allclose(initial_mean, vector)
    np.testing.assert_allclose(
        initial_covariance,
        np.diag([50.0**2, 50.0**2, 50.0**2, 15.0**2, 15.0**2, 15.0**2]),
    )
    assert captured["association_param"] == {
        "gating_probability": 0.99,
        "detection_probability": 0.9,
        "clutter_intensity": 1.0e-6,
        "max_global_hypotheses": 4,
        "max_hypotheses_per_global_hypothesis": 5,
        "max_measurements_per_track": 6,
        "prune_log_weight_delta": 20.0,
    }
    assert captured["tracker_kwargs"] == {
        "log_prior_estimates": False,
        "log_posterior_estimates": False,
    }
