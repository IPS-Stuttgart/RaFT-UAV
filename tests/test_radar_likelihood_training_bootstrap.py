import numpy as np
import pandas as pd

from raft_uav.baselines import radar_likelihood_training as training
from raft_uav.baselines.kalman import TrackingMeasurement


def test_training_collector_does_not_replay_bootstrap_event(monkeypatch):
    tracker_instances = []
    implementation = training._IMPL

    class FakeTracker:
        def __init__(
            self,
            *,
            initial_position: np.ndarray,
            initial_time_s: float,
            acceleration_std_mps2: float,
        ) -> None:
            del initial_position, initial_time_s, acceleration_std_mps2
            self.state = np.zeros(6)
            self.update_times: list[float] = []
            tracker_instances.append(self)

        def update(self, measurement: TrackingMeasurement) -> None:
            self.update_times.append(float(measurement.time_s))

        def predict_to(self, time_s: float) -> None:
            del time_s

    initial_measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([0.0, 0.0]),
        covariance=np.eye(2),
        source="rf",
    )
    candidates = pd.DataFrame(
        [
            {
                "frame_index": 1,
                "track_id": 7,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
            }
        ]
    )
    events = [
        {"kind": "rf", "time_s": 0.0, "measurement": initial_measurement},
        {"kind": "radar", "time_s": 1.0, "candidates": candidates},
    ]

    monkeypatch.setattr(implementation, "AsyncConstantVelocityKalmanTracker", FakeTracker)
    monkeypatch.setattr(implementation, "_events", lambda rf_measurements, radar: events)
    monkeypatch.setattr(
        implementation,
        "_initial_measurement",
        lambda *args, **kwargs: initial_measurement,
    )
    monkeypatch.setattr(implementation, "_catprob_candidate_pool", lambda frame, threshold: frame)
    monkeypatch.setattr(
        implementation,
        "_nis_scored_candidates",
        lambda frame, tracker, covariance: frame.assign(association_nis=1.0),
    )
    monkeypatch.setattr(
        implementation,
        "_nearest_truth_position",
        lambda *args, **kwargs: np.zeros(3),
    )
    monkeypatch.setattr(
        implementation,
        "_candidate_truth_errors",
        lambda frame, truth_xyz: np.zeros(len(frame)),
    )
    monkeypatch.setattr(
        implementation,
        "radar_association_feature_frame",
        lambda frame, **kwargs: pd.DataFrame({"dummy_feature": np.ones(len(frame))}),
    )

    examples = training.collect_radar_association_training_frame(
        rf_measurements=[],
        radar=candidates,
        truth=pd.DataFrame(
            {"time_s": [1.0], "east_m": [1.0], "north_m": [0.0], "up_m": [0.0]}
        ),
        candidate_catprob_threshold=None,
        teacher_association="none",
    )

    assert tracker_instances[0].update_times == []
    assert examples["time_s"].tolist() == [1.0]
