import numpy as np
import pandas as pd

from raft_uav.baselines import radar_association as assoc
from raft_uav.baselines.kalman import TrackingMeasurement


def _rf_measurement(time_s: float) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([1.0, 2.0, 3.0]),
        covariance=np.eye(3),
        source="rf",
    )


def _empty_radar() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "time_s",
            "frame_index",
            "track_id",
            "east_m",
            "north_m",
            "up_m",
            "cat_prob_uav",
        ]
    )


def test_track_bank_does_not_replay_bootstrap_event(monkeypatch):
    first = _rf_measurement(10.0)
    second = _rf_measurement(11.0)
    events = [
        {"time_s": first.time_s, "priority": 0, "kind": "rf", "measurement": first},
        {"time_s": second.time_s, "priority": 0, "kind": "rf", "measurement": second},
    ]
    initialized_at: list[float] = []
    predicted_intervals: list[tuple[float, float]] = []
    updated_at: list[float] = []

    monkeypatch.setattr(assoc, "_events", lambda rf_measurements, radar: events)

    def fake_initial_tracker(initial_measurement, **kwargs):
        initialized_at.append(float(initial_measurement.time_s))
        return object()

    def fake_predict(
        tracker,
        *,
        current_time_s,
        target_time_s,
        acceleration_std_mps2,
    ):
        predicted_intervals.append((float(current_time_s), float(target_time_s)))

    def fake_update(tracker, measurement, **kwargs):
        updated_at.append(float(measurement.time_s))
        return object()

    def fake_record(**kwargs):
        measurement = kwargs["measurement"]
        return {"time_s": float(measurement.time_s), "source": measurement.source}

    monkeypatch.setattr(assoc, "_initial_mht_tracker", fake_initial_tracker)
    monkeypatch.setattr(assoc, "_predict_mht_to", fake_predict)
    monkeypatch.setattr(assoc, "_deterministic_update_mht_hypotheses", fake_update)
    monkeypatch.setattr(assoc, "_mht_record", fake_record)

    records, selected = assoc._run_mht_track_bank(
        rf_measurements=[],
        radar=_empty_radar(),
        covariance=np.eye(3),
        acceleration_std_mps2=1.0,
        gate_probabilities_by_source=None,
        gate_thresholds_by_source=None,
        safety_gate_probabilities_by_source=None,
        safety_gate_thresholds_by_source=None,
        robust_update_by_source=None,
        inflation_alpha_by_source=None,
        max_residual_norms_by_source=None,
        candidate_catprob_threshold=None,
        max_global_hypotheses=2,
        max_assignments_per_hypothesis=2,
        max_candidates_per_track=2,
        gate_probability=0.99,
        detection_probability=0.9,
        clutter_intensity=1.0e-6,
        prune_log_weight_delta=20.0,
    )

    assert initialized_at == [10.0]
    assert predicted_intervals == [(10.0, 11.0)]
    assert updated_at == [11.0]
    assert records == [
        {"time_s": 10.0, "source": "rf"},
        {"time_s": 11.0, "source": "rf"},
    ]
    assert selected.empty
