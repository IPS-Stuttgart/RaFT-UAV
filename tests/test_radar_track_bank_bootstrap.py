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


def test_track_bank_filters_invalid_candidates_before_mht_update(monkeypatch):
    rf = _rf_measurement(0.0)
    candidates = pd.DataFrame(
        {
            "time_s": [1.0, 1.0],
            "frame_index": [1, 1],
            "track_id": [10, 20],
            "east_m": [np.nan, 1.0],
            "north_m": [0.0, 2.0],
            "up_m": [0.0, 3.0],
            "cat_prob_uav": [1.0, 1.0],
        }
    )
    events = [
        {"time_s": rf.time_s, "priority": 0, "kind": "rf", "measurement": rf},
        {"time_s": 1.0, "priority": 1, "kind": "radar", "candidates": candidates},
    ]
    captured: dict[str, object] = {}

    class _TrackerView:
        state = np.zeros(6)
        covariance_matrix = np.eye(6)

    class _FakeTracker:
        def update_linear(self, measurements, observation, covariances):
            del observation
            captured["measurements"] = np.asarray(measurements, dtype=float)
            captured["covariances"] = np.asarray(covariances, dtype=float)

    monkeypatch.setattr(assoc, "_events", lambda rf_measurements, radar: events)
    monkeypatch.setattr(assoc, "_initial_mht_tracker", lambda *args, **kwargs: _FakeTracker())
    monkeypatch.setattr(assoc, "_predict_mht_to", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        assoc,
        "_tracker_from_best_mht_hypothesis",
        lambda *args, **kwargs: _TrackerView(),
    )

    def fake_selected(candidates, scored_candidates, tracker):
        del tracker
        captured["candidate_rows"] = len(candidates)
        captured["scored_rows"] = len(scored_candidates)
        return candidates.iloc[0].copy()

    monkeypatch.setattr(assoc, "_selected_row_from_best_mht_assignment", fake_selected)
    monkeypatch.setattr(
        assoc,
        "_mht_radar_measurement",
        lambda **kwargs: TrackingMeasurement(
            time_s=float(kwargs["time_s"]),
            vector=np.array([1.0, 2.0, 3.0]),
            covariance=np.eye(3),
            source="radar",
        ),
    )
    monkeypatch.setattr(assoc, "_mht_radar_diagnostics", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        assoc,
        "_mht_record",
        lambda **kwargs: {
            "time_s": float(kwargs["measurement"].time_s),
            "source": kwargs["measurement"].source,
        },
    )

    _, selected = assoc._run_mht_track_bank(
        rf_measurements=[],
        radar=candidates,
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

    assert captured["candidate_rows"] == 1
    assert captured["scored_rows"] == 1
    np.testing.assert_allclose(captured["measurements"], np.array([[1.0], [2.0], [3.0]]))
    assert captured["covariances"].shape == (3, 3, 1)
    assert selected["track_id"].tolist() == [20]


def test_radar_row_key_falls_back_to_time_when_frame_index_is_invalid():
    row = pd.Series({"frame_index": "", "time_s": 1.2345678912})

    assert assoc._radar_row_key(row) == ("time_s", 1.234567891)
