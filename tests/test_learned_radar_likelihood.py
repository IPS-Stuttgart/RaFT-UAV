import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines import learned_radar_association as learned_assoc
from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
    TrackingUpdateDiagnostics,
)
from raft_uav.baselines.learned_radar_association import (
    run_async_cv_baseline_with_learned_radar_association,
)
from raft_uav.baselines.learned_radar_likelihood import LearnedRadarAssociationModel
from raft_uav.baselines.radar_association import _nis_scored_candidates
from raft_uav.baselines.radar_likelihood_training import (
    collect_radar_association_training_frame,
)


def test_learned_radar_association_prefers_higher_likelihood_candidate():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 30.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.95,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 0.5,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.05,
            },
        ]
    )
    model = LearnedRadarAssociationModel(
        feature_names=("cat_prob_uav",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([10.0]),
        intercept=-5.0,
    )

    records, selected = run_async_cv_baseline_with_learned_radar_association(
        rf_measurements=[],
        radar=radar,
        model=model,
        candidate_catprob_threshold=None,
    )

    assert len(records) == 1
    assert selected["association_mode"].tolist() == ["learned-likelihood"]
    assert selected["track_id"].tolist() == [1]
    assert selected["association_learned_probability"].iloc[0] > 0.5


def test_learned_radar_association_ignores_invalid_candidate_positions():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 99,
                "time_s": 1.0,
                "east_m": np.nan,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.1,
            },
        ]
    )

    model = LearnedRadarAssociationModel(
        feature_names=("cat_prob_uav",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([10.0]),
        intercept=-5.0,
    )

    records, selected = run_async_cv_baseline_with_learned_radar_association(
        rf_measurements=[],
        radar=radar,
        model=model,
        candidate_catprob_threshold=None,
    )

    assert len(records) == 1
    assert selected["track_id"].tolist() == [1]
    assert selected["association_candidate_rows"].tolist() == [2]
    assert selected["association_invalid_candidate_rows"].tolist() == [1]


def test_learned_radar_association_coasts_skipped_policy_updates(monkeypatch):
    tracker_instances = []

    class FakeTracker:
        def __init__(
            self,
            *,
            initial_position: np.ndarray,
            initial_time_s: float,
            acceleration_std_mps2: float,
        ) -> None:
            del initial_position, acceleration_std_mps2
            self.current_time_s = float(initial_time_s)
            self.state = np.zeros(6)
            self.covariance_matrix = np.eye(6)
            self.coast_times: list[float] = []
            tracker_instances.append(self)

        def predict_to(self, time_s: float) -> None:
            self.current_time_s = float(time_s)

        def coast_to(self, time_s: float) -> None:
            self.coast_times.append(float(time_s))
            self.predict_to(time_s)

        def update(self, measurement: TrackingMeasurement, **kwargs: object):
            del kwargs
            if measurement.source == "rf":
                return TrackingUpdateDiagnostics(
                    time_s=float(measurement.time_s),
                    source="rf",
                    measurement_dim=2,
                    accepted=True,
                    update_action="updated",
                    nis=0.0,
                    gate_threshold=None,
                    safety_gate_threshold=None,
                    residual_gate_threshold_m=None,
                    covariance_scale=1.0,
                    inflation_alpha=None,
                    residual_norm_m=0.0,
                )
            raise AssertionError("skipped radar policy should not run a Kalman update")

    candidates = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.95,
            }
        ]
    )
    measurement = TrackingMeasurement(
        time_s=1.0,
        vector=np.array([10.0, 0.0, 0.0]),
        covariance=np.eye(3),
        source="radar",
    )
    initial_measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([0.0, 0.0]),
        covariance=np.eye(2),
        source="rf",
    )
    skipped = TrackingUpdateDiagnostics(
        time_s=1.0,
        source="radar",
        measurement_dim=3,
        accepted=False,
        update_action="do_no_harm_skip",
        nis=50.0,
        gate_threshold=None,
        safety_gate_threshold=None,
        residual_gate_threshold_m=None,
        covariance_scale=1.0,
        inflation_alpha=None,
        residual_norm_m=float("nan"),
    )

    monkeypatch.setattr(learned_assoc, "AsyncConstantVelocityKalmanTracker", FakeTracker)
    monkeypatch.setattr(
        learned_assoc,
        "_events",
        lambda rf_measurements, radar: [
            {"kind": "rf", "time_s": 0.0, "measurement": initial_measurement},
            {"kind": "radar", "time_s": 1.0, "candidates": candidates},
        ],
    )
    monkeypatch.setattr(
        learned_assoc,
        "_learned_initial_measurement_and_row",
        lambda *args, **kwargs: (initial_measurement, None),
    )
    monkeypatch.setattr(learned_assoc, "_catprob_candidate_pool", lambda frame, threshold: frame)
    monkeypatch.setattr(
        learned_assoc,
        "_nis_scored_candidates",
        lambda frame, tracker, covariance: frame.assign(
            association_nis=50.0,
            association_score=50.0,
        ),
    )
    monkeypatch.setattr(
        learned_assoc,
        "score_radar_candidates_with_learned_likelihood",
        lambda scored, **kwargs: scored.assign(association_score=0.0),
    )
    monkeypatch.setattr(learned_assoc, "_radar_row_to_measurement", lambda row, cov: measurement)
    monkeypatch.setattr(
        learned_assoc,
        "apply_radar_update_policy",
        lambda row, radar_measurement: (row, radar_measurement, skipped),
    )
    monkeypatch.setattr(
        learned_assoc,
        "_record",
        lambda radar_measurement, tracker, diagnostics, **kwargs: {
            "time_s": float(radar_measurement.time_s),
            "accepted": bool(diagnostics.accepted),
            "coast_times": tuple(tracker.coast_times),
            **kwargs,
        },
    )
    monkeypatch.setattr(
        learned_assoc,
        "_selected_rows_frame",
        lambda radar, rows: pd.DataFrame(rows),
    )

    model = LearnedRadarAssociationModel(
        feature_names=("cat_prob_uav",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([1.0]),
        intercept=0.0,
    )
    records, selected = learned_assoc.run_async_cv_baseline_with_learned_radar_association(
        rf_measurements=[],
        radar=candidates,
        model=model,
        candidate_catprob_threshold=None,
    )

    assert selected.empty
    assert tracker_instances[0].coast_times == [1.0]
    assert records[-1]["coast_times"] == (1.0,)


def test_model_scores_candidate_features_without_sklearn():
    tracker = AsyncConstantVelocityKalmanTracker(initial_position=np.zeros(3), initial_time_s=0.0)
    candidates = pd.DataFrame(
        [
            {"track_id": 1, "time_s": 0.0, "east_m": 1.0, "north_m": 0.0, "up_m": 0.0},
            {"track_id": 2, "time_s": 0.0, "east_m": 50.0, "north_m": 0.0, "up_m": 0.0},
        ]
    )
    scored = _nis_scored_candidates(candidates, tracker, np.eye(3))
    model = LearnedRadarAssociationModel(
        feature_names=("log1p_association_nis",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([-1.0]),
        intercept=0.0,
    )

    probabilities = model.predict_proba_features(
        pd.DataFrame({"log1p_association_nis": np.log1p(scored["association_nis"])})
    )

    assert probabilities[0] > probabilities[1]


def test_model_probabilities_are_stable_for_extreme_logits():
    model = LearnedRadarAssociationModel(
        feature_names=("x",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([1.0]),
        intercept=0.0,
    )
    features = pd.DataFrame({"x": [-1000.0, 0.0, 1000.0]})

    with np.errstate(over="raise", invalid="raise", under="ignore"):
        probabilities = model.predict_proba_features(features)

    np.testing.assert_allclose(
        probabilities,
        np.array([0.0, 0.5, 1.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_radar_association_training_ignores_invalid_oracle_candidates():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": np.nan,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.1,
            },
        ]
    )
    truth = pd.DataFrame({"time_s": [0.0], "east_m": [10.0], "north_m": [0.0], "up_m": [0.0]})

    examples = collect_radar_association_training_frame(
        rf_measurements=[],
        radar=radar,
        truth=truth,
        candidate_catprob_threshold=None,
        positive_gate_m=1.0,
    )

    labels = dict(zip(examples["track_id"], examples["label"], strict=True))
    truth_errors = dict(zip(examples["track_id"], examples["truth_error_m"], strict=True))
    assert labels == {2: 1}
    assert truth_errors[2] == 0.0


def test_learned_radar_model_save_writes_json_safe_metadata(tmp_path: Path) -> None:
    model = LearnedRadarAssociationModel(
        feature_names=("cat_prob_uav",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([2.0]),
        intercept=-1.0,
        metadata={
            "fold": np.int64(3),
            "validation_loss": np.nan,
            "source_path": tmp_path / "training.csv",
        },
    )

    path = tmp_path / "learned_radar_model.json"
    model.save(path)

    payload_text = path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    loaded = LearnedRadarAssociationModel.load(path)

    assert "NaN" not in payload_text
    assert payload["metadata"]["fold"] == 3
    assert payload["metadata"]["validation_loss"] is None
    assert payload["metadata"]["source_path"] == str(tmp_path / "training.csv")
    assert loaded.metadata["validation_loss"] is None


def test_learned_radar_model_rejects_non_finite_parameters() -> None:
    with pytest.raises(ValueError, match="weights"):
        LearnedRadarAssociationModel(
            feature_names=("cat_prob_uav",),
            mean=np.array([0.0]),
            scale=np.array([1.0]),
            weights=np.array([np.nan]),
            intercept=0.0,
        )

    with pytest.raises(ValueError, match="intercept"):
        LearnedRadarAssociationModel(
            feature_names=("cat_prob_uav",),
            mean=np.array([0.0]),
            scale=np.array([1.0]),
            weights=np.array([1.0]),
            intercept=float("inf"),
        )
