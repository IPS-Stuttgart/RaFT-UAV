import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker
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
    assert labels == {1: 0, 2: 1}
    assert np.isnan(truth_errors[1])
    assert truth_errors[2] == 0.0
