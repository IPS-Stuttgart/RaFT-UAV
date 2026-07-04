from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines import learned_radar_association as learned_association
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.learned_radar_association import (
    run_async_cv_baseline_with_learned_radar_association,
)
from raft_uav.baselines.learned_radar_likelihood import LearnedRadarAssociationModel


def _model():
    return LearnedRadarAssociationModel(
        feature_names=("cat_prob_uav",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([10.0]),
        intercept=-5.0,
    )


def test_learned_radar_association_records_rf_bootstrap_event_once():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            }
        ]
    )
    rf = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([0.0, 0.0]),
        covariance=np.eye(2),
        source="rf",
    )

    records, selected = run_async_cv_baseline_with_learned_radar_association(
        rf_measurements=[rf],
        radar=radar,
        model=_model(),
        candidate_catprob_threshold=None,
    )

    assert [record["source"] for record in records] == ["rf", "radar"]
    assert records[0]["update_action"] == "initialized"
    assert selected["frame_index"].tolist() == [0]


def test_learned_radar_association_scores_radar_bootstrap_once(monkeypatch):
    scored_times: list[float] = []

    def fake_score_radar_candidates(scored: pd.DataFrame, **kwargs) -> pd.DataFrame:
        del kwargs
        scored_times.append(float(scored["time_s"].iloc[0]))
        out = scored.copy()
        out["association_score"] = out["association_nis"]
        return out

    monkeypatch.setattr(
        learned_association,
        "score_radar_candidates_with_learned_likelihood",
        fake_score_radar_candidates,
    )
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )

    records, selected = learned_association.run_async_cv_baseline_with_learned_radar_association(
        rf_measurements=[],
        radar=radar,
        model=_model(),
        candidate_catprob_threshold=None,
    )

    assert [record["time_s"] for record in records] == [0.0, 1.0]
    assert records[0]["source"] == "radar"
    assert records[0]["update_action"] == "initialized"
    assert scored_times == [0.0, 1.0]
    assert selected["frame_index"].tolist() == [0, 1]


def test_learned_radar_association_respects_initial_catprob_threshold():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.2,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.3,
            },
        ]
    )

    records, selected = run_async_cv_baseline_with_learned_radar_association(
        rf_measurements=[],
        radar=radar,
        model=_model(),
        candidate_catprob_threshold=0.4,
    )

    assert records == []
    assert selected.empty
