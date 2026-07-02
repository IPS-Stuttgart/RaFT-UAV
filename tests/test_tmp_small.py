import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.learned_radar_association import run_async_cv_baseline_with_learned_radar_association
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
    radar = pd.DataFrame([{"frame_index": 0, "track_id": 1, "time_s": 1.0, "east_m": 1.0, "north_m": 0.0, "up_m": 0.0, "cat_prob_uav": 0.9}])
    rf = TrackingMeasurement(time_s=0.0, vector=np.array([0.0, 0.0]), covariance=np.eye(2), source="rf")

    records, selected = run_async_cv_baseline_with_learned_radar_association(
        rf_measurements=[rf],
        radar=radar,
        model=_model(),
        candidate_catprob_threshold=None,
    )

    assert [record["source"] for record in records] == ["rf", "radar"]
    assert records[0]["update_action"] == "initialized"
    assert selected["frame_index"].tolist() == [0]
