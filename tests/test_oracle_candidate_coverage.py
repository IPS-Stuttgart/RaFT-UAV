import numpy as np
import pandas as pd

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.evaluation.oracle_coverage import build_oracle_candidate_coverage


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"time_s": 0.0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0},
            {"time_s": 1.0, "east_m": 10.0, "north_m": 0.0, "up_m": 0.0},
        ]
    )


def test_oracle_candidate_coverage_flags_top_k_pruning_loss():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0,
                "cat_prob_uav": 0.10,
            },
            {
                "frame_index": 0,
                "track_index": 1,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 1,
                "track_index": 0,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 110.0,
                "cat_prob_uav": 0.99,
            },
        ]
    )

    result = build_oracle_candidate_coverage(
        radar=radar,
        truth=_truth(),
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            max_candidates_per_frame=1,
            catprob_weight=5.0,
            anchor_nis_weight=0.0,
            range_gate_m=None,
        ),
    )

    frame = result.frame_coverage
    assert frame["oracle_retained"].tolist() == [False, True]
    assert frame["oracle_drop_reason"].tolist() == ["top_k", "retained"]
    assert result.summary["oracle_available_frames"] == 2
    assert result.summary["oracle_retained_frames"] == 1
    assert np.isclose(result.summary["oracle_retention_rate"], 0.5)
    assert result.summary["top_k_drop_frames"] == 1


def test_oracle_candidate_coverage_separates_catprob_threshold_loss():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.10,
            },
            {
                "frame_index": 0,
                "track_index": 1,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.90,
            },
        ]
    )

    result = build_oracle_candidate_coverage(
        radar=radar,
        truth=_truth(),
        candidate_catprob_threshold=0.5,
        config=TrackletViterbiAssociationConfig(
            max_candidates_per_frame=8,
            catprob_weight=1.0,
            anchor_nis_weight=0.0,
            range_gate_m=None,
        ),
    )

    frame = result.frame_coverage.iloc[0]
    assert not bool(frame["oracle_retained"])
    assert not bool(frame["oracle_passed_catprob_threshold"])
    assert frame["oracle_drop_reason"] == "catprob_threshold"
    assert result.summary["catprob_threshold_drop_frames"] == 1
