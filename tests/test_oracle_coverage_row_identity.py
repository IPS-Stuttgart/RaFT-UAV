import pandas as pd

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.evaluation.oracle_coverage import build_oracle_candidate_coverage


def test_oracle_coverage_keeps_equal_geometry_candidates_distinct() -> None:
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_index": 7,
                "track_id": 11,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.1,
            },
            {
                "frame_index": 0,
                "track_index": 7,
                "track_id": 11,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
        ]
    )
    truth = pd.DataFrame(
        [
            {"time_s": 0.0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0},
            {"time_s": 1.0, "east_m": 1.0, "north_m": 0.0, "up_m": 0.0},
        ]
    )

    result = build_oracle_candidate_coverage(
        radar=radar,
        truth=truth,
        candidate_catprob_threshold=0.5,
        config=TrackletViterbiAssociationConfig(
            max_candidates_per_frame=8,
            catprob_weight=1.0,
            anchor_nis_weight=0.0,
            range_gate_m=None,
        ),
    )

    frame = result.frame_coverage.iloc[0]
    assert frame["oracle_cat_prob_uav"] == 0.1
    assert not bool(frame["oracle_passed_catprob_threshold"])
    assert not bool(frame["oracle_retained"])
    assert frame["oracle_drop_reason"] == "catprob_threshold"
