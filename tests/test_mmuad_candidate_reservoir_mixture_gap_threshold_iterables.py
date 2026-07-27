from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_mixture_gap_frames import summarize_frame_gap


def test_summary_reuses_generator_thresholds_for_every_group_and_oracle() -> None:
    frame_gap = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "mixture_error_3d_m": [3.0, 4.0],
            "oracle_all_3d_m": [1.0, 1.0],
            "oracle_top3_3d_m": [2.0, 2.0],
            "gap_to_oracle_all_3d_m": [2.0, 3.0],
            "gap_to_oracle_top3_3d_m": [1.0, 2.0],
        }
    )

    thresholds = (threshold for threshold in (1.0,))
    summary = summarize_frame_gap(
        frame_gap,
        group_column="sequence_id",
        gap_thresholds_m=thresholds,
    ).set_index("sequence_id")

    assert summary["frames_gap_to_oracle_all_gt_1m"].to_dict() == {
        "seqA": 1,
        "seqB": 1,
    }
    assert summary["frames_gap_to_oracle_top3_gt_1m"].to_dict() == {
        "seqA": 0,
        "seqB": 1,
    }
