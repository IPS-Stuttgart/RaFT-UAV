from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_score_calibration import _attach_truth_targets


def test_truth_target_mask_stays_aligned_for_interleaved_sequences() -> None:
    candidates = pd.DataFrame(
        [
            {
                "row_id": "a-0",
                "sequence_id": "a",
                "time_s": 0.0,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
            },
            {
                "row_id": "b-0",
                "sequence_id": "b",
                "time_s": 0.0,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
            },
            {
                "row_id": "a-1",
                "sequence_id": "a",
                "time_s": 1.0,
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
            },
            {
                "row_id": "b-1",
                "sequence_id": "b",
                "time_s": 1.0,
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
            },
        ]
    )
    truth = pd.DataFrame(
        [
            {"sequence_id": "a", "time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
            {"sequence_id": "a", "time_s": 1.0, "x_m": 1.0, "y_m": 0.0, "z_m": 0.0},
            {"sequence_id": "b", "time_s": 1.0, "x_m": 1.0, "y_m": 0.0, "z_m": 0.0},
        ]
    )

    labelled = _attach_truth_targets(
        candidates,
        truth,
        good_threshold_m=0.1,
        max_truth_time_delta_s=0.01,
    )

    assert labelled["row_id"].tolist() == candidates["row_id"].tolist()
    assert labelled["candidate_truth_matched"].tolist() == [True, False, True, True]
