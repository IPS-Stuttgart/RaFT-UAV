import pandas as pd

from raft_uav.mmuad.submission import estimates_to_official_mmaud_results_frame


def test_official_results_frame_drop_policy_keeps_positions_aligned_after_first_row_drop():
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0", "seq1"],
            "time_s": [None, 1.0],
            "state_x_m": [10.0, 2.0],
            "state_y_m": [20.0, 4.0],
            "state_z_m": [30.0, 6.0],
            "class_id": [0, 2],
        }
    )

    results = estimates_to_official_mmaud_results_frame(
        estimates,
        invalid_row_policy="drop",
    )

    assert results.to_dict("records") == [
        {
            "Sequence": "seq1",
            "Timestamp": 1.0,
            "Position": "(2,4,6)",
            "Classification": 2,
        }
    ]
