from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from raft_uav.mmuad.mot import MultiObjectTrackerConfig, run_mmuad_multi_object_tracker
from raft_uav.mmuad.schema import CandidateFrame


def test_symmetric_optimal_assignment_ignores_input_row_order() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["s1"] * 4,
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["radar"] * 4,
            "track_id": ["left", "right", "south", "north"],
            "x_m": [-1.0, 1.0, 0.0, 0.0],
            "y_m": [0.0, 0.0, -1.0, 1.0],
            "z_m": [2.0] * 4,
            "confidence": [0.9] * 4,
        }
    )
    reversed_rows = pd.concat(
        [
            rows.loc[rows["time_s"] == 0.0].iloc[::-1],
            rows.loc[rows["time_s"] == 1.0].iloc[::-1],
        ],
        ignore_index=True,
    )
    config = MultiObjectTrackerConfig(max_association_distance_m=2.0)
    forward = run_mmuad_multi_object_tracker(CandidateFrame(rows), config=config)
    backward = run_mmuad_multi_object_tracker(
        CandidateFrame(reversed_rows),
        config=config,
    )
    columns = ["time_s", "track_id", "output_track_id"]
    forward_ids = (
        forward.estimates.loc[:, columns]
        .sort_values(columns[:2])
        .reset_index(drop=True)
    )
    backward_ids = (
        backward.estimates.loc[:, columns]
        .sort_values(columns[:2])
        .reset_index(drop=True)
    )

    assert_frame_equal(forward_ids, backward_ids)
