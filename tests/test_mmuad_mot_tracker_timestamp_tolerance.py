from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.mot import MultiObjectTrackerConfig, run_mmuad_multi_object_tracker
from raft_uav.mmuad.schema import CandidateFrame


def test_tracker_groups_tolerance_equivalent_detections_into_one_frame() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1"],
                "time_s": [0.0, 0.5e-9],
                "source": ["radar", "radar"],
                "x_m": [0.0, 10.0],
                "y_m": [0.0, 0.0],
                "z_m": [2.0, 2.0],
                "confidence": [0.9, 0.8],
            }
        )
    )

    output = run_mmuad_multi_object_tracker(
        candidates,
        config=MultiObjectTrackerConfig(max_association_distance_m=15.0),
    )

    assert output.estimates["update_action"].tolist() == ["new_track", "new_track"]
    assert output.estimates["output_track_id"].tolist() == ["mot_1", "mot_2"]
    assert output.estimates["time_s"].tolist() == [0.0, 0.0]
    assert output.metrics["pooled"] == {"count": 2, "track_count": 2}


def test_tracker_groups_transitive_tolerance_chain_independent_of_row_order() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1", "s1"],
                "time_s": [1.5e-9, 0.0, 0.75e-9],
                "source": ["radar", "radar", "radar"],
                "x_m": [20.0, 0.0, 10.0],
                "y_m": [0.0, 0.0, 0.0],
                "z_m": [2.0, 2.0, 2.0],
                "confidence": [0.7, 0.9, 0.8],
            }
        )
    )

    output = run_mmuad_multi_object_tracker(
        candidates,
        config=MultiObjectTrackerConfig(max_association_distance_m=15.0),
    )

    assert output.estimates["update_action"].tolist() == [
        "new_track",
        "new_track",
        "new_track",
    ]
    assert output.estimates["output_track_id"].tolist() == ["mot_1", "mot_2", "mot_3"]
    assert output.estimates["time_s"].tolist() == [0.0, 0.0, 0.0]
    assert output.metrics["pooled"] == {"count": 3, "track_count": 3}
