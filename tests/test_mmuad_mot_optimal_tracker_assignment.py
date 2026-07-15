from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.mot import MultiObjectTrackerConfig, run_mmuad_multi_object_tracker
from raft_uav.mmuad.schema import CandidateFrame


def test_tracker_maximizes_frame_association_cardinality() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1", "s1", "s1"],
                "time_s": [0.0, 0.0, 1.0, 1.0],
                "source": ["radar", "radar", "radar", "radar"],
                "x_m": [0.0, 3.0, 1.0, -1.0],
                "y_m": [0.0, 0.0, 0.0, 0.0],
                "z_m": [2.0, 2.0, 2.0, 2.0],
                "confidence": [0.9, 0.8, 0.9, 0.8],
            }
        )
    )

    output = run_mmuad_multi_object_tracker(
        candidates,
        config=MultiObjectTrackerConfig(max_association_distance_m=2.0),
    )

    second_frame = output.estimates.loc[output.estimates["time_s"] == 1.0]
    assert second_frame["update_action"].tolist() == [
        "matched_update",
        "matched_update",
    ]
    assert second_frame["output_track_id"].tolist() == ["mot_2", "mot_1"]
    assert output.metrics["pooled"] == {"count": 4, "track_count": 2}
