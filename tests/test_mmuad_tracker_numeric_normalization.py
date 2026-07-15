from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker, select_tracklet_path


def _tracker_config() -> TrackerConfig:
    return TrackerConfig(selection_mobility_radius_m=0.0)


def test_tracker_orders_numeric_string_timestamps_chronologically() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq", "seq", "seq"],
                "time_s": ["1", "10", "2"],
                "source": ["radar", "radar", "radar"],
                "x_m": ["1", "10", "2"],
                "y_m": ["0", "0", "0"],
                "z_m": ["0", "0", "0"],
            }
        )
    )

    output = run_mmuad_tracker(candidates, config=_tracker_config())

    assert output.selected_tracklets["time_s"].tolist() == [1.0, 2.0, 10.0]
    assert output.estimates["time_s"].tolist() == [1.0, 2.0, 10.0]


def test_tracker_groups_equivalent_numeric_timestamp_representations() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq", "seq"],
            "time_s": [1, "1.0", "2"],
            "source": ["radar", "radar", "radar"],
            "x_m": [0, "100", 2],
            "y_m": [0, 0, 0],
            "z_m": [0, 0, 0],
            "confidence": ["0.2", "0.9", "0.8"],
        }
    )

    selected = select_tracklet_path(candidates, config=_tracker_config())

    assert selected["time_s"].tolist() == [1.0, 2.0]
    assert selected["x_m"].tolist() == [100.0, 2.0]
