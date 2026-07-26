from __future__ import annotations

import pandas as pd

from raft_uav.io.aerpaw import select_radar_measurement_rows


def test_catprob_selection_scopes_reused_frame_ids_by_sequence() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "frame_index": [0, 0, 0, 0],
            "track_id": [10, 11, 20, 21],
            "cat_prob_uav": [0.90, 0.80, 0.70, 0.60],
        },
        index=[5, 5, 5, 5],
    )

    selected = select_radar_measurement_rows(
        radar,
        selection="catprob",
        catprob_threshold=0.5,
    )

    assert selected["sequence_id"].tolist() == ["flight-a", "flight-b"]
    assert selected["track_id"].tolist() == [10, 20]
    assert selected.index.tolist() == [5, 5]
