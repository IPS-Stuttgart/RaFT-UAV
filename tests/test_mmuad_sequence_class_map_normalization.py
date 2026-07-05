from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.submission import (
    estimates_to_official_mmaud_results_frame,
    load_sequence_class_map,
)


def test_csv_class_map_sequence_ids_are_normalized_like_official_rows(tmp_path: Path) -> None:
    class_map_csv = tmp_path / "class_map.csv"
    pd.DataFrame(
        {
            "Sequence": [" seq0001 ", "seq0002", "   "],
            "uav_type": [" 2 ", "3", "1"],
        }
    ).to_csv(class_map_csv, index=False)

    class_map = load_sequence_class_map(class_map_csv)

    assert class_map == {"seq0001": "2", "seq0002": "3"}

    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "time_s": [0.0, 0.0],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [3.0, 4.0],
            "state_z_m": [5.0, 6.0],
        }
    )
    official = estimates_to_official_mmaud_results_frame(
        estimates,
        classification=0,
        class_map=class_map,
    )

    assert official["Classification"].tolist() == [2, 3]
