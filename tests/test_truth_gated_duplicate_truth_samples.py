from __future__ import annotations

import pandas as pd

from raft_uav.io.aerpaw import select_radar_measurement_rows


def test_truth_gate_uses_final_numeric_equivalent_duplicate_sample() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["flight-a"],
            "track_id": [7],
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [5.0],
        },
        index=[11],
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["flight-a", "flight-a", "flight-a"],
            "time_s": ["0", 0.0, 1.0],
            "east_m": [100.0, 0.0, 1.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [5.0, 5.0, 5.0],
        },
        index=[4, 4, 5],
    )

    selected = select_radar_measurement_rows(
        radar,
        selection="truth-gated",
        truth=truth,
        truth_gate_m=1.0,
        truth_time_gate_s=0.1,
    )

    assert selected["track_id"].tolist() == [7]
    assert selected.index.tolist() == [11]


def test_truth_gate_keeps_unique_timestamp_behavior() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [3],
            "time_s": [0.0],
            "east_m": [2.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [2.0, 3.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    selected = select_radar_measurement_rows(
        radar,
        selection="truth-gated",
        truth=truth,
        truth_gate_m=0.1,
        truth_time_gate_s=0.1,
    )

    assert selected["track_id"].tolist() == [3]
