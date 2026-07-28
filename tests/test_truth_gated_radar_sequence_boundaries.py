from __future__ import annotations

import pandas as pd

from raft_uav.io.aerpaw import select_radar_measurement_rows


def _truth_gated(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    return select_radar_measurement_rows(
        radar,
        selection="truth-gated",
        truth=truth,
        truth_gate_m=1.0,
        truth_time_gate_s=0.1,
    )


def test_truth_gated_radar_selection_respects_sequence_boundaries() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.01, 0.01],
            "east_m": [100.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        },
        index=[7, 7],
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.01],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    selected = _truth_gated(radar, truth)

    assert selected["sequence_id"].tolist() == ["seqB"]
    assert selected.index.tolist() == [7]


def test_truth_gate_scopes_truth_even_for_one_radar_sequence() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.01],
            "east_m": [100.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.01],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    assert _truth_gated(radar, truth).empty
