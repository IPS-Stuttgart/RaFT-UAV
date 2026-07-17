from __future__ import annotations

import pandas as pd

from raft_uav.research.runtime_modes import backward_repair_associations


def _selected_rows(frame_indices: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": frame_indices,
            "track_id": [0, 3],
            "time_s": [0.0, 3.0],
            "east_m": [0.0, 3.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def _candidate_rows(frame_indices: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": frame_indices,
            "track_id": [0, 1, 2, 3],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "east_m": [0.0, 1.0, 2.0, 3.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _repair(selected: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    return backward_repair_associations(
        selected,
        candidates,
        max_gap_s=4.0,
        max_repair_distance_m=0.0,
    )


def test_backward_repair_uses_time_keys_for_fractional_frame_indices() -> None:
    selected = _selected_rows([0.0, 3.0])
    candidates = _candidate_rows([0.0, 1.25, 1.75, 3.0])

    repaired = _repair(selected, candidates)

    assert repaired["time_s"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_backward_repair_preserves_large_integer_frame_indices_exactly() -> None:
    selected = _selected_rows(["9007199254740992", "9007199254740995"])
    candidates = _candidate_rows(
        [
            "9007199254740992",
            "9007199254740993",
            "9007199254740994",
            "9007199254740995",
        ]
    )

    repaired = _repair(selected, candidates)

    assert repaired["time_s"].tolist() == [0.0, 1.0, 2.0, 3.0]
