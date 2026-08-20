from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.paper_selection import select_paper_strict_raw_radar_track


def _radar(
    sequence_ids: list[object],
    *,
    flight_ids: list[object] | None = None,
) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "sequence_id": sequence_ids,
        "time_s": [0.0, 1.0, 2.0, 3.0],
        "frame_index": [0, 1, 2, 3],
        "track_id": [7, 7, 7, 7],
        "track_index": [0, 1, 2, 3],
        "cat_prob_uav": [0.9, 0.9, 0.1, 0.1],
    }
    if flight_ids is not None:
        data["flight_id"] = flight_ids
    return pd.DataFrame(data)


def test_paper_track_selection_rejects_pooled_sequences() -> None:
    radar = _radar(["flight-a", "flight-a", "flight-b", "flight-b"])

    with pytest.raises(ValueError, match="requires one sequence_id"):
        select_paper_strict_raw_radar_track(radar)


def test_paper_track_selection_rejects_pooled_physical_flights() -> None:
    radar = _radar(
        ["shared", "shared", "shared", "shared"],
        flight_ids=["flight-a", "flight-a", "flight-b", "flight-b"],
    )

    with pytest.raises(ValueError, match="requires one flight_id"):
        select_paper_strict_raw_radar_track(radar)


def test_paper_track_selection_allows_one_explicit_sequence() -> None:
    radar = _radar([" flight-a ", None, pd.NA, "flight-a"])

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["time_s"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_paper_track_selection_allows_one_explicit_physical_flight() -> None:
    radar = _radar(
        ["shared", "shared", "shared", "shared"],
        flight_ids=[" flight-a ", None, pd.NA, "flight-a"],
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["time_s"].tolist() == [0.0, 1.0, 2.0, 3.0]
