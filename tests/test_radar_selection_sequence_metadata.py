from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.io.aerpaw import select_radar_measurement_rows


def _position_rows(
    sequence_ids: list[object] | None,
    east_m: list[float],
) -> pd.DataFrame:
    count = len(east_m)
    data: dict[str, list[object] | list[float]] = {
        "time_s": [0.0] * count,
        "east_m": east_m,
        "north_m": [0.0] * count,
        "up_m": [0.0] * count,
    }
    if sequence_ids is not None:
        data["sequence_id"] = sequence_ids
    return pd.DataFrame(data)


def _truth_gated(radar: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    return select_radar_measurement_rows(
        radar,
        selection="truth-gated",
        truth=truth,
        truth_gate_m=1.0,
        truth_time_gate_s=0.1,
    )


def test_catprob_rejects_partial_sequence_metadata() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["flight-a", None],
            "frame_index": [0, 0],
            "track_id": [10, 20],
            "cat_prob_uav": [0.9, 0.8],
        }
    )

    with pytest.raises(ValueError, match="radar sequence_id must be complete"):
        select_radar_measurement_rows(
            radar,
            selection="catprob",
            catprob_threshold=0.5,
        )


def test_truth_gate_rejects_partial_radar_sequence_metadata() -> None:
    radar = _position_rows(["flight-a", None], [0.0, 100.0])
    truth = _position_rows(["flight-a"], [0.0])

    with pytest.raises(ValueError, match="radar sequence_id must be complete"):
        _truth_gated(radar, truth)


def test_truth_gate_rejects_partial_truth_sequence_metadata() -> None:
    radar = _position_rows(["flight-a"], [0.0])
    truth = _position_rows(["flight-a", None], [0.0, 100.0])

    with pytest.raises(ValueError, match="truth sequence_id must be complete"):
        _truth_gated(radar, truth)


def test_truth_gate_rejects_pooled_radar_with_unlabeled_truth() -> None:
    radar = _position_rows(["flight-a", "flight-b"], [0.0, 100.0])
    truth = _position_rows(None, [100.0])

    with pytest.raises(ValueError, match="cannot align pooled radar"):
        _truth_gated(radar, truth)


def test_truth_gate_rejects_unlabeled_radar_with_pooled_truth() -> None:
    radar = _position_rows(None, [100.0])
    truth = _position_rows(["flight-a", "flight-b"], [0.0, 100.0])

    with pytest.raises(ValueError, match="against pooled truth"):
        _truth_gated(radar, truth)


def test_truth_gate_preserves_one_sided_single_sequence_metadata() -> None:
    radar = _position_rows(["flight-a"], [100.0])
    truth = _position_rows(None, [100.0])

    selected = _truth_gated(radar, truth)

    assert selected["sequence_id"].tolist() == ["flight-a"]
