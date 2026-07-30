from __future__ import annotations

import pandas as pd
import pytest

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


def _coincident_rows(sequence_ids: pd.Series | list[object]) -> pd.DataFrame:
    count = len(sequence_ids)
    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "time_s": [0.0] * count,
            "east_m": [0.0] * count,
            "north_m": [0.0] * count,
            "up_m": [0.0] * count,
        }
    )


def test_truth_gate_rejects_text_colliding_radar_sequence_ids() -> None:
    radar = _coincident_rows(pd.Series([1, "1"], dtype=object))
    truth = _coincident_rows(["1"])

    with pytest.raises(ValueError, match="radar sequence_id contains ambiguous values"):
        _truth_gated(radar, truth)


def test_truth_gate_rejects_text_colliding_truth_sequence_ids() -> None:
    radar = _coincident_rows(["1"])
    truth = _coincident_rows(pd.Series([1, "1"], dtype=object))

    with pytest.raises(ValueError, match="truth sequence_id contains ambiguous values"):
        _truth_gated(radar, truth)


def test_truth_gate_matches_numeric_and_text_ids_across_tables() -> None:
    radar = _coincident_rows(pd.Series([1], dtype=object))
    truth = _coincident_rows(["1"])

    selected = _truth_gated(radar, truth)

    assert selected["sequence_id"].tolist() == [1]
