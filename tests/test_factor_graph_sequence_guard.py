from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.research import (
    coordinate_descent_association_and_smoothing,
    smooth_position_trajectory,
)
from raft_uav.research import factor_graph


def _positions(sequence_ids: list[object]) -> pd.DataFrame:
    size = len(sequence_ids)
    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "time_s": [float(index) for index in range(size)],
            "east_m": [10.0 * index for index in range(size)],
            "north_m": [0.0] * size,
            "up_m": [5.0] * size,
        }
    )


def _radar(sequence_ids: list[object]) -> pd.DataFrame:
    frame = _positions(sequence_ids)
    frame["frame_index"] = list(range(len(frame)))
    frame["track_id"] = list(range(1, len(frame) + 1))
    return frame


@pytest.mark.parametrize(
    "smoother",
    [smooth_position_trajectory, factor_graph.smooth_position_trajectory],
)
def test_factor_graph_smoother_rejects_pooled_sequences(smoother) -> None:
    measurements = _positions(["flight-a", "flight-b"])

    with pytest.raises(
        ValueError,
        match="measurements contains multiple sequence_id values",
    ):
        smoother(measurements)


def test_factor_graph_smoother_rejects_mismatched_initial_sequence() -> None:
    measurements = _positions(["flight-a"])
    initial = _positions(["flight-b"])

    with pytest.raises(
        ValueError,
        match="measurements and initial sequence_id values do not match",
    ):
        smooth_position_trajectory(measurements, initial=initial)


def test_factor_graph_coordinate_descent_rejects_pooled_radar_sequences() -> None:
    radar = _radar(["flight-a", "flight-b"])

    with pytest.raises(
        ValueError,
        match="radar contains multiple sequence_id values",
    ):
        coordinate_descent_association_and_smoothing(radar, iterations=0)


def test_factor_graph_coordinate_descent_rejects_mismatched_rf_sequence() -> None:
    radar = _radar(["flight-a"])
    rf = _positions(["flight-b"])

    with pytest.raises(
        ValueError,
        match="radar and rf sequence_id values do not match",
    ):
        coordinate_descent_association_and_smoothing(radar, rf, iterations=0)


def test_factor_graph_smoother_accepts_one_sequence_with_missing_labels() -> None:
    measurements = _positions([" flight-a ", None])

    result = smooth_position_trajectory(measurements)

    assert result.estimates["time_s"].tolist() == [0.0, 1.0]
