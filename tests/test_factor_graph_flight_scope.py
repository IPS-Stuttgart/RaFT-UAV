from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.research import (
    coordinate_descent_association_and_smoothing,
    smooth_position_trajectory,
)
from raft_uav.research import factor_graph


def _positions(
    flight_ids: list[object],
    *,
    sequence_ids: list[object] | None = None,
) -> pd.DataFrame:
    size = len(flight_ids)
    frame = pd.DataFrame(
        {
            "flight_id": flight_ids,
            "time_s": [float(index) for index in range(size)],
            "east_m": [10.0 * index for index in range(size)],
            "north_m": [0.0] * size,
            "up_m": [5.0] * size,
        }
    )
    if sequence_ids is not None:
        frame.insert(0, "sequence_id", sequence_ids)
    return frame


def _radar(
    flight_ids: list[object],
    *,
    sequence_ids: list[object] | None = None,
) -> pd.DataFrame:
    frame = _positions(flight_ids, sequence_ids=sequence_ids)
    frame["frame_index"] = list(range(len(frame)))
    frame["track_id"] = list(range(1, len(frame) + 1))
    return frame


@pytest.mark.parametrize(
    "smoother",
    [smooth_position_trajectory, factor_graph.smooth_position_trajectory],
)
def test_factor_graph_smoother_rejects_pooled_flights_with_shared_sequence(
    smoother,
) -> None:
    measurements = _positions(
        ["flight-a", "flight-b"],
        sequence_ids=["campaign", "campaign"],
    )

    with pytest.raises(
        ValueError,
        match="measurements contains multiple flight_id values",
    ):
        smoother(measurements)


def test_factor_graph_smoother_rejects_mismatched_initial_flight() -> None:
    measurements = _positions(["flight-a"], sequence_ids=["campaign"])
    initial = _positions(["flight-b"], sequence_ids=["campaign"])

    with pytest.raises(
        ValueError,
        match="measurements and initial flight_id values do not match",
    ):
        smooth_position_trajectory(measurements, initial=initial)


def test_factor_graph_coordinate_descent_rejects_flight_id_only_pool() -> None:
    radar = _radar(["flight-a", "flight-b"])

    with pytest.raises(
        ValueError,
        match="radar contains multiple flight_id values",
    ):
        coordinate_descent_association_and_smoothing(radar, iterations=0)


def test_factor_graph_coordinate_descent_rejects_mismatched_rf_flight() -> None:
    radar = _radar(["flight-a"], sequence_ids=["campaign"])
    rf = _positions(["flight-b"], sequence_ids=["campaign"])

    with pytest.raises(
        ValueError,
        match="radar and rf flight_id values do not match",
    ):
        coordinate_descent_association_and_smoothing(radar, rf, iterations=0)


def test_factor_graph_smoother_rejects_partially_missing_flight_labels() -> None:
    measurements = _positions(
        [" flight-a ", None],
        sequence_ids=["campaign", "campaign"],
    )

    with pytest.raises(ValueError, match="partially missing flight_id"):
        smooth_position_trajectory(measurements)
