from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.lofo_tracking import records_to_frame, tracking_metrics


def _record(*, time_s: object = 1.0, first_state: object = 1.0) -> dict[str, object]:
    return {
        "time_s": time_s,
        "source": "radar",
        "state": [first_state, 2.0, 3.0, 4.0, 5.0, 6.0],
    }


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        1.0 + 0.0j,
        np.asarray([1.0]),
        np.ma.masked,
    ],
)
def test_records_to_frame_rejects_invalid_timestamps(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="record 0 time_s must be a finite real scalar",
    ):
        records_to_frame([_record(time_s=value)])


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        1.0 + 0.0j,
        np.asarray([1.0]),
        np.ma.masked,
    ],
)
def test_records_to_frame_rejects_invalid_state_components(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="record 0 state must contain exactly 6 finite real scalars",
    ):
        records_to_frame([_record(first_state=value)])


def test_records_to_frame_accepts_zero_dimensional_real_scalars() -> None:
    frame = records_to_frame(
        [
            {
                "time_s": np.asarray("2.5"),
                "source": "rf",
                "state": [np.asarray(value) for value in range(6)],
            }
        ]
    )

    assert frame["time_s"].tolist() == [2.5]
    np.testing.assert_allclose(
        frame[
            [
                "east_m",
                "north_m",
                "up_m",
                "v_east_mps",
                "v_north_mps",
                "v_up_mps",
            ]
        ].to_numpy(),
        np.arange(6, dtype=float)[None, :],
    )


def _trajectory(*, boolean_column: str | None = None) -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "time_s": [0.0, 1.0],
        "east_m": [0.0, 1.0],
        "north_m": [0.0, 0.0],
        "up_m": [0.0, 0.0],
    }
    if boolean_column is not None:
        values[boolean_column] = [False, values[boolean_column][1]]
    return pd.DataFrame(values)


@pytest.mark.parametrize("boolean_column", ["time_s", "east_m"])
def test_tracking_metrics_does_not_hide_boolean_inputs(boolean_column: str) -> None:
    with pytest.raises(ValueError, match="must not contain Boolean values"):
        tracking_metrics(
            flight_name="flight",
            truth=_trajectory(boolean_column=boolean_column),
            rf=pd.DataFrame(),
            radar=pd.DataFrame(),
            selected=pd.DataFrame(),
            estimates=_trajectory(),
        )
