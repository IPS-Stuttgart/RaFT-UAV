from __future__ import annotations

import numpy as np
import pytest

from raft_uav.imm_cli import _records_to_frame


def _record(
    *,
    time_s: object = 1.0,
    first_state: object = 1.0,
    filtered_first_state: object | None = None,
    mode_probability: object | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "time_s": time_s,
        "source": "radar",
        "state": [first_state, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    if filtered_first_state is not None:
        record["filtered_state"] = [
            filtered_first_state,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
    if mode_probability is not None:
        record["mode_probability_map"] = {"constant-velocity": mode_probability}
    return record


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
        _records_to_frame([_record(time_s=value)])


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
        _records_to_frame([_record(first_state=value)])


def test_records_to_frame_rejects_invalid_filtered_state_components() -> None:
    with pytest.raises(
        ValueError,
        match="record 0 filtered_state must contain exactly 6 finite real scalars",
    ):
        _records_to_frame([_record(filtered_first_state=True)])


def test_records_to_frame_rejects_invalid_mode_probabilities() -> None:
    with pytest.raises(
        ValueError,
        match="record 0 mode probability 'constant-velocity' must be a finite real scalar",
    ):
        _records_to_frame([_record(mode_probability=np.nan)])


def test_records_to_frame_accepts_scalar_like_reals_and_sorts_numerically() -> None:
    frame = _records_to_frame(
        [
            {
                "time_s": np.asarray("10"),
                "source": "rf",
                "state": [np.asarray(value) for value in range(6)],
                "filtered_state": [np.asarray(str(value)) for value in range(6)],
                "mode_probability_map": {"constant-velocity": np.asarray("0.75")},
            },
            {
                "time_s": "2",
                "source": "radar",
                "state": np.arange(6, dtype=float).reshape(2, 3),
            },
        ]
    )

    assert frame["time_s"].tolist() == [2.0, 10.0]
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
        np.arange(6, dtype=float)[None, :].repeat(2, axis=0),
    )
    assert frame.loc[1, "mode_probability_constant_velocity"] == 0.75
    assert frame.loc[1, "filtered_east_m"] == 0.0
