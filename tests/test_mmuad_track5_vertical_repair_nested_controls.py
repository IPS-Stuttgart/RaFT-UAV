from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 1.0, 2.0],
            "Classification": [2, 2, 2],
        }
    )


def _object_scalar(value: object) -> np.ndarray:
    boxed = np.empty((), dtype=object)
    boxed[()] = value
    return boxed


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        (
            "max_vertical_speed_mps",
            _object_scalar(np.array(True)),
            "max_vertical_speed_mps must be finite and non-negative",
        ),
        (
            "max_neighbor_vertical_speed_mps",
            _object_scalar(np.array([10.0])),
            "max_neighbor_vertical_speed_mps must be finite and non-negative",
        ),
        (
            "max_vertical_residual_m",
            _object_scalar(np.array(False)),
            "max_vertical_residual_m must be finite and non-negative",
        ),
        (
            "max_horizontal_speed_mps",
            _object_scalar(np.array([80.0])),
            "max_horizontal_speed_mps must be finite and non-negative",
        ),
        (
            "iterations",
            _object_scalar(np.array(True)),
            "iterations must be a positive integer",
        ),
    ],
)
def test_vertical_repair_rejects_nested_pseudo_scalars(
    keyword: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        repair_track5_vertical_spikes(_submission_rows(), **{keyword: value})


def test_vertical_repair_rejects_cyclic_object_scalar() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(
        ValueError,
        match="max_vertical_speed_mps must be finite and non-negative",
    ):
        repair_track5_vertical_spikes(
            _submission_rows(),
            max_vertical_speed_mps=cyclic,
        )


def test_vertical_repair_accepts_recursively_boxed_real_scalars() -> None:
    repaired, diagnostics = repair_track5_vertical_spikes(
        _submission_rows(),
        max_vertical_speed_mps=_object_scalar(np.array(20.0)),
        max_neighbor_vertical_speed_mps=_object_scalar(np.array(10.0)),
        max_vertical_residual_m=_object_scalar(np.array(15.0)),
        max_horizontal_speed_mps=_object_scalar(np.array(80.0)),
        iterations=_object_scalar(np.array(2)),
    )

    assert len(repaired) == 3
    assert len(diagnostics) == 3
