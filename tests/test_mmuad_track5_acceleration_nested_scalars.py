from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import (
    repair_track5_acceleration_kinks,
)


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 10.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


def _nested_object_array(value: object) -> np.ndarray:
    nested = np.empty((), dtype=object)
    nested[()] = value
    return nested


def _set_object_cell(
    rows: pd.DataFrame,
    column: str,
    value: object,
) -> None:
    rows[column] = rows[column].astype(object)
    rows.at[1, column] = value


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (
            "max_acceleration_mps2",
            _nested_object_array(np.array(True)),
        ),
        (
            "max_direct_speed_mps",
            _nested_object_array(np.array([20.0])),
        ),
        (
            "min_interpolation_residual_m",
            _nested_object_array(np.ma.array(1.0, mask=True)),
        ),
        (
            "repair_blend",
            _nested_object_array(np.array(0.5 + 0.25j)),
        ),
        (
            "iterations",
            _nested_object_array(np.array(True)),
        ),
    ],
)
def test_acceleration_limit_rejects_unsafe_nested_controls(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        repair_track5_acceleration_kinks(_submission(), **{name: value})


def test_acceleration_limit_rejects_cyclic_control_array() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match="max_acceleration_mps2"):
        repair_track5_acceleration_kinks(
            _submission(),
            max_acceleration_mps2=cyclic,
        )


def test_acceleration_limit_accepts_recursively_nested_real_controls() -> None:
    repaired, diagnostics = repair_track5_acceleration_kinks(
        _submission(),
        max_acceleration_mps2=_nested_object_array(np.array(5.0)),
        max_direct_speed_mps=_nested_object_array(
            np.ma.array(20.0, mask=False)
        ),
        min_interpolation_residual_m=_nested_object_array(np.array(1.0)),
        iterations=_nested_object_array(np.array(1)),
        repair_blend=_nested_object_array(np.array(0.5)),
    )

    midpoint = repaired.loc[repaired["time_s"] == 1.0].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.5)
    assert diagnostics["acceleration_limit_applied"].sum() == 1


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "time_s",
            _nested_object_array(np.array(True)),
            r"Boolean numeric values: time_s rows \[1\]",
        ),
        (
            "state_x_m",
            _nested_object_array(np.array(1.0 + 2.0j)),
            r"complex numeric values: state_x_m rows \[1\]",
        ),
        (
            "state_y_m",
            _nested_object_array(np.array([1.0])),
            r"non-scalar numeric values: state_y_m rows \[1\]",
        ),
        (
            "Classification",
            _nested_object_array(np.ma.array(2, mask=True)),
            r"masked numeric values: Classification rows \[1\]",
        ),
    ],
)
def test_acceleration_limit_rejects_unsafe_nested_numeric_cells(
    column: str,
    value: object,
    message: str,
) -> None:
    rows = _submission()
    _set_object_cell(rows, column, value)

    with pytest.raises(ValueError, match=message):
        repair_track5_acceleration_kinks(rows)


def test_acceleration_limit_rejects_cyclic_numeric_cell() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    rows = _submission()
    _set_object_cell(rows, "state_z_m", cyclic)

    with pytest.raises(
        ValueError,
        match=r"non-scalar numeric values: state_z_m rows \[1\]",
    ):
        repair_track5_acceleration_kinks(rows)


def test_acceleration_limit_accepts_recursively_nested_real_numeric_cells() -> None:
    expected, expected_diagnostics = repair_track5_acceleration_kinks(_submission())
    rows = _submission()
    _set_object_cell(rows, "time_s", _nested_object_array(np.array(1.0)))
    _set_object_cell(
        rows,
        "state_x_m",
        _nested_object_array(np.ma.array(10.0, mask=False)),
    )
    _set_object_cell(
        rows,
        "Classification",
        _nested_object_array(np.array(2)),
    )

    actual, actual_diagnostics = repair_track5_acceleration_kinks(rows)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual_diagnostics, expected_diagnostics)
