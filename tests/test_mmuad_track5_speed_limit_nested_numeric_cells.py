from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 100.0, 200.0],
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
def test_speed_limit_rejects_unsafe_nested_numeric_cells(
    column: str,
    value: object,
    message: str,
) -> None:
    rows = _submission()
    _set_object_cell(rows, column, value)

    with pytest.raises(ValueError, match=message):
        project_track5_speed_limit(rows)


def test_speed_limit_rejects_cyclic_numeric_cell() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    rows = _submission()
    _set_object_cell(rows, "state_z_m", cyclic)

    with pytest.raises(
        ValueError,
        match=r"non-scalar numeric values: state_z_m rows \[1\]",
    ):
        project_track5_speed_limit(rows)


def test_speed_limit_accepts_recursively_nested_real_scalar_cells() -> None:
    expected, expected_diagnostics = project_track5_speed_limit(_submission())
    rows = _submission()
    _set_object_cell(rows, "time_s", _nested_object_array(np.array(1.0)))
    _set_object_cell(
        rows,
        "state_x_m",
        _nested_object_array(np.ma.array(100.0, mask=False)),
    )
    _set_object_cell(rows, "Classification", _nested_object_array(np.array(2)))

    actual, actual_diagnostics = project_track5_speed_limit(rows)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual_diagnostics, expected_diagnostics)
