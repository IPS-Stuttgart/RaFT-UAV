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


def _set_object_cell(rows: pd.DataFrame, column: str, value: object) -> None:
    rows[column] = rows[column].astype(object)
    rows.at[1, column] = value


@pytest.mark.parametrize(
    "value",
    [np.array(True), np.array(False), np.ma.array(True, mask=False)],
)
def test_speed_limit_rejects_wrapped_boolean_cells(value: object) -> None:
    rows = _submission()
    _set_object_cell(rows, "Classification", value)

    with pytest.raises(
        ValueError,
        match=r"Boolean numeric values: Classification rows \[1\]",
    ):
        project_track5_speed_limit(rows)


@pytest.mark.parametrize(
    "value",
    [
        1.0 + 2.0j,
        np.complex128(3.0 + 0.0j),
        np.array(4.0 + 5.0j),
        np.ma.array(6.0 + 0.0j, mask=False),
    ],
)
def test_speed_limit_rejects_complex_cells(value: object) -> None:
    rows = _submission()
    _set_object_cell(rows, "state_x_m", value)

    with pytest.raises(
        ValueError,
        match=r"complex numeric values: state_x_m rows \[1\]",
    ):
        project_track5_speed_limit(rows)


@pytest.mark.parametrize("value", [np.ma.array(1.0, mask=True), np.ma.masked])
def test_speed_limit_rejects_masked_cells(value: object) -> None:
    rows = _submission()
    _set_object_cell(rows, "state_y_m", value)

    with pytest.raises(
        ValueError,
        match=r"masked numeric values: state_y_m rows \[1\]",
    ):
        project_track5_speed_limit(rows)


@pytest.mark.parametrize("value", [np.array([1.0, 2.0]), np.array([[1.0]])])
def test_speed_limit_rejects_non_scalar_cells(value: object) -> None:
    rows = _submission()
    _set_object_cell(rows, "state_z_m", value)

    with pytest.raises(
        ValueError,
        match=r"non-scalar numeric values: state_z_m rows \[1\]",
    ):
        project_track5_speed_limit(rows)


def test_speed_limit_preserves_supported_zero_dimensional_numeric_cells() -> None:
    rows = _submission()
    _set_object_cell(rows, "time_s", np.array(1.0))
    _set_object_cell(rows, "state_x_m", np.ma.array(100.0, mask=False))

    limited, diagnostics = project_track5_speed_limit(rows, max_speed_mps=10.0)

    assert len(limited) == len(rows)
    assert len(diagnostics) == len(rows)
