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


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize(
    ("control", "message"),
    [
        ("max_speed_mps", "max_speed_mps must be positive and finite"),
        ("anchor_blend", r"anchor_blend must be finite and in \[0, 1\)"),
    ],
)
def test_speed_limit_rejects_boolean_scalar_controls(
    control: str,
    message: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=message):
        project_track5_speed_limit(_submission(), **{control: value})


@pytest.mark.parametrize(
    "column",
    ["time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"],
)
@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_speed_limit_rejects_boolean_fixed_grid_values(
    column: str,
    value: object,
) -> None:
    rows = _submission()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value

    with pytest.raises(ValueError) as error:
        project_track5_speed_limit(rows)

    assert "submission contains Boolean numeric values" in str(error.value)
    assert f"{column} rows [1]" in str(error.value)


def test_speed_limit_keeps_ordinary_zero_and_one_numeric_values() -> None:
    limited, diagnostics = project_track5_speed_limit(
        _submission(),
        max_speed_mps=1,
        iterations=1,
        anchor_blend=0,
    )

    assert len(limited) == 3
    assert len(diagnostics) == 3
