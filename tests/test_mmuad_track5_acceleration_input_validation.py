from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import repair_track5_acceleration_kinks


def _kink_submission() -> pd.DataFrame:
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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_acceleration_mps2", True),
        ("max_acceleration_mps2", np.bool_(True)),
        ("max_direct_speed_mps", True),
        ("min_interpolation_residual_m", False),
        ("repair_blend", True),
        ("repair_blend", np.array(False)),
    ],
)
def test_acceleration_limit_rejects_boolean_controls(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        repair_track5_acceleration_kinks(_kink_submission(), **{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_acceleration_mps2", np.array([5.0])),
        ("max_direct_speed_mps", np.array([20.0])),
        ("min_interpolation_residual_m", np.array([1.0])),
        ("repair_blend", np.array([0.5])),
    ],
)
def test_acceleration_limit_rejects_non_scalar_controls(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        repair_track5_acceleration_kinks(_kink_submission(), **{name: value})


@pytest.mark.parametrize(
    "iterations",
    [0, -1, 1.5, True, np.bool_(True), np.nan, np.inf, np.array([1])],
)
def test_acceleration_limit_rejects_invalid_iterations(iterations: object) -> None:
    with pytest.raises(ValueError, match="iterations"):
        repair_track5_acceleration_kinks(
            _kink_submission(),
            iterations=iterations,
        )


def test_acceleration_limit_accepts_numeric_scalar_like_controls() -> None:
    repaired, diagnostics = repair_track5_acceleration_kinks(
        _kink_submission(),
        max_acceleration_mps2=np.array(5.0),
        max_direct_speed_mps=np.float64(20.0),
        min_interpolation_residual_m=np.int64(1),
        iterations=np.array(1),
        repair_blend=np.float64(0.5),
    )

    midpoint = repaired.loc[repaired["time_s"] == 1.0].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.5)
    assert diagnostics["acceleration_limit_applied"].sum() == 1


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("time_s", np.nan, "non-finite numeric values"),
        ("state_x_m", "not-a-number", "non-finite numeric values"),
        ("Classification", np.nan, "non-finite numeric values"),
        ("state_y_m", True, "Boolean numeric values"),
        ("Classification", np.bool_(False), "Boolean numeric values"),
    ],
)
def test_acceleration_limit_rejects_invalid_normalized_rows(
    column: str,
    value: object,
    message: str,
) -> None:
    rows = _kink_submission()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value

    with pytest.raises(ValueError, match=message):
        repair_track5_acceleration_kinks(rows)


def test_acceleration_limit_rejects_duplicate_fixed_grid_keys() -> None:
    rows = _kink_submission()
    rows.loc[2, "time_s"] = 1.0
    rows.loc[2, "state_x_m"] = 100.0

    with pytest.raises(ValueError) as error:
        repair_track5_acceleration_kinks(rows)

    message = str(error.value)
    assert "1 duplicate (sequence_id, time_s) key(s)" in message
    assert "seq@1" in message


def test_acceleration_limit_rejects_numeric_equivalent_duplicate_timestamps() -> None:
    rows = _kink_submission()
    rows["time_s"] = rows["time_s"].astype(object)
    rows.loc[2, "time_s"] = "1"

    with pytest.raises(ValueError, match=r"seq@1"):
        repair_track5_acceleration_kinks(rows)


def test_acceleration_limit_allows_timestamp_reuse_across_sequences() -> None:
    rows = _kink_submission()
    rows.loc[2, "sequence_id"] = "other"
    rows.loc[2, "time_s"] = 1.0

    repaired, diagnostics = repair_track5_acceleration_kinks(rows)

    assert len(repaired) == len(rows)
    assert len(diagnostics) == len(rows)
    assert set(repaired["sequence_id"]) == {"seq", "other"}
