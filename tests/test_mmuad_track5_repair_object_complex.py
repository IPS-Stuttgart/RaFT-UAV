from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import (
    repair_track5_acceleration_kinks,
)
from raft_uav.mmuad.track5_hampel_repair import repair_track5_hampel_spikes
from raft_uav.mmuad.track5_jerk_limit import repair_track5_jerk_kinks
from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 4,
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "state_x_m": [0.0, 1.0, 8.0, 3.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
            "Classification": [2, 2, 2, 2],
        }
    )


def _object_scalar(value: object) -> object:
    return np.asarray(value, dtype=object)


def _unmasked_object_scalar(value: object) -> object:
    return np.ma.array(np.asarray(value, dtype=object), mask=False)


_REPAIRS: tuple[Callable[..., object], ...] = (
    repair_track5_acceleration_kinks,
    repair_track5_hampel_spikes,
    repair_track5_jerk_kinks,
    repair_track5_vertical_spikes,
)


@pytest.mark.parametrize(
    "repair",
    _REPAIRS,
    ids=("acceleration", "hampel", "jerk", "vertical"),
)
@pytest.mark.parametrize(
    "wrap",
    (_object_scalar, _unmasked_object_scalar),
    ids=("object-array", "unmasked-object-array"),
)
def test_track5_repairs_reject_object_wrapped_complex_coordinate_cells(
    repair: Callable[..., object],
    wrap: Callable[[object], object],
) -> None:
    rows = _submission()
    rows["state_x_m"] = rows["state_x_m"].astype(object)
    rows.at[1, "state_x_m"] = wrap(np.complex64(1.0 + 2.0j))

    with pytest.raises(
        ValueError,
        match=r"complex numeric values: state_x_m rows \[1\]",
    ):
        repair(rows)


_CONTROL_CASES = (
    pytest.param(
        repair_track5_acceleration_kinks,
        "max_acceleration_mps2",
        5.0,
        id="acceleration-max-acceleration",
    ),
    pytest.param(
        repair_track5_acceleration_kinks,
        "max_direct_speed_mps",
        20.0,
        id="acceleration-max-direct-speed",
    ),
    pytest.param(
        repair_track5_acceleration_kinks,
        "min_interpolation_residual_m",
        1.0,
        id="acceleration-min-residual",
    ),
    pytest.param(
        repair_track5_acceleration_kinks,
        "iterations",
        2.0,
        id="acceleration-iterations",
    ),
    pytest.param(
        repair_track5_acceleration_kinks,
        "repair_blend",
        0.5,
        id="acceleration-repair-blend",
    ),
    pytest.param(
        repair_track5_jerk_kinks,
        "max_jerk_mps3",
        80.0,
        id="jerk-max-jerk",
    ),
    pytest.param(
        repair_track5_jerk_kinks,
        "smoothness_weight",
        10.0,
        id="jerk-smoothness",
    ),
    pytest.param(
        repair_track5_jerk_kinks,
        "min_correction_m",
        1.0,
        id="jerk-min-correction",
    ),
    pytest.param(
        repair_track5_jerk_kinks,
        "max_correction_m",
        5.0,
        id="jerk-max-correction",
    ),
    pytest.param(
        repair_track5_jerk_kinks,
        "iterations",
        2.0,
        id="jerk-iterations",
    ),
    pytest.param(
        repair_track5_jerk_kinks,
        "repair_blend",
        0.5,
        id="jerk-repair-blend",
    ),
)


@pytest.mark.parametrize(("repair", "field", "real_part"), _CONTROL_CASES)
@pytest.mark.parametrize("complex_type", (complex, np.complex64, np.complex128))
@pytest.mark.parametrize(
    "wrap",
    (_object_scalar, _unmasked_object_scalar),
    ids=("object-array", "unmasked-object-array"),
)
def test_track5_repairs_reject_object_wrapped_complex_controls(
    repair: Callable[..., object],
    field: str,
    real_part: float,
    complex_type: Callable[[complex], object],
    wrap: Callable[[object], object],
) -> None:
    value = wrap(complex_type(complex(real_part, 1.0)))

    with pytest.raises(ValueError, match=field):
        repair(_submission(), **{field: value})


def test_track5_repairs_accept_object_wrapped_real_controls() -> None:
    acceleration, acceleration_diagnostics = repair_track5_acceleration_kinks(
        _submission(),
        max_acceleration_mps2=_object_scalar(np.float64(5.0)),
        max_direct_speed_mps=_object_scalar(np.float64(20.0)),
        min_interpolation_residual_m=_object_scalar(np.float64(1.0)),
        iterations=_object_scalar(np.int64(2)),
        repair_blend=_object_scalar(np.float64(0.5)),
    )
    jerk, jerk_diagnostics = repair_track5_jerk_kinks(
        _submission(),
        max_jerk_mps3=_object_scalar(np.float64(80.0)),
        smoothness_weight=_object_scalar(np.float64(10.0)),
        min_correction_m=_object_scalar(np.float64(1.0)),
        max_correction_m=_object_scalar(np.float64(5.0)),
        iterations=_object_scalar(np.int64(2)),
        repair_blend=_object_scalar(np.float64(0.5)),
    )

    assert len(acceleration) == len(acceleration_diagnostics) == 4
    assert len(jerk) == len(jerk_diagnostics) == 4
