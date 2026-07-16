from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_hampel_repair import repair_track5_hampel_spikes


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 100.0, 3.0, 4.0],
            "state_y_m": [0.0] * 5,
            "state_z_m": [5.0, 5.0, 50.0, 5.0, 5.0],
            "Classification": [2] * 5,
        }
    )


@pytest.mark.parametrize("field", ["window_radius", "iterations"])
@pytest.mark.parametrize(
    "bad_value",
    [0, -1, 1.5, True, np.bool_(True), np.nan, np.inf, -np.inf, pd.NA, np.array([1])],
)
def test_hampel_repair_rejects_invalid_integer_controls(
    field: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        repair_track5_hampel_spikes(_submission_rows(), **{field: bad_value})


@pytest.mark.parametrize("field", ["window_radius", "iterations"])
@pytest.mark.parametrize(
    "value",
    [1, 1.0, np.int64(1), np.float64(1.0), np.array(1)],
)
def test_hampel_repair_accepts_integer_equivalent_controls(
    field: str,
    value: object,
) -> None:
    repaired, diagnostics = repair_track5_hampel_spikes(
        _submission_rows(),
        **{field: value},
    )

    assert len(repaired) == len(_submission_rows())
    assert len(diagnostics) == len(_submission_rows())


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("sigma_threshold", True),
        ("sigma_threshold", np.bool_(True)),
        ("min_scale_m", True),
        ("min_residual_m", True),
        ("repair_blend", True),
        ("repair_blend", np.array(False)),
        ("sigma_threshold", np.array([3.0])),
        ("min_scale_m", np.array([1.0])),
        ("min_residual_m", np.array([3.0])),
        ("repair_blend", np.array([0.5])),
        ("sigma_threshold", pd.NA),
        ("min_scale_m", np.inf),
        ("min_residual_m", np.nan),
        ("repair_blend", -np.inf),
    ],
)
def test_hampel_repair_rejects_malformed_float_controls(
    field: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        repair_track5_hampel_spikes(_submission_rows(), **{field: bad_value})


def test_hampel_repair_accepts_numeric_scalar_like_float_controls() -> None:
    repaired, diagnostics = repair_track5_hampel_spikes(
        _submission_rows(),
        sigma_threshold=np.array(3.0),
        min_scale_m=np.float64(1.0),
        min_residual_m=np.int64(3),
        repair_blend=np.array(0.5),
    )

    assert len(repaired) == len(_submission_rows())
    assert len(diagnostics) == len(_submission_rows())
