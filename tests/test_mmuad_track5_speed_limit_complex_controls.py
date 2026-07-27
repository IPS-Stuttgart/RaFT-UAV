from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _submission_rows() -> pd.DataFrame:
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


def _object_scalar(value: object) -> object:
    return np.asarray(value, dtype=object)


def _unmasked_object_scalar(value: object) -> object:
    return np.ma.array(np.asarray(value, dtype=object), mask=False)


@pytest.mark.parametrize(
    ("field", "real_part", "message"),
    [
        (
            "max_speed_mps",
            10.0,
            "max_speed_mps must be positive and finite",
        ),
        (
            "iterations",
            2.0,
            "iterations must be a positive integer",
        ),
        (
            "anchor_blend",
            0.25,
            r"anchor_blend must be finite and in \[0, 1\)",
        ),
    ],
)
@pytest.mark.parametrize("complex_type", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "wrap",
    [_object_scalar, _unmasked_object_scalar],
    ids=["object-array", "unmasked-object-array"],
)
def test_speed_limit_rejects_object_wrapped_complex_controls(
    field: str,
    real_part: float,
    message: str,
    complex_type: type[np.complexfloating],
    wrap: Callable[[object], object],
) -> None:
    kwargs: dict[str, object] = {
        "max_speed_mps": 10.0,
        "iterations": 2,
        "anchor_blend": 0.0,
    }
    kwargs[field] = wrap(complex_type(complex(real_part, 1.0)))

    with pytest.raises(ValueError, match=message):
        project_track5_speed_limit(_submission_rows(), **kwargs)


def test_speed_limit_accepts_object_wrapped_real_controls() -> None:
    limited, diagnostics = project_track5_speed_limit(
        _submission_rows(),
        max_speed_mps=np.asarray(np.float64(10.0), dtype=object),
        iterations=np.asarray(np.int64(2), dtype=object),
        anchor_blend=np.asarray(np.float64(0.0), dtype=object),
    )

    assert limited["state_x_m"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert diagnostics["output_speed_prev_mps"].dropna().max() <= 10.0
