from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import (
    resample_estimates_to_track5_template,
)


def _box(value: object) -> np.ndarray:
    boxed = np.empty((), dtype=object)
    boxed[()] = value
    return boxed


def _control(kind: str) -> object:
    if kind == "boolean":
        return _box(_box(True))
    if kind == "complex":
        return _box(_box(np.complex128(0.5 + 2.0j)))
    if kind == "vector":
        return _box(_box(np.asarray([0.5])))
    if kind == "cycle":
        cyclic = np.empty((), dtype=object)
        cyclic[()] = cyclic
        return cyclic
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "field",
    ["max_nearest_time_delta_s", "max_interpolation_gap_s"],
)
@pytest.mark.parametrize("kind", ["boolean", "complex", "vector", "cycle"])
def test_template_resample_rejects_recursively_boxed_controls(
    field: str,
    kind: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field} must be a finite non-negative number",
    ):
        resample_estimates_to_track5_template(
            pd.DataFrame(),
            pd.DataFrame(),
            **{field: _control(kind)},
        )


@pytest.mark.parametrize(
    "field",
    ["max_nearest_time_delta_s", "max_interpolation_gap_s"],
)
def test_template_resample_accepts_recursively_boxed_real_controls(field: str) -> None:
    resampled, diagnostics = resample_estimates_to_track5_template(
        pd.DataFrame(),
        pd.DataFrame(),
        **{field: _box(_box(np.float64(0.5)))},
    )

    assert resampled.empty
    assert diagnostics.empty
