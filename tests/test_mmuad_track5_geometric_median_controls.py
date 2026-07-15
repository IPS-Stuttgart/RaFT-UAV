from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_geometric_median_ensemble import (
    build_track5_geometric_median_ensemble,
    weighted_geometric_median,
)

POINTS = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ]
)
WEIGHTS = np.asarray([10.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "value",
    [0, -1, 1.5, True, np.nan, np.inf, np.array([4])],
)
def test_weighted_geometric_median_rejects_invalid_max_iterations(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        weighted_geometric_median(POINTS, WEIGHTS, max_iterations=value)


@pytest.mark.parametrize(
    "value",
    [-1.0, True, np.nan, np.inf, -np.inf, np.array([1.0e-4])],
)
def test_weighted_geometric_median_rejects_invalid_tolerance(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="tolerance_m must be a finite non-negative scalar",
    ):
        weighted_geometric_median(POINTS, WEIGHTS, tolerance_m=value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iterations": 0}, "max_iterations must be a positive integer"),
        ({"tolerance_m": np.inf}, "tolerance_m must be a finite non-negative scalar"),
    ],
)
def test_geomedian_build_validates_controls_before_empty_template(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_track5_geometric_median_ensemble(
            [],
            pd.DataFrame(),
            **kwargs,
        )


def test_weighted_geometric_median_accepts_numpy_scalar_controls() -> None:
    center, iterations, displacement = weighted_geometric_median(
        POINTS,
        WEIGHTS,
        max_iterations=np.array(128),
        tolerance_m=np.float64(1.0e-10),
    )

    np.testing.assert_allclose(center, [-1.0, 0.0, 0.0], atol=1.0e-7)
    assert 1 < iterations <= 128
    assert displacement <= 1.0e-10


def test_weighted_geometric_median_accepts_zero_tolerance() -> None:
    center, iterations, displacement = weighted_geometric_median(
        POINTS,
        WEIGHTS,
        max_iterations=4,
        tolerance_m=0.0,
    )

    assert np.isfinite(center).all()
    assert 1 <= iterations <= 4
    assert np.isfinite(displacement)
