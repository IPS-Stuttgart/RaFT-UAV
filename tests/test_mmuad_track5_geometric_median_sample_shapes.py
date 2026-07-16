from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.track5_geometric_median_ensemble import weighted_geometric_median


def test_weighted_geometric_median_accepts_plain_empty_sequences() -> None:
    center, iterations, displacement = weighted_geometric_median([], [])

    assert center.shape == (3,)
    assert np.isnan(center).all()
    assert iterations == 0
    assert np.isnan(displacement)


@pytest.mark.parametrize(
    ("xyz", "weights", "message"),
    [
        (
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0]),
            r"xyz must be a numeric 2D array with shape \(n, 3\)",
        ),
        (
            np.zeros((2, 2)),
            np.ones(2),
            r"xyz must be a numeric 2D array with shape \(n, 3\)",
        ),
        (
            np.zeros((2, 3)),
            np.ones((2, 1)),
            "weights must be a numeric 1D array",
        ),
        (
            np.zeros((2, 3)),
            np.ones(1),
            "xyz and weights must have the same row count",
        ),
    ],
)
def test_weighted_geometric_median_rejects_malformed_sample_shapes(
    xyz: np.ndarray,
    weights: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        weighted_geometric_median(xyz, weights)


def test_weighted_geometric_median_rejects_ragged_points_cleanly() -> None:
    with pytest.raises(
        ValueError,
        match=r"xyz must be a numeric 2D array with shape \(n, 3\)",
    ):
        weighted_geometric_median([[0.0, 0.0, 0.0], [1.0]], [1.0, 1.0])
