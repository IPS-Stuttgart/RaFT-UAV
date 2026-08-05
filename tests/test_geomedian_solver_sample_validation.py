from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.track5_geometric_median_ensemble import weighted_geometric_median


@pytest.mark.parametrize(
    ("xyz", "weights", "message"),
    [
        ([[True, 0.0, 0.0]], [1.0], "xyz must not contain Boolean values"),
        ([[1.0 + 0.0j, 0.0, 0.0]], [1.0], "xyz must not contain complex values"),
        ([[0.0, 0.0, 0.0]], [True], "weights must not contain Boolean values"),
        ([[0.0, 0.0, 0.0]], [1.0 + 0.0j], "weights must not contain complex values"),
    ],
)
def test_geomedian_solver_rejects_lossy_scalar_samples(
    xyz: object,
    weights: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        weighted_geometric_median(xyz, weights)


def test_geomedian_solver_rejects_masked_samples() -> None:
    xyz = np.ma.array(
        [[1.0, 2.0, 3.0]],
        mask=[[True, False, False]],
    )

    with pytest.raises(ValueError, match="xyz must not contain masked values"):
        weighted_geometric_median(xyz, [1.0])


def test_geomedian_solver_preserves_supported_scalar_like_and_missing_values() -> None:
    center, iterations, displacement = weighted_geometric_median(
        [
            [np.array(1.0), "2.0", np.float64(3.0)],
            [None, 20.0, 30.0],
        ],
        np.asarray([np.array(2.0), 1.0], dtype=object),
    )

    assert center.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert iterations == 0
    assert displacement == 0.0
