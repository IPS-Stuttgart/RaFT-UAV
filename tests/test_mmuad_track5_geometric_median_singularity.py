from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_geometric_median_ensemble import (
    build_track5_geometric_median_ensemble,
    weighted_geometric_median,
)


def _single_row_estimate(x_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [x_m],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )


def test_weighted_geometric_median_escapes_nonoptimal_input_point() -> None:
    points = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    weights = np.asarray([10.0, 1.0, 1.0])
    weighted_mean = np.sum(weights[:, None] * points, axis=0) / np.sum(weights)

    center, iterations, displacement = weighted_geometric_median(
        points,
        weights,
        max_iterations=128,
        tolerance_m=1.0e-10,
    )

    np.testing.assert_allclose(weighted_mean, [0.0, 0.0, 0.0], atol=0.0)
    np.testing.assert_allclose(center, [-1.0, 0.0, 0.0], atol=1.0e-7)
    assert iterations > 1
    assert displacement <= 1.0e-10

    objective_at_mean = np.sum(
        weights * np.linalg.norm(points - weighted_mean[None, :], axis=1)
    )
    objective_at_center = np.sum(
        weights * np.linalg.norm(points - center[None, :], axis=1)
    )
    assert objective_at_center < objective_at_mean


def test_track5_geomedian_uses_singularity_safe_solver() -> None:
    template = pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]})

    estimates, diagnostics = build_track5_geometric_median_ensemble(
        [
            ("left", _single_row_estimate(-1.0), 10.0),
            ("middle", _single_row_estimate(0.0), 1.0),
            ("right", _single_row_estimate(10.0), 1.0),
        ],
        template,
        max_iterations=128,
        tolerance_m=1.0e-10,
    )

    assert estimates.loc[0, "state_x_m"] == pytest.approx(-1.0, abs=1.0e-7)
    assert estimates.loc[0, "geomedian_source_count"] == 3
    assert diagnostics.loc[0, "geomedian_to_weighted_mean_m"] == pytest.approx(1.0)
