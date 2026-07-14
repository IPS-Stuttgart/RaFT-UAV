from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_geometric_median_ensemble import (
    build_track5_geometric_median_ensemble,
    weighted_geometric_median,
)


def _coincident_start_case() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    weights = np.asarray([0.1, 2.0, 1.0], dtype=float)
    return points, weights


def test_weighted_geometric_median_escapes_nonoptimal_coincident_start() -> None:
    points, weights = _coincident_start_case()

    center, iterations, displacement = weighted_geometric_median(points, weights)

    assert center == pytest.approx([-1.0, 0.0, 0.0], abs=1.0e-3)
    assert iterations > 1
    assert displacement <= 1.0e-4


def test_track5_geometric_median_ensemble_uses_singularity_safe_solver() -> None:
    points, weights = _coincident_start_case()
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )
    inputs = []
    for index, (point, weight) in enumerate(zip(points, weights, strict=True)):
        inputs.append(
            (
                f"candidate_{index}",
                pd.DataFrame(
                    {
                        "sequence_id": ["seq0001"],
                        "time_s": [0.0],
                        "state_x_m": [point[0]],
                        "state_y_m": [point[1]],
                        "state_z_m": [point[2]],
                    }
                ),
                float(weight),
            )
        )

    estimates, diagnostics = build_track5_geometric_median_ensemble(inputs, template)

    assert estimates.loc[0, "state_x_m"] == pytest.approx(-1.0, abs=1.0e-3)
    assert estimates.loc[0, "state_y_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert estimates.loc[0, "state_z_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert diagnostics.loc[0, "valid_input_count"] == 3
