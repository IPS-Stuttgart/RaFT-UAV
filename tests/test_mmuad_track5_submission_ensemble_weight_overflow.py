from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_submission_ensemble import ensemble_track5_submissions
from raft_uav.mmuad.track5_submission_ensemble import parse_submission_input


def _write_submission(
    path: Path,
    *,
    x_m: float,
    classification: int,
) -> None:
    pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": [f"({x_m}, 0.0, 0.0)"],
            "Classification": [classification],
        }
    ).to_csv(path, index=False)


def test_track5_submission_ensemble_scales_large_finite_weights(
    tmp_path: Path,
) -> None:
    specs = (
        ("class1_a", 9.0e307, 0.0, 1),
        ("class1_b", 9.0e307, 0.0, 1),
        ("class2_a", 1.0e308, 2.0, 2),
        ("class2_b", 1.0e308, 2.0, 2),
    )
    inputs = []
    for label, weight, x_m, classification in specs:
        path = tmp_path / f"{label}.csv"
        _write_submission(path, x_m=x_m, classification=classification)
        inputs.append(parse_submission_input(f"{label}={weight}:{path}"))

    estimates, diagnostics = ensemble_track5_submissions(inputs)

    estimate = estimates.iloc[0]
    diagnostic = diagnostics.iloc[0]
    assert np.isfinite(
        estimate[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(dtype=float)
    ).all()
    assert estimate["state_x_m"] == pytest.approx(20.0 / 19.0)
    assert estimate["Classification"] == 2
    assert np.isfinite(float(diagnostic["position_spread_m"]))
    assert diagnostic["classification_vote_margin"] == pytest.approx(2.0e307)
