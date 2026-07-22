from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.factor_graph import coordinate_descent_association_and_smoothing


def _radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [0],
            "track_id": [1],
            "time_s": [0.0],
            "east_m": [1.0],
            "north_m": [2.0],
            "up_m": [3.0],
        }
    )


@pytest.mark.parametrize(
    "candidate_gate_m",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([10.0]),
        np.ma.masked,
    ],
)
def test_coordinate_descent_rejects_invalid_candidate_gate(
    candidate_gate_m: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="candidate_gate_m must be a finite non-negative real scalar",
    ):
        coordinate_descent_association_and_smoothing(
            _radar(),
            iterations=0,
            candidate_gate_m=candidate_gate_m,
        )


@pytest.mark.parametrize(
    "iterations",
    [
        -1,
        1.5,
        np.nan,
        np.inf,
        True,
        1.0 + 0.0j,
        np.array([1]),
        np.ma.masked,
    ],
)
def test_coordinate_descent_rejects_malformed_iteration_counts(
    iterations: object,
) -> None:
    with pytest.raises(ValueError, match="iterations must be a non-negative integer"):
        coordinate_descent_association_and_smoothing(
            _radar(),
            iterations=iterations,
        )


def test_coordinate_descent_accepts_zero_dimensional_scalar_controls() -> None:
    trajectory, selected = coordinate_descent_association_and_smoothing(
        _radar(),
        iterations=np.array(1.0),
        candidate_gate_m=np.array(0.0),
    )

    assert len(trajectory) == 1
    assert selected["track_id"].tolist() == [1]
