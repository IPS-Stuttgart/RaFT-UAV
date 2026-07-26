from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "track_id": ["best", "second"],
            "x_m": [0.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.5],
        }
    )


@pytest.mark.parametrize(
    "radius_m",
    [
        True,
        np.bool_(False),
        -0.1,
        np.nan,
        np.inf,
        [1.0],
        1.0 + 0.0j,
        np.ma.masked,
    ],
)
def test_diversity_rejects_invalid_radius_controls(radius_m: object) -> None:
    with pytest.raises(
        ValueError,
        match="radius_m must be a finite non-negative number",
    ):
        diversify_candidate_reservoir(_rows(), radius_m=radius_m)


@pytest.mark.parametrize(
    "cap",
    [
        True,
        np.bool_(False),
        0,
        -1,
        1.5,
        np.nan,
        np.inf,
        [2],
        2.0 + 0.0j,
        np.ma.masked,
    ],
)
def test_diversity_rejects_invalid_frame_caps(cap: object) -> None:
    with pytest.raises(
        ValueError,
        match="max_candidates_per_frame must be a positive integer",
    ):
        diversify_candidate_reservoir(_rows(), max_candidates_per_frame=cap)


def test_diversity_accepts_valid_numpy_scalar_controls() -> None:
    output = diversify_candidate_reservoir(
        _rows(),
        radius_m=np.array(0.0),
        max_candidates_per_frame=np.float64(2.0),
    )

    assert output["track_id"].tolist() == ["best", "second"]
    assert output["candidate_diversity_radius_m"].eq(0.0).all()


def test_diversity_validates_controls_before_empty_input_return() -> None:
    with pytest.raises(ValueError, match="max_candidates_per_frame"):
        diversify_candidate_reservoir(pd.DataFrame(), max_candidates_per_frame=1.5)
