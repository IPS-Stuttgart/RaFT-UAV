from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_diversity import diversity_cap_reservoir


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * 2,
            "time_s": [0.0, 0.0],
            "source": ["s1", "s2"],
            "track_id": ["high", "low"],
            "candidate_branch": ["translated", "raw"],
            "x_m": [1.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "candidate_reservoir_score": [0.9, 0.1],
            "confidence": [0.9, 0.1],
        }
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_candidates_per_frame", True),
        ("max_candidates_per_frame", 1.5),
        ("max_candidates_per_frame", -1),
        ("max_candidates_per_frame", np.nan),
        ("min_per_source", np.bool_(False)),
        ("min_per_source", np.ma.masked),
        ("min_per_source", np.ma.array(1, mask=True)),
        ("min_per_branch", np.array([1])),
        ("min_per_branch", np.inf),
        ("min_per_branch", "not-an-integer"),
    ],
)
def test_diversity_cap_rejects_invalid_integer_controls(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be a non-negative integer"):
        diversity_cap_reservoir(_rows(), **{name: value})


def test_diversity_cap_validates_controls_before_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="max_candidates_per_frame must be a non-negative integer",
    ):
        diversity_cap_reservoir(
            pd.DataFrame(),
            max_candidates_per_frame=2.5,
        )


def test_diversity_cap_accepts_exact_integer_equivalent_controls() -> None:
    capped = diversity_cap_reservoir(
        _rows(),
        max_candidates_per_frame=np.array(0),
        min_per_source="0",
        min_per_branch=np.float64(0.0),
    )

    assert len(capped) == 2


def test_diversity_cap_zero_disables_only_the_requested_quota() -> None:
    capped = diversity_cap_reservoir(
        _rows(),
        max_candidates_per_frame=1,
        min_per_source=0,
        min_per_branch=0,
    )

    assert capped["track_id"].tolist() == ["high"]
