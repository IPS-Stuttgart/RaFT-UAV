from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [1.0] * 4,
            "source": ["a", "a", "b", "c"],
            "track_id": ["best", "duplicate", "protected", "far"],
            "x_m": [0.0, 0.1, 0.2, 5.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.9, 0.1, 0.5],
            "candidate_reservoir_protected": [False, False, True, False],
        }
    )


def test_diversity_suppresses_near_duplicate_and_keeps_far_candidate() -> None:
    output = diversify_candidate_reservoir(_rows(), radius_m=1.0)
    assert set(output["track_id"]) == {"best", "protected", "far"}
    assert "duplicate" not in set(output["track_id"])


def test_diversity_can_disable_protected_override() -> None:
    output = diversify_candidate_reservoir(
        _rows(), radius_m=1.0, preserve_protected=False
    )
    assert set(output["track_id"]) == {"best", "far"}


def test_diversity_respects_per_frame_cap() -> None:
    output = diversify_candidate_reservoir(_rows(), radius_m=0.0, max_candidates_per_frame=2)
    assert len(output) == 2
    assert output["candidate_diversity_rank"].tolist() == [1, 2]


def test_diversity_does_not_expand_duplicate_input_index_labels() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 3,
            "time_s": [1.0] * 3,
            "track_id": ["best", "duplicate", "far"],
            "x_m": [0.0, 0.1, 5.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.9, 0.5],
            "candidate_reservoir_protected": [False, False, False],
        },
        index=[7, 7, 8],
    )

    output = diversify_candidate_reservoir(
        rows,
        radius_m=1.0,
        max_candidates_per_frame=2,
    )

    assert output["track_id"].tolist() == ["best", "far"]
    assert len(output) == 2


@pytest.mark.parametrize(
    "radius_m",
    [True, np.bool_(False), -0.1, np.nan, np.inf, "not-a-number"],
)
def test_diversity_rejects_invalid_radius_controls(radius_m: object) -> None:
    with pytest.raises(ValueError, match="radius_m must be a finite non-negative number"):
        diversify_candidate_reservoir(_rows(), radius_m=radius_m)


@pytest.mark.parametrize(
    "cap",
    [True, np.bool_(False), 0, -1, 1.5, np.nan, np.inf, "not-an-integer"],
)
def test_diversity_rejects_invalid_frame_caps(cap: object) -> None:
    with pytest.raises(ValueError, match="max_candidates_per_frame must be a positive integer"):
        diversify_candidate_reservoir(_rows(), max_candidates_per_frame=cap)


def test_diversity_accepts_integer_equivalent_frame_cap() -> None:
    output = diversify_candidate_reservoir(
        _rows(),
        radius_m=np.float64(0.0),
        max_candidates_per_frame=2.0,
    )

    assert len(output) == 2
    assert output["candidate_diversity_radius_m"].eq(0.0).all()
