from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_diversity import diversity_cap_reservoir


def _reservoir() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["s1", "s1", "s2"],
            "track_id": ["a", "b", "c"],
            "candidate_branch": ["raw", "translated", "raw"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
            "candidate_reservoir_score": [0.9, 0.8, 0.7],
            "confidence": [0.9, 0.8, 0.7],
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        2.5,
        float("nan"),
        np.asarray([2]),
        np.ma.masked,
    ],
)
def test_diversity_cap_rejects_malformed_max_candidate_counts(value: object) -> None:
    with pytest.raises(ValueError, match="max_candidates_per_frame must be an integer"):
        diversity_cap_reservoir(
            _reservoir(),
            max_candidates_per_frame=value,
        )


@pytest.mark.parametrize("field", ["min_per_source", "min_per_branch"])
@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        np.bool_(False),
        1.5,
        float("nan"),
        np.asarray([1]),
        np.ma.masked,
    ],
)
def test_diversity_cap_rejects_malformed_minimum_quotas(
    field: str,
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field} must be an integer greater than or equal to 0",
    ):
        diversity_cap_reservoir(
            _reservoir(),
            **{field: value},
        )


@pytest.mark.parametrize("value", ["2", 2.0, np.asarray(2)])
def test_diversity_cap_accepts_exact_scalar_like_candidate_counts(value: object) -> None:
    capped = diversity_cap_reservoir(
        _reservoir(),
        max_candidates_per_frame=value,
        min_per_source=0,
        min_per_branch=0,
    )

    assert len(capped) == 2


def test_diversity_cap_preserves_nonpositive_unbounded_semantics() -> None:
    capped = diversity_cap_reservoir(
        _reservoir(),
        max_candidates_per_frame=-1,
        min_per_source=0,
        min_per_branch=0,
    )

    assert len(capped) == len(_reservoir())
