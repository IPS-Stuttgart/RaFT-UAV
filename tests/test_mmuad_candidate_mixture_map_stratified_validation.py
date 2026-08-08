from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map_stratified import (
    StratifiedMixtureTopKConfig,
    select_stratified_mixture_candidates,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "candidate_branch": ["raw", "translated"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "ranker_score": [0.5, 0.4],
            "predicted_sigma_m": [1.0, 1.0],
        }
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top_k", 1.5, "top_k must be a positive integer"),
        ("top_k", True, "top_k must be a positive integer"),
        (
            "min_per_branch",
            0.5,
            "min_per_branch must be a non-negative integer",
        ),
        (
            "min_per_source",
            -0.5,
            "min_per_source must be a non-negative integer",
        ),
        (
            "min_per_source_branch",
            np.array([1]),
            "min_per_source_branch must be a non-negative integer",
        ),
    ],
)
def test_stratified_topk_rejects_non_integer_quota_values(
    field: str,
    value: object,
    message: str,
) -> None:
    config = StratifiedMixtureTopKConfig(**{field: value})

    with pytest.raises(ValueError, match=message):
        select_stratified_mixture_candidates(_candidate_rows(), config=config)
