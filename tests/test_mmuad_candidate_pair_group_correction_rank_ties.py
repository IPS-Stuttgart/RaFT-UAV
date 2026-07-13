from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
)
from raft_uav.mmuad.candidate_pair_group_correction import (
    PairGroupMultiplicityConfig,
    attach_group_corrected_pair_prior,
)


def test_rank_ties_preserve_equal_physical_group_mass() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["raw", "translated", "dynamic"],
            "candidate_branch": ["raw", "translated", "dynamic"],
            "track_id": ["duplicate-a", "duplicate-b", "singleton"],
            "candidate_origin_row": ["physical-a", "physical-a", "physical-b"],
            "x_m": [0.0, 0.1, 10.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
            "ranker_score": [0.0, 0.0, 0.0],
            "predicted_sigma_m": [1.0, 1.0, 1.0],
            "confidence": [1.0, 1.0, 1.0],
        }
    )
    pair_config = CandidatePairForwardBackwardConfig(
        score_column="ranker_score",
        fallback_score_columns=(),
        sigma_column="predicted_sigma_m",
        score_normalization="rank",
        score_weight=1.0,
        sigma_log_weight=0.0,
        output_score_column="test_pair_score",
    )

    augmented, _, _ = attach_group_corrected_pair_prior(
        candidates,
        pair_config=pair_config,
        group_config=PairGroupMultiplicityConfig(correction_strength=1.0),
    )

    rows = augmented.rows
    normalized = rows["candidate_pair_group_base_normalized_score"]
    group_mass = rows.groupby("candidate_origin_row")["test_pair_score"].sum()

    assert normalized.to_numpy(float) == pytest.approx([0.5, 0.5, 0.5])
    assert group_mass["physical-a"] == pytest.approx(0.5)
    assert group_mass["physical-b"] == pytest.approx(0.5)
