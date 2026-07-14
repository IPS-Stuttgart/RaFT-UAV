from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
)
from raft_uav.mmuad.candidate_pair_group_correction import (
    PairGroupMultiplicityConfig,
    prepare_group_corrected_pair_candidates,
)


def test_group_correction_resolves_score_fallbacks_per_candidate_row() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "source": ["candidate", "candidate"],
            "track_id": ["primary", "fallback"],
            "candidate_origin_row": ["origin-primary", "origin-fallback"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "primary_score": [10.0, None],
            "fallback_score": [0.0, 7.0],
            "predicted_sigma_m": [1.0, 1.0],
        }
    )
    prepared, _, _ = prepare_group_corrected_pair_candidates(
        candidates,
        pair_config=CandidatePairForwardBackwardConfig(
            score_column="primary_score",
            fallback_score_columns=("fallback_score",),
            score_normalization="rank",
            sigma_log_weight=0.0,
        ),
        group_config=PairGroupMultiplicityConfig(correction_strength=0.0),
    )
    normalized = prepared.set_index("track_id")[
        "candidate_pair_group_base_normalized_score"
    ]

    assert normalized.loc["primary"] == pytest.approx(1.0)
    assert normalized.loc["fallback"] == pytest.approx(0.0)
