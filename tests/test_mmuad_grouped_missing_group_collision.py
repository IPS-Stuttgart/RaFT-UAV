from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    HypothesisGroupConfig,
    compute_grouped_candidate_responsibilities,
    prepare_hypothesis_group_candidates,
)


def _collision_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq", "seq", "seq"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["a", "b", "c"],
            "track_id": ["explicit-a", "explicit-b", "missing"],
            "candidate_branch": ["raw", "calibrated", "raw"],
            "candidate_origin_row": ["row:2", "row:2", np.nan],
            "x_m": [0.0, 0.0, 0.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [1.0, 1.0, 1.0],
            "predicted_sigma_m": [1.0, 1.0, 1.0],
        }
    )


def _neutral_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        fallback_score_columns=(),
        score_normalization="none",
        score_weight=0.0,
        sigma_log_weight=0.0,
        loss="squared",
        smoothness_weight=0.0,
        iterations=1,
    )


def test_missing_group_label_does_not_collide_with_explicit_row_label() -> None:
    prepared, _, summary = prepare_hypothesis_group_candidates(
        _collision_candidates(),
        mixture_config=_neutral_config(),
        group_config=HypothesisGroupConfig(
            group_column="candidate_origin_row",
            correction_strength=1.0,
            missing_group_policy="unique",
        ),
    )

    explicit = prepared.loc[
        prepared["track_id"].isin(["explicit-a", "explicit-b"])
    ]
    missing = prepared.loc[prepared["track_id"] == "missing"].iloc[0]

    assert explicit["mixture_hypothesis_group"].nunique() == 1
    assert explicit["mixture_hypothesis_group_size"].tolist() == [2, 2]
    assert missing["mixture_hypothesis_group"] != "row:2"
    assert missing["mixture_hypothesis_group_size"] == 1
    assert summary["duplicate_hypothesis_group_count"] == 1


def test_collision_safe_missing_group_preserves_equal_group_mass() -> None:
    responsibilities = compute_grouped_candidate_responsibilities(
        _collision_candidates(),
        np.zeros(3),
        mixture_config=_neutral_config(),
        group_config=HypothesisGroupConfig(
            group_column="candidate_origin_row",
            correction_strength=1.0,
            missing_group_policy="unique",
        ),
    )

    group_mass = responsibilities.groupby("mixture_hypothesis_group")[
        "mixture_responsibility"
    ].sum()
    assert sorted(group_mass.to_numpy(float)) == pytest.approx([0.5, 0.5])
