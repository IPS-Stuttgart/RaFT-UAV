from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    CandidateMixtureMapResult,
)
from raft_uav.mmuad.candidate_mixture_map_multistart import (
    compute_candidate_mixture_selection_objective,
)


def _result() -> CandidateMixtureMapResult:
    return CandidateMixtureMapResult(
        estimates=pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqA"],
                "time_s": [0.0, 1.0],
                "state_x_m": [0.0, 1.0],
                "state_y_m": [0.0, 0.0],
                "state_z_m": [0.0, 0.0],
            }
        ),
        assignments=pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqA"],
                "time_s": [0.0, 1.0],
                "mixture_log_weight": [0.0, 0.0],
            }
        ),
        iteration_summary=pd.DataFrame(),
        summary={},
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "source": ["radar", "radar"],
            "track_id": ["track", "track"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [1.0, 1.0],
            "predicted_sigma_m": [1.0, 1.0],
        }
    )


def _initial_estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "state_x_m": [10.0, 11.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )


def test_selection_objective_includes_restart_specific_anchor_penalty() -> None:
    objective = compute_candidate_mixture_selection_objective(
        _result(),
        mixture_config=CandidateMixtureMapConfig(
            smoothness_weight=0.0,
            anchor_weight=2.0,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
        ),
        candidates=_candidates(),
        initial_estimates=_initial_estimates(),
    )

    assert objective["mixture_data_nll"] == pytest.approx(0.0)
    assert objective["smoothness_penalty"] == pytest.approx(0.0)
    assert objective["anchor_penalty"] == pytest.approx(400.0)
    assert objective["selection_objective"] == pytest.approx(400.0)


def test_anchor_aware_objective_requires_candidates_for_nonzero_anchor() -> None:
    with pytest.raises(ValueError, match="candidates are required"):
        compute_candidate_mixture_selection_objective(
            _result(),
            mixture_config=CandidateMixtureMapConfig(
                smoothness_weight=0.0,
                anchor_weight=1.0,
            ),
        )


def test_zero_anchor_keeps_context_free_objective_compatible() -> None:
    objective = compute_candidate_mixture_selection_objective(
        _result(),
        mixture_config=CandidateMixtureMapConfig(
            smoothness_weight=0.0,
            anchor_weight=0.0,
        ),
    )

    assert objective["anchor_penalty"] == pytest.approx(0.0)
    assert objective["selection_objective"] == pytest.approx(0.0)
