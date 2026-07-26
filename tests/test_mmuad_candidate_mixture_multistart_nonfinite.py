from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad import candidate_mixture_map_multistart as multistart
from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    CandidateMixtureMapResult,
)


def _result(
    *,
    state_x_m: list[float] | None = None,
    log_weights: list[float] | None = None,
) -> CandidateMixtureMapResult:
    state_x_m = state_x_m or [0.0, 1.0, 2.0]
    log_weights = log_weights or [0.0, 0.0, 0.0]
    return CandidateMixtureMapResult(
        estimates=pd.DataFrame(
            {
                "sequence_id": ["seqA"] * 3,
                "time_s": [0.0, 1.0, 2.0],
                "state_x_m": state_x_m,
                "state_y_m": [0.0, 0.0, 0.0],
                "state_z_m": [0.0, 0.0, 0.0],
            }
        ),
        assignments=pd.DataFrame(
            {
                "sequence_id": ["seqA"] * 3,
                "time_s": [0.0, 1.0, 2.0],
                "mixture_log_weight": log_weights,
            }
        ),
        iteration_summary=pd.DataFrame(),
        summary={},
    )


def test_selection_objective_rejects_nonfinite_mixture_weights() -> None:
    objective = multistart.compute_candidate_mixture_selection_objective(
        _result(log_weights=[0.0, float("nan"), 0.0]),
        mixture_config=CandidateMixtureMapConfig(smoothness_weight=0.0),
    )

    assert all(np.isinf(value) for value in objective.values())


def test_selection_objective_rejects_nonfinite_estimates() -> None:
    objective = multistart.compute_candidate_mixture_selection_objective(
        _result(state_x_m=[0.0, float("nan"), 2.0]),
        mixture_config=CandidateMixtureMapConfig(smoothness_weight=100.0),
    )

    assert all(np.isinf(value) for value in objective.values())


def test_selection_objective_keeps_finite_results() -> None:
    objective = multistart.compute_candidate_mixture_selection_objective(
        _result(),
        mixture_config=CandidateMixtureMapConfig(smoothness_weight=0.0),
    )

    assert objective == {
        "selection_objective": 0.0,
        "mixture_data_nll": 0.0,
        "smoothness_penalty": 0.0,
        "anchor_penalty": 0.0,
    }


def test_multistart_rejects_when_every_restart_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid = _result(log_weights=[0.0, float("inf"), 0.0])
    monkeypatch.setattr(
        multistart._IMPL,
        "build_candidate_mixture_initializations",
        lambda *args, **kwargs: {"invalid": None},
    )
    monkeypatch.setattr(
        multistart.core,
        "run_candidate_mixture_map",
        lambda *args, **kwargs: invalid,
    )

    with pytest.raises(ValueError, match="no finite restart objective"):
        multistart.run_multistart_candidate_mixture_map(pd.DataFrame())
