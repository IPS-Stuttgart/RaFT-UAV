from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_grid import (
    run_candidate_reservoir_offset_grid,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq-a"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "source": ["radar"],
            "candidate_branch": ["raw"],
            "ranker_score": [1.0],
            "confidence": [1.0],
        }
    )


@pytest.mark.parametrize(
    "top_k_values",
    [
        [True],
        [1.5],
        [np.nan],
        [np.inf],
        [np.array([3])],
        [0],
        [-1],
        "13",
    ],
)
def test_reservoir_grid_rejects_lossy_top_k_values(top_k_values: object) -> None:
    with pytest.raises(ValueError, match="top_k_values"):
        run_candidate_reservoir_offset_grid(
            _candidate_rows(),
            top_k_values=top_k_values,
        )


def test_reservoir_grid_accepts_exact_scalar_like_top_k_values() -> None:
    summary, best = run_candidate_reservoir_offset_grid(
        _candidate_rows(),
        top_k_values=[np.int64(1), 3.0, "5"],
        write_best_reservoir=True,
    )

    assert summary["grid_label"].tolist() == ["identity"]
    assert best is not None
    assert len(best) == 1
