from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward_agreement_cv import (
    aggregate_agreement_pair_cv_folds,
)


def _fold_row(grid_label: str, sequence_id: str, value: float) -> dict[str, object]:
    return {
        "grid_label": grid_label,
        "holdout_sequence_id": sequence_id,
        "min_pair_weight": 0.0,
        "max_pair_weight": 1.0,
        "entropy_power": 1.0,
        "agreement_power": 1.0,
        "agreement_floor": 0.0,
        "mse_3d_m": value,
    }


def test_duplicate_holdout_rows_cannot_hide_a_missing_sequence() -> None:
    rows = pd.DataFrame(
        [
            _fold_row("duplicate", "seqA", 0.1),
            _fold_row("duplicate", "seqA", 0.2),
            _fold_row("duplicate", "seqB", 0.3),
            _fold_row("complete", "seqA", 1.0),
            _fold_row("complete", "seqB", 1.0),
            _fold_row("complete", "seqC", 1.0),
        ]
    )

    aggregate = aggregate_agreement_pair_cv_folds(
        rows,
        expected_sequence_count=3,
    ).set_index("grid_label")

    assert not bool(aggregate.loc["duplicate", "eligible"])
    assert aggregate.loc["duplicate", "valid_fold_count"] == 3
    assert aggregate.loc["duplicate", "valid_holdout_sequence_count"] == 2
    assert aggregate.loc["duplicate", "duplicate_fold_count"] == 1
    assert np.isinf(aggregate.loc["duplicate", "risk_score"])
    assert bool(aggregate.loc["complete", "eligible"])


def test_fold_aggregation_requires_holdout_sequence_ids() -> None:
    rows = pd.DataFrame([_fold_row("grid", "seqA", 1.0)]).drop(
        columns=["holdout_sequence_id"]
    )

    with pytest.raises(ValueError, match="holdout_sequence_id"):
        aggregate_agreement_pair_cv_folds(rows, expected_sequence_count=1)
