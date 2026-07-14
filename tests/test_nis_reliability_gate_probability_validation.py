from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.nis_reliability import nis_reliability_summary


def _nis_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["rf", "rf"],
            "measurement_dim": [2, 2],
            "nis": [1.0, 2.0],
        }
    )


@pytest.mark.parametrize(
    "probability",
    [np.nan, np.inf, -np.inf, 0.0, 1.0, True, "not-a-probability"],
)
def test_invalid_gate_probabilities_fail_even_for_empty_frames(
    probability: object,
) -> None:
    empty = pd.DataFrame(columns=["nis"])

    with pytest.raises(ValueError, match="gate_probabilities"):
        nis_reliability_summary(
            empty,
            gate_probabilities=[probability],
        )


def test_gate_probabilities_reject_rounded_column_suffix_collisions() -> None:
    with pytest.raises(ValueError, match="same output column suffix"):
        nis_reliability_summary(
            _nis_rows(),
            gate_probabilities=[0.9501, 0.9502],
        )


def test_exact_duplicate_gate_probabilities_are_deduplicated() -> None:
    summary = nis_reliability_summary(
        _nis_rows(),
        gate_probabilities=[0.95, 0.95, 0.99],
    )

    assert len(summary) == 1
    assert "gate_threshold_0p950" in summary.columns
    assert "gate_threshold_0p990" in summary.columns
    assert len([column for column in summary if column == "gate_threshold_0p950"]) == 1
