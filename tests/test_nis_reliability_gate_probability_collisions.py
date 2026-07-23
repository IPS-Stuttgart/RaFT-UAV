from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.diagnostics.nis_reliability import nis_reliability_summary


def _diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["radar", "radar"],
            "measurement_dim": [3, 3],
            "nis": [1.0, 2.0],
        }
    )


def test_nis_reliability_rejects_colliding_probability_suffixes() -> None:
    with pytest.raises(
        ValueError,
        match=r"duplicate output column suffixes: 0p950: 0\.9501, 0\.9504",
    ):
        nis_reliability_summary(
            _diagnostics(),
            gate_probabilities=(0.9501, 0.9504),
        )


def test_nis_reliability_validates_probabilities_before_empty_return() -> None:
    empty = _diagnostics().iloc[0:0]

    with pytest.raises(ValueError, match="gate probability must be in \\(0, 1\\)"):
        nis_reliability_summary(
            empty,
            gate_probabilities=(float("nan"),),
        )
