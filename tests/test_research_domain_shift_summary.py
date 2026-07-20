from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.diagnostics import domain_shift_summary


EXPECTED_COLUMNS = [
    "feature",
    "train_count",
    "heldout_count",
    "train_mean",
    "heldout_mean",
    "mean_shift_z",
    "train_p50",
    "heldout_p50",
    "train_p90",
    "heldout_p90",
    "ks_distance",
]


def test_domain_shift_summary_returns_empty_schema_without_comparable_features() -> None:
    training = pd.DataFrame({"label": ["train"]})
    heldout = pd.DataFrame({"label": ["heldout"]})

    summary = domain_shift_summary(training, heldout)

    assert summary.empty
    assert list(summary.columns) == EXPECTED_COLUMNS


def test_domain_shift_summary_ignores_nonfinite_observations() -> None:
    training = pd.DataFrame({"signal": [0.0, 1.0, np.inf]})
    heldout = pd.DataFrame({"signal": [1.0, 2.0, -np.inf]})

    summary = domain_shift_summary(training, heldout)

    assert summary.loc[0, "feature"] == "signal"
    assert summary.loc[0, "train_count"] == 2
    assert summary.loc[0, "heldout_count"] == 2
    assert np.isclose(summary.loc[0, "train_mean"], 0.5)
    assert np.isclose(summary.loc[0, "heldout_mean"], 1.5)
    assert np.isfinite(summary.loc[0, "mean_shift_z"])
    assert np.isfinite(summary.loc[0, "ks_distance"])
