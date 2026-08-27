from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.diagnostics import domain_shift_summary


def test_domain_shift_summary_keeps_large_representable_statistics_finite() -> None:
    training = pd.DataFrame(
        {
            "positive": [8.0e307, 1.0e308],
            "signed": [-1.0e308, 1.0e308],
        }
    )
    heldout = pd.DataFrame(
        {
            "positive": [9.0e307, 1.1e308],
            "signed": [-8.0e307, 8.0e307],
        }
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        summary = domain_shift_summary(training, heldout).set_index("feature")

    numeric = summary[
        [
            "train_mean",
            "heldout_mean",
            "mean_shift_z",
            "train_p50",
            "heldout_p50",
            "train_p90",
            "heldout_p90",
            "ks_distance",
        ]
    ].to_numpy(dtype=float)
    assert np.isfinite(numeric).all()

    np.testing.assert_allclose(
        summary.loc["positive", "train_mean"],
        9.0e307,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        summary.loc["positive", "heldout_mean"],
        1.0e308,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        summary.loc["positive", "mean_shift_z"],
        1.0,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        summary.loc["positive", "train_p90"],
        9.8e307,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        summary.loc["positive", "heldout_p90"],
        1.08e308,
        rtol=1.0e-12,
    )

    np.testing.assert_allclose(summary.loc["signed", "train_mean"], 0.0, atol=0.0)
    np.testing.assert_allclose(summary.loc["signed", "heldout_mean"], 0.0, atol=0.0)
    np.testing.assert_allclose(summary.loc["signed", "train_p50"], 0.0, atol=0.0)
    np.testing.assert_allclose(summary.loc["signed", "heldout_p50"], 0.0, atol=0.0)
    np.testing.assert_allclose(
        summary.loc["signed", "train_p90"],
        8.0e307,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        summary.loc["signed", "heldout_p90"],
        6.4e307,
        rtol=1.0e-12,
    )
