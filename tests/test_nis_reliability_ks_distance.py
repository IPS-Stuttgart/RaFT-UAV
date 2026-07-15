from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from raft_uav.diagnostics.nis_reliability import nis_reliability_summary


def test_nis_reliability_ks_distance_checks_both_sides_of_cdf_jump() -> None:
    frame = pd.DataFrame(
        {
            "source": ["rf"],
            "measurement_dim": [2],
            "nis": [10.0],
        }
    )

    report = nis_reliability_summary(frame, gate_probabilities=(0.95,))

    theoretical_cdf = float(chi2.cdf(10.0, df=2))
    expected = max(theoretical_cdf, 1.0 - theoretical_cdf)
    actual = float(report.iloc[0]["chi2_ks_distance"])
    assert np.isclose(actual, expected)
    assert actual > 0.99
