from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from raft_uav.diagnostics.nis_reliability import nis_reliability_summary


def test_nis_reliability_summary_reports_chi_square_gate_rates() -> None:
    gate = float(chi2.ppf(0.95, df=3))
    frame = pd.DataFrame(
        {
            "source": ["radar", "radar", "radar"],
            "measurement_dim": [3, 3, 3],
            "nis": [1.0, 2.0, 10.0],
            "accepted": [True, "true", False],
            "gate_threshold": [gate, gate, gate],
        }
    )

    report = nis_reliability_summary(frame, gate_probabilities=(0.95,))

    assert len(report) == 1
    row = report.iloc[0]
    assert row["source"] == "radar"
    assert int(row["measurement_dim"]) == 3
    assert int(row["count"]) == 3
    assert int(row["accepted_count"]) == 2
    assert np.isclose(float(row["accepted_fraction"]), 2.0 / 3.0)
    assert np.isclose(float(row["actual_under_gate_0p950"]), 2.0 / 3.0)
    assert np.isfinite(float(row["chi2_ks_distance"]))


def test_nis_reliability_accepted_only_filters_rejected_updates() -> None:
    frame = pd.DataFrame(
        {
            "source": ["rf", "rf", "rf"],
            "measurement_dim": [2, 2, 2],
            "nis": [1.0, 2.0, 100.0],
            "accepted": [True, True, False],
        }
    )

    report = nis_reliability_summary(
        frame,
        gate_probabilities=(0.95,),
        accepted_only=True,
    )

    row = report.iloc[0]
    assert int(row["count"]) == 2
    assert float(row["nis_max"]) == 2.0
