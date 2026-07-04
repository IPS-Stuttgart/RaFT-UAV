from __future__ import annotations

import pandas as pd

from raft_uav.diagnostics.nis_reliability import nis_reliability_summary


def test_nis_reliability_drops_noninteger_measurement_dim() -> None:
    frame = pd.DataFrame(
        {
            "source": ["radar"],
            "measurement_dim": [2.9],
            "nis": [1.0],
        }
    )

    summary = nis_reliability_summary(frame)

    assert summary.empty


def test_nis_reliability_keeps_integer_like_measurement_dim() -> None:
    frame = pd.DataFrame(
        {
            "source": ["radar", "radar"],
            "measurement_dim": [3.0, "3"],
            "nis": [1.0, 2.0],
        }
    )

    summary = nis_reliability_summary(frame, gate_probabilities=(0.95,))

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["count"] == 2
    assert row["chi2_mean_expected"] == 3.0
    assert "gate_threshold_0p950" in summary.columns
