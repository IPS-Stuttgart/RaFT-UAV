from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.evaluation.fifth_wave_diagnostics import track_purity_summary


def test_track_purity_preserves_large_ids_and_rejects_fractional_values() -> None:
    lower = 2**80 + 101
    upper = lower + 1
    selected = pd.DataFrame(
        {
            "track_id": [
                str(lower),
                str(upper),
                str(upper),
                "7.5",
            ]
        }
    )

    summary = track_purity_summary(selected)

    assert summary["selected_radar_rows"] == 4
    assert summary["dominant_track_id"] == upper
    assert summary["dominant_track_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["selected_track_count"] == 2
    assert summary["selected_track_entropy"] > 0.0
