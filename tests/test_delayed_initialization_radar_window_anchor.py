from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.baselines.delayed_initialization import (
    build_delayed_initial_hypotheses,
)


def test_malformed_early_radar_row_does_not_anchor_initialization_window() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 100.0, 101.0],
            "track_id": [7, 7, 7],
            "east_m": ["bad-position", 10.0, 11.0],
            "north_m": [0.0, 20.0, 20.0],
            "up_m": [0.0, 30.0, 30.0],
        }
    )

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[],
        radar=radar,
        window_s=5.0,
    )

    assert {hypothesis.time_s for hypothesis in hypotheses} == {100.0, 101.0}
    assert all(hypothesis.source == "radar" for hypothesis in hypotheses)
    for hypothesis in hypotheses:
        assert hypothesis.state[3:6] == pytest.approx([1.0, 0.0, 0.0])
