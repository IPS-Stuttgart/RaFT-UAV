from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from raft_uav.baselines.delayed_initialization import (
    build_delayed_initial_hypotheses,
)


def _radar_at_matching_horizontal_position() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [1000.0],
        }
    )


def test_two_dimensional_rf_support_ignores_unobserved_altitude() -> None:
    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[
            SimpleNamespace(time_s=0.0, vector=np.array([10.0, 20.0]))
        ],
        radar=_radar_at_matching_horizontal_position(),
        max_hypotheses=1,
    )

    assert len(hypotheses) == 1
    assert hypotheses[0].source == "rf"
    assert np.isclose(hypotheses[0].score, 0.0)


def test_three_dimensional_rf_support_keeps_altitude_residual() -> None:
    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[
            SimpleNamespace(time_s=0.0, vector=np.array([10.0, 20.0, 0.0]))
        ],
        radar=_radar_at_matching_horizontal_position(),
        max_hypotheses=2,
    )

    rf_hypothesis = next(item for item in hypotheses if item.source == "rf")
    assert np.isclose(rf_hypothesis.score, 10.0)
