from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.delayed_initialization import (
    build_delayed_initial_hypotheses,
)


def test_delayed_initialization_skips_malformed_radar_rows() -> None:
    radar = pd.DataFrame(
        {
            "time_s": ["bad-time", 0.0, 1.0, 2.0],
            "track_id": [7, 7, 7, 7],
            "east_m": [0.0, "bad-position", 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
            "cat_prob_uav": [0.5, 0.5, 0.5, 0.5],
        }
    )

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[],
        radar=radar,
    )

    assert {hypothesis.time_s for hypothesis in hypotheses} == {1.0, 2.0}
    assert all(hypothesis.source == "radar" for hypothesis in hypotheses)
    for hypothesis in hypotheses:
        assert hypothesis.state[3:6] == pytest.approx([1.0, 0.0, 0.0])


def test_rf_support_ignores_malformed_radar_coordinates() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "track_id": [1, 2],
            "east_m": ["bad-position", 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    rf = [SimpleNamespace(time_s=0.5, vector=np.array([0.0, 0.0, 0.0]))]

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=rf,
        radar=radar,
    )

    rf_hypothesis = next(
        hypothesis for hypothesis in hypotheses if hypothesis.source == "rf"
    )
    assert rf_hypothesis.score == pytest.approx(0.02)
    assert sum(hypothesis.source == "radar" for hypothesis in hypotheses) == 1


def test_delayed_initialization_skips_malformed_rf_measurements() -> None:
    valid = SimpleNamespace(time_s=0.5, vector=np.array([1.0, 2.0, 3.0]))
    malformed = [
        SimpleNamespace(time_s=np.nan, vector=np.array([4.0, 5.0, 6.0])),
        SimpleNamespace(time_s=1.0, vector=np.array([np.inf, 5.0, 6.0])),
        SimpleNamespace(time_s="bad-time", vector=np.array([4.0, 5.0, 6.0])),
        SimpleNamespace(time_s=1.0, vector=["bad-position", 5.0, 6.0]),
        SimpleNamespace(vector=np.array([4.0, 5.0, 6.0])),
    ]

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[*malformed, valid],
        radar=pd.DataFrame(),
    )

    assert len(hypotheses) == 1
    assert hypotheses[0].source == "rf"
    assert hypotheses[0].time_s == pytest.approx(0.5)
    assert hypotheses[0].state == pytest.approx([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
