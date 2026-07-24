from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.delayed_initialization import (
    build_delayed_initial_hypotheses,
)


def test_delayed_initialization_ignores_boolean_track_ids_in_history() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "track_id": pd.Series([1, 1, True], dtype=object),
            "east_m": [0.0, 1.0, 100.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [1.0, 1.0, 1.0],
        }
    )

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[],
        radar=radar,
    )

    track_one = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.metadata["track_id"] == 1
    ]
    assert len(track_one) == 2
    for hypothesis in track_one:
        assert hypothesis.state[3:6] == pytest.approx([1.0, 0.0, 0.0])
        assert hypothesis.metadata["support_score"] == pytest.approx(0.5)


def test_delayed_initialization_keeps_integer_equivalent_track_ids() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "track_id": pd.Series(["1.0", np.int64(1)], dtype=object),
            "east_m": [0.0, 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [1.0, 1.0],
        }
    )

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[],
        radar=radar,
    )

    assert len(hypotheses) == 2
    for hypothesis in hypotheses:
        assert hypothesis.metadata["track_id"] == 1
        assert hypothesis.state[3:6] == pytest.approx([2.0, 0.0, 0.0])
        assert hypothesis.metadata["support_score"] == pytest.approx(0.5)
