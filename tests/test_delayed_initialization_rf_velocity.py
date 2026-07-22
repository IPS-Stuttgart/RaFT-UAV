from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from raft_uav.baselines.delayed_initialization import (
    build_delayed_initial_hypotheses,
)


def test_delayed_initialization_preserves_six_dimensional_rf_state() -> None:
    vector = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
    rf = [SimpleNamespace(time_s=4.0, vector=vector)]

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=rf,
        radar=pd.DataFrame(),
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.source == "rf"
    assert hypothesis.metadata["rf_dimension"] == 6
    np.testing.assert_allclose(hypothesis.state, vector)
