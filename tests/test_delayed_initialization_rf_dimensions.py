import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.delayed_initialization import build_delayed_initial_hypotheses


@pytest.mark.parametrize("unsupported_dimension", [4, 5, 7])
def test_delayed_initialization_skips_unsupported_rf_dimensions_before_window_anchor(
    unsupported_dimension: int,
) -> None:
    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[
            {
                "time_s": 0.0,
                "vector": np.arange(unsupported_dimension, dtype=float),
            },
            {
                "time_s": 10.0,
                "vector": [10.0, 20.0, 30.0],
            },
        ],
        radar=pd.DataFrame(),
        window_s=1.0,
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.source == "rf"
    assert hypothesis.time_s == 10.0
    assert hypothesis.metadata == {"rf_dimension": 3}
    np.testing.assert_allclose(
        hypothesis.state,
        [10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    ("vector", "expected_state"),
    [
        ([10.0, 20.0], [10.0, 20.0, 0.0, 0.0, 0.0, 0.0]),
        ([10.0, 20.0, 30.0], [10.0, 20.0, 30.0, 0.0, 0.0, 0.0]),
        (
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
        ),
    ],
)
def test_delayed_initialization_keeps_supported_rf_dimensions(
    vector: list[float],
    expected_state: list[float],
) -> None:
    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[{"time_s": 2.0, "vector": vector}],
        radar=pd.DataFrame(),
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.metadata == {"rf_dimension": len(vector)}
    np.testing.assert_allclose(hypothesis.state, expected_state)
