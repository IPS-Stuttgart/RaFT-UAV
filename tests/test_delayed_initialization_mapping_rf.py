import numpy as np
import pandas as pd

from raft_uav.baselines.delayed_initialization import build_delayed_initial_hypotheses


def test_delayed_initialization_accepts_mapping_rf_measurements() -> None:
    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[
            {
                "sequence_id": "flight-a",
                "time_s": 1.25,
                "vector": [10.0, 20.0, 30.0],
            }
        ],
        radar=pd.DataFrame(),
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.source == "rf"
    assert hypothesis.time_s == 1.25
    assert hypothesis.metadata == {"rf_dimension": 3}
    np.testing.assert_allclose(hypothesis.state, [10.0, 20.0, 30.0, 0.0, 0.0, 0.0])
