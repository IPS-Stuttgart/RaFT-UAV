from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.radar_likelihood_training import (
    collect_radar_association_training_frame,
)


@pytest.mark.parametrize("field", ["truth_gate_m", "truth_time_gate_s"])
@pytest.mark.parametrize(
    "gate",
    [
        -0.1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([0.5]),
    ],
)
def test_radar_training_rejects_invalid_truth_gates(
    field: str,
    gate: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        collect_radar_association_training_frame(
            rf_measurements=(),
            radar=pd.DataFrame(),
            truth=pd.DataFrame(),
            **{field: gate},
        )


def test_radar_training_accepts_zero_dimensional_zero_truth_gates() -> None:
    rows = collect_radar_association_training_frame(
        rf_measurements=(),
        radar=pd.DataFrame(),
        truth=pd.DataFrame(),
        truth_gate_m=np.array(0.0),
        truth_time_gate_s=np.array(0.0),
    )

    assert rows.empty
