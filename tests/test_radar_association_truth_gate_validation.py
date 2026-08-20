from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines import radar_association
from raft_uav.baselines.radar_association import (
    run_async_cv_baseline_with_radar_association,
)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("truth_gate_m", math.nan, "truth_gate_m must be finite"),
        ("truth_gate_m", math.inf, "truth_gate_m must be finite"),
        ("truth_gate_m", -1.0, "truth_gate_m must be nonnegative"),
        ("truth_gate_m", True, "truth_gate_m must be finite"),
        ("truth_gate_m", np.array([150.0]), "truth_gate_m must be finite"),
        ("truth_time_gate_s", math.nan, "truth_time_gate_s must be finite"),
        ("truth_time_gate_s", math.inf, "truth_time_gate_s must be finite"),
        ("truth_time_gate_s", -1.0, "truth_time_gate_s must be nonnegative"),
        ("truth_time_gate_s", np.bool_(False), "truth_time_gate_s must be finite"),
        ("truth_time_gate_s", np.array([1.0]), "truth_time_gate_s must be finite"),
    ],
)
def test_radar_association_rejects_invalid_truth_gates(
    parameter: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_async_cv_baseline_with_radar_association(
            rf_measurements=[],
            radar=pd.DataFrame(),
            association="oracle-nearest-truth",
            truth=pd.DataFrame(),
            **{parameter: value},
        )


def test_radar_association_normalizes_valid_truth_gates() -> None:
    bound = radar_association._RUN_SIGNATURE.bind(
        rf_measurements=[],
        radar=pd.DataFrame(),
        association="oracle-nearest-truth",
        truth=pd.DataFrame(),
        truth_gate_m="150.0",
        truth_time_gate_s=np.array(1.0),
    )
    bound.apply_defaults()

    radar_association._validate_radar_association_parameters(bound.arguments)

    assert bound.arguments["truth_gate_m"] == 150.0
    assert isinstance(bound.arguments["truth_gate_m"], float)
    assert bound.arguments["truth_time_gate_s"] == 1.0
    assert isinstance(bound.arguments["truth_time_gate_s"], float)
