from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_classification_relabel import relabel_track5_classification


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )


@pytest.mark.parametrize(
    "tolerance",
    [
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="positive-infinity"),
        pytest.param(-math.inf, id="negative-infinity"),
        pytest.param(-0.1, id="negative"),
        pytest.param(True, id="boolean"),
        pytest.param(0.25 + 0.0j, id="complex"),
        pytest.param(np.ma.masked, id="masked"),
        pytest.param(np.array([0.25]), id="non-scalar"),
    ],
)
def test_nearest_time_relabel_rejects_invalid_delta_gate(tolerance: object) -> None:
    with pytest.raises(
        ValueError,
        match="max_nearest_time_delta_s must be a finite non-negative number",
    ):
        relabel_track5_classification(
            _rows(),
            _rows(),
            mode="by-nearest-time",
            max_nearest_time_delta_s=tolerance,
        )


def test_nearest_time_relabel_accepts_zero_delta_gate() -> None:
    result = relabel_track5_classification(
        _rows(),
        _rows(),
        mode="by-nearest-time",
        max_nearest_time_delta_s=0.0,
    )

    assert result.rows["Classification"].tolist() == [1]
    assert result.manifest["max_nearest_time_delta_s"] == 0.0
