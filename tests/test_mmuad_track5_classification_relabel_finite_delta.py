from __future__ import annotations

import math
import runpy
import sys

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


@pytest.mark.parametrize("tolerance", [math.nan, math.inf, -math.inf])
def test_nearest_time_relabel_rejects_nonfinite_delta_gate(tolerance: float) -> None:
    with pytest.raises(ValueError, match="max_nearest_time_delta_s must be finite"):
        relabel_track5_classification(
            _rows(),
            _rows(),
            mode="by-nearest-time",
            max_nearest_time_delta_s=tolerance,
        )


def test_classification_relabel_package_supports_python_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "raft_uav.mmuad.track5_classification_relabel"
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")

    assert exc_info.value.code == 0
