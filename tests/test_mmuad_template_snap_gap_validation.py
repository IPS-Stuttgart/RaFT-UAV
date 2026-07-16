from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [0.0, 10.0],
            "Position": ["(0,0,0)", "(10,20,2)"],
            "Classification": [2, 2],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [5.0]})


@pytest.mark.parametrize(
    "bad_gap",
    [
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        np.bool_(False),
        np.array(True),
        np.array([1.0]),
        np.array([[1.0]]),
        1.0 + 0.0j,
        np.ma.masked,
    ],
)
def test_snap_official_results_rejects_invalid_max_interpolation_gap_s(
    bad_gap: object,
) -> None:
    with pytest.raises(ValueError, match="max_interpolation_gap_s"):
        snap_official_results_to_template(
            _results(),
            _template(),
            max_interpolation_gap_s=bad_gap,  # type: ignore[arg-type]
        )


def test_snap_official_results_accepts_zero_gap_threshold() -> None:
    snapped, diagnostics = snap_official_results_to_template(
        _results(),
        _template(),
        max_interpolation_gap_s=0.0,
    )

    assert snapped["Position"].tolist() == ["(0,0,0)"]
    assert diagnostics["method"].tolist() == ["nearest-large-gap-fallback"]
    assert diagnostics["large_gap_fallback"].tolist() == [True]


def test_snap_official_results_accepts_zero_dimensional_real_scalar() -> None:
    snapped, diagnostics = snap_official_results_to_template(
        _results(),
        _template(),
        max_interpolation_gap_s=np.array(10.0),
    )

    assert snapped["Position"].tolist() == ["(5,10,1)"]
    assert diagnostics["method"].tolist() == ["linear"]
    assert diagnostics["large_gap_fallback"].tolist() == [False]
