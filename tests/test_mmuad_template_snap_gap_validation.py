from __future__ import annotations

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


@pytest.mark.parametrize("bad_gap", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_snap_official_results_rejects_invalid_max_interpolation_gap_s(
    bad_gap: float,
) -> None:
    with pytest.raises(ValueError, match="max_interpolation_gap_s"):
        snap_official_results_to_template(
            _results(),
            _template(),
            max_interpolation_gap_s=bad_gap,
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
