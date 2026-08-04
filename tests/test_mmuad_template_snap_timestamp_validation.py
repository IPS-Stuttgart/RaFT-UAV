from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template
from raft_uav.mmuad.template_snap_utils import (
    load_official_track5_results_frame_from_frame,
)


def _results(timestamp: object = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [timestamp],
            "Position": ["(1,2,3)"],
            "Classification": [1],
        }
    )


def _template(timestamp: object = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [timestamp],
        }
    )


@pytest.mark.parametrize("timestamp", [True, np.bool_(False), np.asarray(True)])
def test_results_normalizer_rejects_boolean_timestamps(timestamp: object) -> None:
    with pytest.raises(ValueError, match="Timestamp values.*not booleans"):
        load_official_track5_results_frame_from_frame(_results(timestamp))


@pytest.mark.parametrize("timestamp", [1.0 + 2.0j, np.complex128(3.0 + 4.0j)])
def test_results_normalizer_rejects_complex_timestamps(timestamp: object) -> None:
    with pytest.raises(ValueError, match="Timestamp values.*not complex numbers"):
        load_official_track5_results_frame_from_frame(_results(timestamp))


def test_results_normalizer_rejects_zero_dimensional_boolean_classification() -> None:
    results = _results()
    results["Classification"] = [np.asarray(True)]

    with pytest.raises(ValueError, match="Classification values.*not booleans"):
        load_official_track5_results_frame_from_frame(results)


@pytest.mark.parametrize("timestamp", [True, np.asarray(False)])
def test_template_snap_rejects_boolean_template_timestamps(timestamp: object) -> None:
    with pytest.raises(ValueError, match="Track 5 template Timestamp values.*not booleans"):
        snap_official_results_to_template(_results(), _template(timestamp))


def test_template_snap_rejects_zero_dimensional_complex_template_timestamp() -> None:
    timestamp = np.asarray(1.0 + 2.0j)

    with pytest.raises(ValueError, match="Track 5 template Timestamp values.*complex"):
        snap_official_results_to_template(_results(), _template(timestamp))


def test_template_snap_retains_numeric_string_timestamp_support() -> None:
    snapped, diagnostics = snap_official_results_to_template(
        _results("1.0"),
        _template("1.0"),
    )

    assert snapped["Timestamp"].tolist() == [1.0]
    assert diagnostics["valid"].tolist() == [True]
