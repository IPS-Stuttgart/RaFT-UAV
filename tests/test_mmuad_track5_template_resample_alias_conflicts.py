from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import (
    resample_estimates_to_track5_template,
)


def _estimate_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq-a"],
            "time_s": [1.0],
            "state_x_m": [10.0],
            "state_y_m": [20.0],
            "state_z_m": [30.0],
            "classification": [2],
        }
    )


def _template_row() -> pd.DataFrame:
    return pd.DataFrame({"Sequence": ["seq-a"], "Timestamp": [1.0]})


def test_resampler_rejects_conflicting_estimate_sequence_aliases() -> None:
    estimates = _estimate_row()
    estimates["Sequence"] = "seq-b"

    with pytest.raises(ValueError, match="conflicting sequence aliases.*row indices: 0"):
        resample_estimates_to_track5_template(estimates, _template_row())


def test_resampler_rejects_conflicting_estimate_time_aliases() -> None:
    estimates = _estimate_row()
    estimates["Timestamp"] = 2.0

    with pytest.raises(ValueError, match="conflicting timestamp aliases.*row indices: 0"):
        resample_estimates_to_track5_template(estimates, _template_row())


def test_resampler_rejects_conflicting_estimate_classification_aliases() -> None:
    estimates = _estimate_row()
    estimates["Classification"] = 3

    with pytest.raises(
        ValueError,
        match="conflicting classification aliases.*row indices: 0",
    ):
        resample_estimates_to_track5_template(estimates, _template_row())


def test_resampler_accepts_equivalent_redundant_estimate_aliases() -> None:
    estimates = _estimate_row()
    estimates["Sequence"] = " seq-a "
    estimates["Timestamp"] = "1.0"
    estimates["Classification"] = "2"

    resampled, diagnostics = resample_estimates_to_track5_template(
        estimates,
        _template_row(),
    )

    assert resampled["sequence_id"].tolist() == ["seq-a"]
    assert resampled["time_s"].tolist() == [1.0]
    assert resampled["classification"].tolist() == [2]
    assert diagnostics["valid"].tolist() == [True]


def test_resampler_rejects_conflicting_template_aliases() -> None:
    template = _template_row()
    template["sequence_id"] = "seq-b"

    with pytest.raises(ValueError, match="conflicting sequence aliases.*row indices: 0"):
        resample_estimates_to_track5_template(_estimate_row(), template)
