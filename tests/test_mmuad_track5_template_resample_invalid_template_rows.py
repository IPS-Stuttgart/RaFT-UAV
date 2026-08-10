from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_sequence_gate import (
    blend_track5_estimate_sequence_gate,
)
from raft_uav.mmuad.track5_template_resample import (
    resample_estimates_to_track5_template,
)


def _estimate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 1.0, 2.0],
            "state_z_m": [10.0, 11.0, 12.0],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0],
        },
        index=[20, 21, 22],
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame({"sequence_id": ["seq0001"], "weight": [0.5]})


@pytest.mark.parametrize(
    "invalid_value",
    [
        "not-a-time",
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
    ],
)
def test_template_resampler_rejects_malformed_template_timestamps(
    invalid_value: object,
) -> None:
    template = _template()
    template["Timestamp"] = template["Timestamp"].astype(object)
    template.at[21, "Timestamp"] = invalid_value

    with pytest.raises(
        ValueError,
        match=r"non-finite or non-numeric timestamp values at row indices: 21",
    ):
        resample_estimates_to_track5_template(_estimate_rows(), template)


@pytest.mark.parametrize("invalid_value", [None, np.nan, pd.NA, "", "   "])
def test_template_resampler_rejects_invalid_template_sequence_ids(
    invalid_value: object,
) -> None:
    template = _template()
    template["Sequence"] = template["Sequence"].astype(object)
    template.at[21, "Sequence"] = invalid_value

    with pytest.raises(
        ValueError,
        match=r"invalid sequence identifiers at row indices: 21",
    ):
        resample_estimates_to_track5_template(_estimate_rows(), template)


def test_template_resampler_accepts_lossless_scalar_like_timestamps() -> None:
    template = _template()
    template["Timestamp"] = template["Timestamp"].astype(object)
    template.at[20, "Timestamp"] = "0.0"
    template.at[21, "Timestamp"] = np.array(1.0)

    resampled, diagnostics = resample_estimates_to_track5_template(
        _estimate_rows(),
        template,
    )

    assert resampled["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert diagnostics["time_s"].tolist() == [0.0, 1.0, 2.0]


def test_estimate_sequence_gate_reuses_strict_template_validation() -> None:
    template = _template()
    template["Timestamp"] = template["Timestamp"].astype(object)
    template.at[21, "Timestamp"] = "not-a-time"

    with pytest.raises(
        ValueError,
        match=r"non-finite or non-numeric timestamp values at row indices: 21",
    ):
        blend_track5_estimate_sequence_gate(
            base_estimates=_estimate_rows(),
            alternate_estimates=_estimate_rows(),
            template=template,
            sequence_weights=_weights(),
        )
