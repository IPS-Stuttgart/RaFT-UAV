from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import (
    resample_estimates_to_track5_template,
)


def _estimate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 2.0],
            "state_x_m": [0.0, 2.0],
            "state_y_m": [0.0, 2.0],
            "state_z_m": [10.0, 12.0],
        }
    )


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0],
        },
        index=[20, 21, 22],
    )


@pytest.mark.parametrize(
    "invalid_value",
    [None, "not-a-time", np.nan, np.inf, -np.inf, True, np.bool_(False)],
)
def test_template_resampler_rejects_invalid_timestamp_rows(
    invalid_value: object,
) -> None:
    template = _template_rows()
    template["Timestamp"] = template["Timestamp"].astype(object)
    template.loc[21, "Timestamp"] = invalid_value

    with pytest.raises(
        ValueError,
        match=r"template contains an invalid timestamp at row 21",
    ):
        resample_estimates_to_track5_template(_estimate_rows(), template)


def test_template_resampler_still_filters_missing_sequence_rows() -> None:
    template = _template_rows()
    template["Sequence"] = template["Sequence"].astype(object)
    template.loc[21, "Sequence"] = None

    resampled, diagnostics = resample_estimates_to_track5_template(
        _estimate_rows(),
        template,
    )

    assert resampled["time_s"].tolist() == [0.0, 2.0]
    assert diagnostics["valid"].tolist() == [True, True]


def test_template_resampler_keeps_every_valid_requested_row() -> None:
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [1.0, 1.0, 2.0],
        }
    )

    resampled, diagnostics = resample_estimates_to_track5_template(
        _estimate_rows(),
        template,
    )

    assert len(resampled) == len(template)
    assert len(diagnostics) == len(template)
    assert resampled["time_s"].tolist() == [1.0, 1.0, 2.0]
