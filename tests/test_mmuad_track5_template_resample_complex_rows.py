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
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 1.0, 2.0],
            "state_z_m": [10.0, 11.0, 12.0],
        },
        index=[10, 11, 12],
    )


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0],
        },
        index=[20, 21, 22],
    )


@pytest.mark.parametrize("column", ["time_s", "state_x_m", "state_y_m", "state_z_m"])
@pytest.mark.parametrize("invalid_value", [1.0 + 2.0j, np.complex128(1.0 + 0.0j)])
def test_template_resampler_rejects_complex_estimate_values(
    column: str,
    invalid_value: complex,
) -> None:
    estimates = _estimate_rows()
    estimates[column] = estimates[column].astype(object)
    estimates.loc[11, column] = invalid_value

    with pytest.raises(
        ValueError,
        match=r"complex time or position values at row indices: 11",
    ):
        resample_estimates_to_track5_template(estimates, _template_rows())


@pytest.mark.parametrize("invalid_value", [1.0 + 2.0j, np.complex128(1.0 + 0.0j)])
def test_template_resampler_rejects_complex_template_timestamps(
    invalid_value: complex,
) -> None:
    template = _template_rows()
    template["Timestamp"] = template["Timestamp"].astype(object)
    template.loc[21, "Timestamp"] = invalid_value

    with pytest.raises(
        ValueError,
        match=r"template contains complex timestamp values at row indices: 21",
    ):
        resample_estimates_to_track5_template(_estimate_rows(), template)
