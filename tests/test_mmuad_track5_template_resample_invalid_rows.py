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


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("time_s", "not-a-time"),
        ("state_x_m", np.nan),
        ("state_y_m", np.inf),
        ("state_z_m", -np.inf),
    ],
)
def test_template_resampler_rejects_malformed_estimate_rows(
    column: str,
    invalid_value: object,
) -> None:
    estimates = _estimate_rows()
    estimates[column] = estimates[column].astype(object)
    estimates.loc[11, column] = invalid_value
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"non-finite or non-numeric time or position values at row indices: 11",
    ):
        resample_estimates_to_track5_template(estimates, template)
