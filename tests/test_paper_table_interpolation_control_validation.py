from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.paper_table import run_paper_table_diagnostic


@pytest.mark.parametrize(
    "field",
    [
        "radar_interpolation_max_gap_s",
        "radar_interpolation_max_speed_mps",
    ],
)
@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        np.bool_(False),
        0.0,
        -1.0,
        np.nan,
        np.inf,
        1.0 + 0.0j,
        np.array([1.0]),
        pd.NA,
        np.ma.masked,
    ],
)
def test_paper_table_rejects_invalid_radar_interpolation_caps(
    field: str,
    bad_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field} must be a finite positive real scalar",
    ):
        run_paper_table_diagnostic(
            dataset_root=Path("missing-dataset"),
            flight_name="missing-flight",
            **{field: bad_value},
        )
