from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.paper_table import run_paper_table_diagnostic


_INVALID_POSITIVE_REAL_CONTROLS = [
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
]


@pytest.mark.parametrize("bad_value", _INVALID_POSITIVE_REAL_CONTROLS)
def test_paper_table_rejects_invalid_radar_interpolation_max_gap(
    bad_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="radar_interpolation_max_gap_s must be a finite positive real scalar",
    ):
        run_paper_table_diagnostic(
            dataset_root=Path("missing-dataset"),
            flight_name="missing-flight",
            radar_interpolation_max_gap_s=bad_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_value", _INVALID_POSITIVE_REAL_CONTROLS)
def test_paper_table_rejects_invalid_radar_interpolation_max_speed(
    bad_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "radar_interpolation_max_speed_mps must be a finite positive real scalar"
        ),
    ):
        run_paper_table_diagnostic(
            dataset_root=Path("missing-dataset"),
            flight_name="missing-flight",
            radar_interpolation_max_speed_mps=bad_value,  # type: ignore[arg-type]
        )
