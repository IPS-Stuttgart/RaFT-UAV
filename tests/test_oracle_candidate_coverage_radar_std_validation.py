from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.oracle_candidate_coverage import (
    build_oracle_candidate_coverage_diagnostics,
)


@pytest.mark.parametrize("field", ("radar_xy_std_m", "radar_z_std_m"))
@pytest.mark.parametrize(
    "value",
    (
        0.0,
        -1.0,
        np.nan,
        np.inf,
        True,
        25.0 + 1.0j,
        np.array([25.0]),
        np.array(np.complex64(25.0 + 1.0j), dtype=object),
        np.ma.masked,
    ),
)
def test_oracle_candidate_coverage_rejects_invalid_radar_standard_deviations(
    field: str,
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"^{field} must be a finite positive real scalar$",
    ):
        build_oracle_candidate_coverage_diagnostics(
            radar=pd.DataFrame(),
            truth=pd.DataFrame(),
            **{field: value},
        )


def test_oracle_candidate_coverage_accepts_scalar_like_real_standard_deviations() -> None:
    report, summary = build_oracle_candidate_coverage_diagnostics(
        radar=pd.DataFrame(),
        truth=pd.DataFrame(),
        radar_xy_std_m=np.asarray(np.float64(25.0), dtype=object),
        radar_z_std_m=np.ma.array(np.float64(35.0), mask=False),
    )

    assert report.empty
    assert summary["radar_frame_count"] == 0
