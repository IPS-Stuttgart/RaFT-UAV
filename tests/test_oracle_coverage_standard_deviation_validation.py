from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.oracle_coverage import build_oracle_candidate_coverage


def _empty_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["time_s", "east_m", "north_m", "up_m"]
    return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)


@pytest.mark.parametrize("field_name", ["radar_xy_std_m", "radar_z_std_m"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="infinity"),
        pytest.param(True, id="boolean"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(np.array([1.0]), id="non-scalar"),
        pytest.param(np.ma.masked, id="masked"),
    ],
)
def test_oracle_coverage_rejects_invalid_radar_standard_deviations(
    field_name: str,
    invalid_value: object,
) -> None:
    radar, truth = _empty_inputs()

    with pytest.raises(
        ValueError,
        match=rf"{field_name} must be a finite positive real scalar",
    ):
        build_oracle_candidate_coverage(
            radar=radar,
            truth=truth,
            **{field_name: invalid_value},
        )


def test_oracle_coverage_accepts_scalar_like_radar_standard_deviations() -> None:
    radar, truth = _empty_inputs()

    result = build_oracle_candidate_coverage(
        radar=radar,
        truth=truth,
        radar_xy_std_m=np.asarray(25.0),
        radar_z_std_m="35.0",
    )

    assert result.frame_coverage.empty
