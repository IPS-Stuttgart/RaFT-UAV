from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.io.aerpaw import rf_measurements_to_enu


def _normalized_rf(**extra_columns: object) -> pd.DataFrame:
    data: dict[str, object] = {
        "time_s": [1.0],
        "east_m": [10.0],
        "north_m": [20.0],
    }
    data.update(extra_columns)
    return pd.DataFrame(data)


def test_normalized_rf_converter_uses_default_when_std_column_is_missing() -> None:
    [measurement] = rf_measurements_to_enu(_normalized_rf(), default_std_m=4.0)

    np.testing.assert_allclose(measurement.covariance, np.diag([16.0, 16.0]))


@pytest.mark.parametrize("row_std", [None, True, 0.0, -1.0, np.nan])
def test_normalized_rf_converter_uses_default_for_invalid_row_std(row_std: object) -> None:
    [measurement] = rf_measurements_to_enu(
        _normalized_rf(std_m=[row_std]),
        default_std_m=3.0,
    )

    np.testing.assert_allclose(measurement.covariance, np.diag([9.0, 9.0]))


@pytest.mark.parametrize("default_std", [None, True, 0.0, -1.0, np.nan])
def test_normalized_rf_converter_rejects_invalid_default_std(default_std: object) -> None:
    with pytest.raises(ValueError, match="default_std_m"):
        rf_measurements_to_enu(_normalized_rf(), default_std_m=default_std)
