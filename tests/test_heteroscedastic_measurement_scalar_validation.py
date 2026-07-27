from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from raft_uav.heteroscedastic_cli import (
    nis_scored_candidates_with_row_covariance,
    radar_measurements_to_enu_with_row_covariance,
    rf_measurements_to_enu_with_row_covariance,
)
from raft_uav.heteroscedastic_measurements import (
    radar_measurements_to_enu_with_uncertainty,
    rf_measurements_to_enu_with_uncertainty,
)

_RF_CONVERTERS: tuple[Callable[..., object], ...] = (
    rf_measurements_to_enu_with_uncertainty,
    rf_measurements_to_enu_with_row_covariance,
)
_RADAR_CONVERTERS: tuple[Callable[..., object], ...] = (
    radar_measurements_to_enu_with_uncertainty,
    radar_measurements_to_enu_with_row_covariance,
)


class _TrackerView:
    state = np.zeros(6)
    covariance_matrix = np.eye(6)


def _rf_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "std_m": [75.0],
        }
    )


def _radar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [2.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
        }
    )


@pytest.mark.parametrize("converter", _RF_CONVERTERS, ids=lambda item: item.__name__)
@pytest.mark.parametrize("field", ["time_s", "east_m", "north_m"])
def test_rf_conversion_rejects_complex_required_scalars(
    converter: Callable[..., object],
    field: str,
) -> None:
    frame = _rf_frame()
    frame[field] = np.complex128(1.0 + 2.0j)

    with pytest.raises(ValueError, match=field):
        converter(frame)


@pytest.mark.parametrize("converter", _RADAR_CONVERTERS, ids=lambda item: item.__name__)
@pytest.mark.parametrize("field", ["time_s", "east_m", "north_m", "up_m"])
def test_radar_conversion_rejects_complex_required_scalars(
    converter: Callable[..., object],
    field: str,
) -> None:
    frame = _radar_frame()
    frame[field] = np.complex128(1.0 + 2.0j)

    with pytest.raises(ValueError, match=field):
        converter(frame)


@pytest.mark.parametrize("converter", _RF_CONVERTERS, ids=lambda item: item.__name__)
def test_rf_conversion_rejects_complex_default_standard_deviation(
    converter: Callable[..., object],
) -> None:
    with pytest.raises(ValueError, match="default_std_m"):
        converter(_rf_frame(), default_std_m=np.complex128(75.0 + 0.0j))


@pytest.mark.parametrize("converter", _RADAR_CONVERTERS, ids=lambda item: item.__name__)
def test_radar_conversion_rejects_complex_default_standard_deviations(
    converter: Callable[..., object],
) -> None:
    with pytest.raises(ValueError, match="default_xy_std_m"):
        converter(
            _radar_frame(),
            default_xy_std_m=np.complex128(25.0 + 0.0j),
        )


def test_nis_scoring_rejects_complex_candidate_coordinates() -> None:
    candidates = _radar_frame().drop(columns="time_s")
    candidates["east_m"] = np.complex128(10.0 + 5.0j)

    with pytest.raises(ValueError, match="east_m"):
        nis_scored_candidates_with_row_covariance(
            candidates,
            _TrackerView(),
            np.eye(3),
        )


@pytest.mark.parametrize("converter", _RF_CONVERTERS, ids=lambda item: item.__name__)
def test_rf_conversion_preserves_valid_numpy_real_scalars(
    converter: Callable[..., object],
) -> None:
    frame = _rf_frame().astype(object)
    frame.loc[0, "time_s"] = np.float64(1.25)
    frame.loc[0, "east_m"] = np.float32(11.0)
    frame.loc[0, "north_m"] = np.array(22.0)

    [measurement] = converter(frame, default_std_m=np.float64(80.0))

    assert measurement.time_s == pytest.approx(1.25)
    np.testing.assert_allclose(measurement.vector, [11.0, 22.0])
