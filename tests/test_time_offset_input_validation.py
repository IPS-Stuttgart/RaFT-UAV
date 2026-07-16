from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.time_offset import (
    aggregate_measurement_time_offset_sweep,
    aggregate_radar_time_offset_sweep,
    apply_time_offset,
    fit_measurement_time_offset,
    fit_radar_time_offset,
)


@pytest.mark.parametrize(
    "offset_s",
    [
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        np.array(True),
        np.array([0.5]),
    ],
)
def test_apply_time_offset_rejects_invalid_scalar_offsets(offset_s: object) -> None:
    frame = pd.DataFrame({"time_s": [1.0, 2.0]})

    with pytest.raises(ValueError, match="offset_s must be a finite real scalar"):
        apply_time_offset(frame, offset_s)


@pytest.mark.parametrize(
    ("sweep", "kwargs"),
    [
        (aggregate_radar_time_offset_sweep, {}),
        (aggregate_measurement_time_offset_sweep, {"dimensions": 2}),
    ],
)
@pytest.mark.parametrize("offset_s", [True, np.bool_(False), np.nan, np.inf])
def test_time_offset_sweeps_reject_invalid_offsets(
    sweep: Callable[..., pd.DataFrame],
    kwargs: dict[str, object],
    offset_s: object,
) -> None:
    with pytest.raises(ValueError, match="offset_s must be a finite real scalar"):
        sweep([], offsets_s=[offset_s], **kwargs)


@pytest.mark.parametrize(
    "dimensions",
    [1, 4, True, np.bool_(False), 2.5, np.nan, np.array([2])],
)
def test_measurement_sweep_rejects_invalid_dimensions(dimensions: object) -> None:
    with pytest.raises(ValueError, match="dimensions must be 2 or 3"):
        aggregate_measurement_time_offset_sweep(
            [],
            offsets_s=[0.0],
            dimensions=dimensions,
        )


def test_fit_helpers_dispatch_through_validated_public_sweeps() -> None:
    with pytest.raises(ValueError, match="offset_s must be a finite real scalar"):
        fit_radar_time_offset([], offsets_s=[np.nan])

    with pytest.raises(ValueError, match="dimensions must be 2 or 3"):
        fit_measurement_time_offset([], offsets_s=[0.0], dimensions=4)


def test_time_offset_paths_accept_zero_dimensional_numeric_scalars() -> None:
    frame = pd.DataFrame({"time_s": [1.0, 2.0]})

    shifted = apply_time_offset(frame, np.array(0.25))
    radar_sweep = aggregate_radar_time_offset_sweep(
        [],
        offsets_s=[np.array(0.25)],
    )
    measurement_sweep = aggregate_measurement_time_offset_sweep(
        [],
        offsets_s=[np.float64(0.25)],
        dimensions=np.array(2),
    )

    np.testing.assert_allclose(shifted["time_s"], [1.25, 2.25])
    assert radar_sweep["time_offset_s"].tolist() == [0.25]
    assert measurement_sweep["time_offset_s"].tolist() == [0.25]
