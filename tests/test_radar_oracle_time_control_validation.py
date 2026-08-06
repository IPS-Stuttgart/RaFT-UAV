from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.time_offset import (
    aggregate_measurement_time_offset_sweep,
    aggregate_radar_time_offset_sweep,
    fit_measurement_time_offset,
    fit_radar_time_offset,
)
from raft_uav.evaluation.radar_oracle_diagnostics import (
    interpolate_truth_positions,
    nearest_candidate_oracle,
    time_offset_sweep,
)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 2.0],
            "east_m": [0.0, 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def _radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [0],
            "time_s": [1.0],
            "east_m": [1.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )


@pytest.mark.parametrize(
    "maximum",
    [
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        -0.1,
        1 + 0j,
        np.array([1.0]),
        np.ma.masked,
        np.ma.array(1.0, mask=True),
    ],
)
def test_truth_freshness_gate_rejects_malformed_controls(maximum: object) -> None:
    with pytest.raises(ValueError, match="max_time_delta_s"):
        interpolate_truth_positions(
            _truth(),
            [1.0],
            max_time_delta_s=maximum,
        )


@pytest.mark.parametrize(
    "offset",
    [
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        1 + 0j,
        np.array([0.0]),
        np.ma.masked,
        np.ma.array(0.0, mask=True),
    ],
)
def test_nearest_candidate_oracle_rejects_malformed_offsets(offset: object) -> None:
    with pytest.raises(ValueError, match="time_offset_s"):
        nearest_candidate_oracle(
            _radar(),
            _truth(),
            time_offset_s=offset,
        )


def test_empty_oracle_inputs_still_validate_time_controls() -> None:
    with pytest.raises(ValueError, match="time_offset_s"):
        nearest_candidate_oracle(
            pd.DataFrame(),
            _truth(),
            time_offset_s=True,
        )
    with pytest.raises(ValueError, match="max_time_delta_s"):
        nearest_candidate_oracle(
            pd.DataFrame(),
            _truth(),
            max_time_delta_s=np.nan,
        )


def test_time_offset_sweep_validates_controls_before_legacy_coercion() -> None:
    with pytest.raises(ValueError, match="time_offset_s"):
        time_offset_sweep(_radar(), _truth(), [True])
    with pytest.raises(ValueError, match="max_time_delta_s"):
        time_offset_sweep(
            _radar(),
            _truth(),
            [0.0],
            max_time_delta_s=np.nan,
        )


def test_valid_scalar_like_controls_and_unbounded_gate_remain_supported() -> None:
    positions, valid = interpolate_truth_positions(
        _truth(),
        [1.0],
        max_time_delta_s=np.array(1.0),
    )
    np.testing.assert_allclose(positions[0], [1.0, 0.0, 0.0])
    assert valid.tolist() == [True]

    selected = nearest_candidate_oracle(
        _radar(),
        _truth(),
        time_offset_s=np.array(0.0),
        max_time_delta_s=None,
    )
    assert selected["oracle_error_3d_m"].tolist() == pytest.approx([0.0])


@pytest.mark.parametrize(
    ("call", "kwargs"),
    [
        (
            aggregate_radar_time_offset_sweep,
            {"training_pairs": [(_radar(), _truth())], "offsets_s": [0.0]},
        ),
        (
            aggregate_measurement_time_offset_sweep,
            {
                "training_pairs": [(_radar(), _truth())],
                "offsets_s": [0.0],
                "dimensions": 3,
            },
        ),
        (
            fit_radar_time_offset,
            {"training_pairs": [(_radar(), _truth())], "offsets_s": [0.0]},
        ),
        (
            fit_measurement_time_offset,
            {
                "training_pairs": [(_radar(), _truth())],
                "offsets_s": [0.0],
                "dimensions": 3,
            },
        ),
    ],
)
def test_calibration_paths_reject_nan_freshness_gate(
    call: Callable[..., object],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="max_time_delta_s"):
        call(**kwargs, max_time_delta_s=np.nan)
