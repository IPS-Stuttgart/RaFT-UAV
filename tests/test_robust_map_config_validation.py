from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.robust_map import (
    RobustMapSmootherConfig,
    robust_map_smooth_records,
)
from raft_uav.baselines.smoothing import smooth_tracking_records


@pytest.mark.parametrize(
    "field",
    [
        "loss_scale",
        "relative_tolerance",
        "measurement_time_tolerance_s",
        "process_position_floor_m",
        "process_velocity_floor_mps",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_robust_map_config_rejects_nonfinite_real_controls(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        RobustMapSmootherConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    ["loss_scale", "relative_tolerance"],
)
def test_robust_map_config_rejects_zero_positive_controls(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RobustMapSmootherConfig(**{field: 0.0})


@pytest.mark.parametrize(
    "field",
    [
        "measurement_time_tolerance_s",
        "process_position_floor_m",
        "process_velocity_floor_mps",
    ],
)
def test_robust_map_config_rejects_negative_nonnegative_controls(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RobustMapSmootherConfig(**{field: -1.0})


@pytest.mark.parametrize(
    "value",
    [True, 1.5, float("nan"), float("inf"), np.ma.masked, np.array([2])],
)
def test_robust_map_config_rejects_malformed_iteration_counts(value: object) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        RobustMapSmootherConfig(max_iterations=value)


@pytest.mark.parametrize("value", [0, 1, "false", None, np.array([False])])
def test_robust_map_config_rejects_nonboolean_accepted_only(value: object) -> None:
    with pytest.raises(ValueError, match="accepted_measurements_only"):
        RobustMapSmootherConfig(accepted_measurements_only=value)


def test_robust_map_config_normalizes_valid_scalar_like_values() -> None:
    config = RobustMapSmootherConfig(
        loss_scale=np.float64(2.0),
        max_iterations=np.array(4),
        relative_tolerance=np.float32(1.0e-4),
        measurement_time_tolerance_s=np.array(0.0),
        process_position_floor_m=np.float64(0.5),
        process_velocity_floor_mps=np.float32(0.25),
        accepted_measurements_only=np.bool_(True),
    )

    assert config.loss_scale == 2.0
    assert isinstance(config.loss_scale, float)
    assert config.max_iterations == 4
    assert isinstance(config.max_iterations, int)
    assert config.accepted_measurements_only is True


def test_direct_robust_map_rejects_falsy_config_before_empty_fast_path() -> None:
    with pytest.raises(TypeError, match="config"):
        robust_map_smooth_records(
            [],
            measurements=None,
            acceleration_std_mps2=1.0,
            config=False,
        )


def test_generic_robust_map_rejects_falsy_config_before_empty_fast_path() -> None:
    with pytest.raises(TypeError, match="robust_map_config"):
        smooth_tracking_records(
            [],
            method="robust-map",
            acceleration_std_mps2=1.0,
            robust_map_config={},
        )
