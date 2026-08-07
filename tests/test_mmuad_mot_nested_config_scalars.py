from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.mot import MultiObjectTrackerConfig


def _nested_zero_dimensional_object(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    "field_name",
    [
        "acceleration_std_mps2",
        "max_association_distance_m",
        "max_track_age_s",
        "min_new_track_confidence",
        "covariance_scale",
    ],
)
def test_mot_config_rejects_recursively_boxed_boolean_scalars(
    field_name: str,
) -> None:
    value = _nested_zero_dimensional_object(True)

    with pytest.raises(ValueError, match=field_name):
        MultiObjectTrackerConfig(**{field_name: value})


def test_mot_config_accepts_recursively_boxed_finite_real_scalars() -> None:
    config = MultiObjectTrackerConfig(
        acceleration_std_mps2=_nested_zero_dimensional_object(8.0),
        max_association_distance_m=_nested_zero_dimensional_object(15.0),
        max_track_age_s=_nested_zero_dimensional_object(1.5),
        min_new_track_confidence=_nested_zero_dimensional_object(0.05),
        covariance_scale=_nested_zero_dimensional_object(1.0),
    )

    assert config == MultiObjectTrackerConfig()
