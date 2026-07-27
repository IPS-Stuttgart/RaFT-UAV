from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker, IMMMode


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        np.nan,
        np.inf,
        True,
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
        np.ma.array(1.0, mask=True),
    ],
)
def test_imm_mode_rejects_invalid_acceleration_standard_deviation(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="acceleration_std_mps2 must be a finite nonnegative real scalar",
    ):
        IMMMode("invalid-acceleration", value)


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([0.1]),
        np.ma.masked,
        np.ma.array(0.1, mask=True),
    ],
)
def test_imm_mode_rejects_invalid_turn_rate(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="turn_rate_radps must be a finite real scalar",
    ):
        IMMMode("invalid-turn-rate", 1.0, value)


@pytest.mark.parametrize("name", ["", "   ", 3, None])
def test_imm_mode_rejects_invalid_names(name: object) -> None:
    with pytest.raises(ValueError, match="IMM mode name must be a non-empty string"):
        IMMMode(name, 1.0)


def test_imm_mode_normalizes_valid_scalar_like_dynamics() -> None:
    mode = IMMMode(
        "valid",
        np.array(2.5),
        np.float64(-0.125),
    )

    assert mode.acceleration_std_mps2 == 2.5
    assert mode.turn_rate_radps == -0.125
    assert isinstance(mode.acceleration_std_mps2, float)
    assert isinstance(mode.turn_rate_radps, float)


def test_imm_tracker_rejects_duplicate_mode_names() -> None:
    modes = (
        IMMMode("duplicate", 1.0),
        IMMMode("duplicate", 2.0),
    )

    with pytest.raises(ValueError, match="IMM mode names must be unique"):
        AsyncInteractingMultipleModelTracker(
            initial_position=np.zeros(3),
            initial_time_s=0.0,
            modes=modes,
        )


def test_imm_tracker_preserves_distinct_mode_probability_keys() -> None:
    modes = (
        IMMMode("slow", 1.0),
        IMMMode("fast", 2.0),
    )
    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=np.zeros(3),
        initial_time_s=0.0,
        modes=modes,
        initial_mode_probabilities=(0.25, 0.75),
    )

    assert tracker.mode_names == ("slow", "fast")
    assert tracker.mode_probability_map == {"slow": 0.25, "fast": 0.75}
