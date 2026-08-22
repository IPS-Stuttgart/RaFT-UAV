from __future__ import annotations

import numpy as np

from raft_uav.multi_uav_lts.temporal_residual_proposals import (
    TemporalResidualParameters,
    _translate_with_nan,
    estimate_translation,
    temporal_residual_frame,
)


def _shift_without_wrap(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = _translate_with_nan(image, dy, dx)
    return np.nan_to_num(shifted, nan=0.0)


def test_phase_registration_recovers_integer_translation() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(96, 128))
    moving = _shift_without_wrap(reference, 5, -7)

    dy, dx = estimate_translation(reference, moving, stride=1)

    assert abs(dy + 5.0) <= 1.0
    assert abs(dx - 7.0) <= 1.0


def test_temporal_residual_detects_moving_tiny_target_after_camera_shift() -> None:
    rng = np.random.default_rng(8)
    background = rng.uniform(0.1, 0.7, size=(96, 128))
    previous = background.copy()
    current = _shift_without_wrap(background, 2, 3)
    current[45:49, 67:71] = 1.0
    parameters = TemporalResidualParameters(
        registration_stride=1,
        max_registration_shift_px=20.0,
        residual_sigma_floor=0.005,
        smooth_sigma_px=0.5,
        z_threshold=4.0,
        min_component_area_px=1,
        max_component_area_px=100,
        box_padding_px=2.0,
        max_proposals_per_frame=20,
        bidirectional=False,
    )

    proposals, shifts = temporal_residual_frame(
        previous,
        current,
        None,
        parameters=parameters,
    )

    assert shifts
    assert proposals
    assert any(
        x1 <= 69.0 <= x1 + width and y1 <= 47.0 <= y1 + height
        for x1, y1, width, height, _confidence in proposals
    )


def test_residual_detector_returns_nothing_without_temporal_neighbor() -> None:
    current = np.zeros((32, 32), dtype=float)
    proposals, shifts = temporal_residual_frame(
        None,
        current,
        None,
        parameters=TemporalResidualParameters(),
    )
    assert proposals == ()
    assert shifts == ()
