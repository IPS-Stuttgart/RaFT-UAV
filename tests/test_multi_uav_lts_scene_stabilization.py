import math

import numpy as np
from scipy import ndimage

from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.scene_stabilization import (
    StabilizationConfig,
    estimate_cumulative_translations,
    make_stabilized_geometry,
    phase_translation,
)
from raft_uav.multi_uav_lts.seeded_multiscan import MultiScanConfig, edge_cost


def detection(frame: int, identity: int, center_x: float, center_y: float = 20.0):
    return Detection(frame, identity, center_x - 3, center_y - 3, 6, 6, 0.9, 1, 1.0)


def test_phase_translation_recovers_known_shift():
    rng = np.random.default_rng(2)
    reference = ndimage.gaussian_filter(rng.normal(size=(96, 96)), 1.5)
    moving = ndimage.shift(reference, (5, -7), mode="nearest")
    estimate = phase_translation(
        reference,
        moving,
        StabilizationConfig(downsample=1, max_shift=20, min_peak_ratio=1.0),
    )
    assert estimate.accepted
    assert abs(estimate.dy + 5) < 0.4
    assert abs(estimate.dx - 7) < 0.4


def test_rejected_frame_is_not_promoted_to_registration_anchor():
    rng = np.random.default_rng(3)
    first = ndimage.gaussian_filter(rng.normal(size=(80, 80)), 1.2)
    textureless = np.zeros_like(first)
    third = ndimage.shift(first, (0, 6), mode="nearest")
    cumulative, diagnostics = estimate_cumulative_translations(
        {1: first, 2: textureless, 3: third},
        config=StabilizationConfig(
            downsample=1,
            max_shift=12,
            min_peak_ratio=0.8,
        ),
    )
    assert diagnostics["frames"]["2"]["accepted"] is False
    assert diagnostics["frames"]["3"]["from_frame"] == 1
    assert math.isclose(cumulative[3][1], -6.0, abs_tol=0.6)


def test_stabilized_geometry_reduces_camera_jitter_cost():
    geometry = make_stabilized_geometry({1: (0, 0), 2: (0, -20)})
    raw = edge_cost(
        detection(1, 1, 10),
        detection(2, 2, 30),
        MultiScanConfig(),
    )
    stabilized = edge_cost(
        detection(1, 1, 10),
        detection(2, 2, 30),
        MultiScanConfig(),
        geometry=geometry,
    )
    assert stabilized < raw
