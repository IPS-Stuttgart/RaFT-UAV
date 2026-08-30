import numpy as np
from scipy import ndimage

from raft_uav.multi_uav_lts.registered_temporal_proposals import (
    RegisteredTemporalConfig,
    registered_residual_components,
)
from raft_uav.multi_uav_lts.scene_stabilization import StabilizationConfig


def test_registered_residual_recovers_middle_frame_target():
    rng = np.random.default_rng(4)
    background = ndimage.gaussian_filter(rng.normal(size=(64, 64)), 1.2)
    background = (background - background.min()) / (background.max() - background.min())
    current = background.copy()
    current[30:34, 29:33] += 0.8
    neighbours = [
        ndimage.shift(background, shift, mode="nearest")
        for shift in ((2, -1), (-2, 1), (1, 2), (-1, -2))
    ]
    components, diagnostics = registered_residual_components(
        current,
        neighbours,
        RegisteredTemporalConfig(
            robust_z=2.5,
            min_neighbours=2,
            registration=StabilizationConfig(
                downsample=1,
                max_shift=8,
                min_peak_ratio=0.8,
            ),
        ),
    )
    assert diagnostics["accepted_registrations"] >= 2
    assert any(
        x <= 31 <= x + width and y <= 32 <= y + height
        for x, y, width, height, _ in components
    )
