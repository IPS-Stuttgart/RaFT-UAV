import numpy as np
import torch
from scipy import ndimage

from raft_uav.multi_uav_lts.scene_stabilization import StabilizationConfig
from raft_uav.multi_uav_lts.temporal_p2_detector import (
    TemporalP2Config,
    TemporalP2Detector,
    temporal_feature_stack,
)


def test_temporal_feature_stack_exposes_middle_frame_residual():
    rng = np.random.default_rng(6)
    background = ndimage.gaussian_filter(rng.normal(size=(64, 64)), 1.2)
    background = (background - background.min()) / (background.max() - background.min())
    current = background.copy()
    current[31:35, 30:34] = 1.0
    neighbours = [
        ndimage.shift(background, shift, mode="nearest")
        for shift in ((1, -2), (-1, 2), (2, 1), (-2, -1))
    ]
    features, diagnostics = temporal_feature_stack(
        current,
        neighbours,
        TemporalP2Config(
            registration=StabilizationConfig(
                downsample=1,
                max_shift=8,
                min_peak_ratio=0.8,
            )
        ),
    )
    assert features.shape == (5, 64, 64)
    assert diagnostics["accepted_registrations"] >= 2
    assert features[3, 33, 32] > np.median(features[3]) + 0.2


def test_temporal_p2_model_is_stride_four_and_five_channel():
    model = TemporalP2Detector(channels=16, blocks=1)
    heat, offset, size = model(torch.zeros((1, 5, 64, 64)))
    assert heat.shape == (1, 1, 16, 16)
    assert offset.shape == (1, 2, 16, 16)
    assert size.shape == (1, 2, 16, 16)
