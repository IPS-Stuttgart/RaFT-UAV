import numpy as np
from raft_uav.multi_uav_lts.temporal_roi_proposals import TemporalRoiConfig, phase_translation, residual_components

def test_phase_translation_recovers_integer_shift():
    a = np.zeros((32, 32))
    a[10:14, 12:16] = 1
    b = np.roll(a, (3, -2), (0, 1))
    dy, dx = phase_translation(a, b)
    assert (dy, dx) == (-3, 2)

def test_temporal_component_detects_new_blob():
    current = np.zeros((32, 32))
    current[15:18, 16:19] = 1
    result = residual_components(current, [np.zeros_like(current), np.zeros_like(current)], TemporalRoiConfig(robust_z=1, min_area=2))
    assert result and result[0][2] >= 3
