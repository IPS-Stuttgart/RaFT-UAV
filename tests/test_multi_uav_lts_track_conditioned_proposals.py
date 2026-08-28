from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.track_conditioned_proposals import RoiConfig, predict_roi

def row(frame, x):
    return Detection(frame, 1, x, 20, 8, 8, 0.9, 1, 1.0)

def test_roi_extrapolates_motion_and_clips():
    x, y, w, h = predict_roi([row(1, 10), row(2, 14), row(3, 18)], 4, (80, 100), RoiConfig(min_margin=5, box_scale=2, upscale=1, max_roi_side=200))
    assert x + w / 2 > 20
    assert 0 <= x < x + w <= 100 and 0 <= y < y + h <= 80
