from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.adaptive_kinematic import smooth_track


def detection(frame: int, center_x: float):
    return Detection(frame, 1, center_x - 3, 17, 6, 6, 0.9, 1, 1.0)


def test_adaptive_smoother_downweights_large_outlier():
    rows = [
        detection(frame, center)
        for frame, center in enumerate((10, 12, 80, 16, 18), 1)
    ]
    smoothed = smooth_track(rows)
    assert 10 < smoothed[2].center_x < 60


def test_adaptive_smoother_clips_output_to_image():
    rows = [detection(1, 98), detection(2, 102), detection(3, 106)]
    smoothed = smooth_track(rows, image_shape=(100, 100))
    assert all(row.x1 + row.width <= 100 for row in smoothed)
