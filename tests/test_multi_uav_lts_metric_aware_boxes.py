import math

from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.metric_aware_boxes import (
    MetricBoxConfig,
    transform_sequence,
)


def detection(frame: int, center_x: float, width: float = 5.0):
    return Detection(
        frame,
        1,
        center_x - width / 2,
        20 - width / 2,
        width,
        width,
        0.9,
        1,
        1.0,
    )


def test_metric_boxes_preserve_center_and_fill_short_gap():
    rows = [detection(1, 10), detection(3, 14)]
    transformed, diagnostics = transform_sequence(
        rows,
        MetricBoxConfig(max_interpolation_gap=1, maximum_scale=1.8),
        image_shape=(100, 100),
    )
    by_frame = {row.frame_id: row for row in transformed}
    assert set(by_frame) == {1, 2, 3}
    assert math.isclose(by_frame[1].center_x, 10)
    assert by_frame[1].width > 5
    assert diagnostics["interpolation_count"] == 1


def test_fixed_expansion_is_clipped_at_image_boundary():
    transformed, _ = transform_sequence(
        [detection(1, 98, width=4)],
        MetricBoxConfig(
            base_scale=2.0,
            tiny_gain=0,
            innovation_gain=0,
            gap_gain=0,
            maximum_scale=2.0,
        ),
        image_shape=(100, 100),
    )
    assert transformed[0].x1 + transformed[0].width <= 100


def test_guarded_maximum_scale_retains_half_iou_at_same_center():
    scale = MetricBoxConfig().maximum_scale
    same_center_iou = 1.0 / (scale * scale)
    assert scale < math.sqrt(2.0)
    assert same_center_iou > 0.5
