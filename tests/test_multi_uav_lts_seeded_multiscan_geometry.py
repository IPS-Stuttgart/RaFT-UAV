from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.scene_stabilization import make_stabilized_geometry
from raft_uav.multi_uav_lts.seeded_multiscan import (
    MultiScanConfig,
    associate_sequence,
)


def detection(frame: int, identity: int, center_x: float):
    return Detection(frame, identity, center_x - 3, 17, 6, 6, 0.9, 1, 1.0)


def test_geometry_extension_keeps_source_coordinate_boxes():
    seed = detection(1, 7, 10)
    proposal = detection(2, 1, 30)
    geometry = make_stabilized_geometry({1: (0, 0), 2: (0, -20)})
    rows, diagnostics = associate_sequence(
        [proposal],
        [seed],
        MultiScanConfig(dual_iterations=2),
        geometry=geometry,
    )
    linked = next(row for row in rows if row.frame_id == 2)
    assert linked.object_id == 7
    assert linked.center_x == 30
    assert diagnostics["stabilized_geometry"] is True
