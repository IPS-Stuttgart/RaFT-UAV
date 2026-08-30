import math

from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.formation_reassociation import (
    bidirectional_refine_sequence,
    refine_sequence,
)
from raft_uav.multi_uav_lts.scene_stabilization import make_stabilized_geometry


def detection(frame: int, identity: int, center_x: float):
    return Detection(frame, identity, center_x - 3, 17, 6, 6, 0.9, 1, 1.0)


def crossing_rows():
    seeds = [detection(1, 1, 10), detection(1, 2, 30)]
    rows = [
        *seeds,
        detection(2, 1, 14),
        detection(2, 2, 26),
        detection(3, 1, 22),
        detection(3, 2, 18),
        detection(4, 1, 18),
        detection(4, 2, 22),
    ]
    return seeds, rows


def test_formation_reassociation_repairs_crossing_ids():
    seeds, rows = crossing_rows()
    refined, diagnostics = refine_sequence(rows, seeds)
    by_key = {(row.frame_id, row.object_id): row for row in refined}
    assert math.isclose(by_key[(3, 1)].center_x, 18)
    assert math.isclose(by_key[(3, 2)].center_x, 22)
    assert diagnostics["relabels"] >= 2


def test_bidirectional_consensus_retains_sequence_coverage():
    seeds, rows = crossing_rows()
    consensus, diagnostics = bidirectional_refine_sequence(rows, seeds)
    assert len(consensus) == len(rows)
    assert diagnostics["forward_frames_selected"] + diagnostics["backward_frames_selected"] == 4


def test_formation_motion_cost_accepts_stabilized_geometry():
    seeds = [detection(1, 1, 10), detection(1, 2, 30)]
    rows = [
        *seeds,
        detection(2, 1, 30),
        detection(2, 2, 50),
    ]
    geometry = make_stabilized_geometry({1: (0, 0), 2: (0, -20)})
    refined, diagnostics = refine_sequence(rows, seeds, geometry=geometry)
    assert len(refined) == len(rows)
    assert diagnostics["stabilized_geometry"] is True
