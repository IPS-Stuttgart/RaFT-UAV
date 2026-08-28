from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.seeded_multiscan import MultiScanConfig, associate_sequence

def d(frame, obj, x, score=0.9):
    return Detection(frame, obj, x, 10, 6, 6, score, 1, 1.0)

def test_seeded_paths_preserve_ids_and_do_not_share_proposals():
    seeds = [d(1, 1, 10), d(1, 2, 40)]
    proposals = [d(2, 1, 13), d(2, 2, 37), d(3, 3, 16), d(3, 4, 34)]
    output, diag = associate_sequence(proposals, seeds, MultiScanConfig(min_birth_hits=99))
    keys = [(r.frame_id, r.object_id) for r in output]
    assert (1, 1) in keys and (1, 2) in keys
    assert len([(r.frame_id, round(r.center_x, 2)) for r in output]) == len(set(((r.frame_id, round(r.center_x, 2)) for r in output)))
    assert diag['seed_count'] == 2
