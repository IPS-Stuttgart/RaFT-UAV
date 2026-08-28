import numpy as np
from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.imm_trajectory import ImmConfig, smooth_track

def test_imm_reduces_center_jitter():
    xs = [10, 12, 13, 17, 17, 21, 22]
    rows = [Detection(i + 1, 1, x, 10, 8, 8, 0.8, 1, 1.0) for i, x in enumerate(xs)]
    out = smooth_track(rows, ImmConfig(process_noise=0.2, measurement_noise=2))
    before = np.var(np.diff([r.center_x for r in rows], 2))
    after = np.var(np.diff([r.center_x for r in out], 2))
    assert after < before
    assert [r.object_id for r in out] == [1] * len(rows)
