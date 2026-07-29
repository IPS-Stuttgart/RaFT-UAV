from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.upstream_fixes import (
    UpstreamPatchError,
    apply_upstream_fixes,
)


INFERENCE = '''import torch
import torch.backends.cudnn as cudnn


def run():
        if pred[0].numel() != 0:
            res_list.append([
                idx, tid, round(tlwh[0], 2), round(tlwh[1], 2),
                round(tlwh[2], 2), round(tlwh[3], 2), 1, 1, 1
            ])
        else:

            # ===== Detect from Files =====
            pass

if __name__ == '__main__':
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")
    opt.jde = False
    opt.ablation = False
'''

TRACKER = '''class Tracker:
    def update(self):
        if not self.args.mot20:
            first()
        if not self.args.mot20:
            second()
        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
'''


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "BoT-SORT"
    (root / "tools").mkdir(parents=True)
    (root / "tracker").mkdir()
    (root / "tools" / "inference.py").write_text(INFERENCE, encoding="utf-8")
    (root / "tracker" / "mc_bot_sort.py").write_text(TRACKER, encoding="utf-8")
    return root


def test_applies_upstream_fixes_idempotently(tmp_path: Path) -> None:
    root = _checkout(tmp_path)

    first = apply_upstream_fixes(root)
    second = apply_upstream_fixes(root)

    assert first.changed_file_count == 2
    assert first.needs_update
    assert second.changed_file_count == 0
    assert not second.needs_update
    inference = (root / "tools" / "inference.py").read_text(encoding="utf-8")
    tracker = (root / "tracker" / "mc_bot_sort.py").read_text(encoding="utf-8")
    assert "import numpy as np" in inference
    assert "tracker.update(np.empty((0, 6), dtype=np.float32), im0s)" in inference
    assert 'dest="fuse_score", default=True' in inference
    assert '"--no-fuse-score"' in inference
    assert "round(tlwh[0], 2)" not in inference
    assert "opt.mot20 = not opt.fuse_score" in inference
    assert tracker.count("if not self.args.mot20:") == 2
    assert "if track.is_activated" in tracker
    assert (root / "tools" / "inference.py.raft-uav-original").exists()
    assert (root / "tracker" / "mc_bot_sort.py.raft-uav-original").exists()


def test_check_mode_does_not_modify_files(tmp_path: Path) -> None:
    root = _checkout(tmp_path)

    summary = apply_upstream_fixes(root, check_only=True)

    assert summary.needs_update
    assert summary.changed_file_count == 2
    assert (root / "tools" / "inference.py").read_text(encoding="utf-8") == INFERENCE
    assert (root / "tracker" / "mc_bot_sort.py").read_text(encoding="utf-8") == TRACKER


def test_validation_failure_does_not_partially_modify_checkout(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    tracker_path = root / "tracker" / "mc_bot_sort.py"
    malformed_tracker = "class Tracker:\n    pass\n"
    tracker_path.write_text(malformed_tracker, encoding="utf-8")

    with pytest.raises(UpstreamPatchError, match="confirmed-track output"):
        apply_upstream_fixes(root)

    assert (root / "tools" / "inference.py").read_text(encoding="utf-8") == INFERENCE
    assert tracker_path.read_text(encoding="utf-8") == malformed_tracker
    assert not (root / "tools" / "inference.py.raft-uav-original").exists()
