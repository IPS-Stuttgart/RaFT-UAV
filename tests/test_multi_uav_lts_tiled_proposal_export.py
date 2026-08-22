from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.proposal_export import apply_proposal_export_patch
from raft_uav.multi_uav_lts.tiled_proposal_export import (
    TiledProposalPatchError,
    apply_tiled_proposal_patch,
)


INFERENCE = (
    "import os\n"
    "import copy\n"
    "from numpy import random\n\n\n"
    "def run():\n"
    "    os.makedirs(opt.save_path_answer, exist_ok=True)\n"
    "    res_list = []\n"
    "    for path, img, im0s, vid_cap in dataset:\n"
    "        from ultralytics.utils.ops import non_max_suppression\n"
    "        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, "
    "classes=opt.classes, agnostic=opt.agnostic_nms)[0]  # Keep only the first "
    "image's detections\n"
    "        ################\n"
    "        # first frame use gt, no need to detect\n"
    "        idx += 1\n"
    "    answer_file = os.path.join(opt.save_path_answer, f\"{base_name}.txt\")\n"
    "    with open(answer_file, \"w\") as file:\n"
    "        for row in res_list:\n"
    "            file.write(\",\".join(map(str, row)) + \"\\n\")\n"
    "    if is_video_file(opt.source):\n"
    "        print('.mp4 saved to: {}'.format(save_dir))\n\n"
    "if __name__ == '__main__':\n"
    "    parser.add_argument('--save_path_answer', type=str, default=None, "
    "help='Path to save the label files. If not set, \"_label\" is appended "
    "to source.')\n"
    "    opt = parser.parse_args()\n"
    "    opt.jde = False\n"
    "    opt.ablation = False\n"
)


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "BoT-SORT"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "inference.py").write_text(INFERENCE, encoding="utf-8")
    return root


def test_tiled_patch_is_idempotent_after_proposal_export(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    apply_proposal_export_patch(root)

    first = apply_tiled_proposal_patch(root)
    second = apply_tiled_proposal_patch(root)

    source = (root / "tools" / "inference.py").read_text(encoding="utf-8")
    assert first.changed
    assert first.needs_update
    assert not second.changed
    assert not second.needs_update
    assert "import numpy as np" in source
    assert "--proposal-tile-size" in source
    assert "proposal_tile_overlap" in source
    assert "tile_raw = model(tile_tensor" in source
    assert "1000000 + tile_offset" in source
    assert "proposal_handle.write" in source
    compile(source, "inference.py", "exec")
    assert (root / "tools" / "inference.py.raft-uav-tiled-proposal-original").is_file()


def test_tiled_patch_requires_low_threshold_export_first(tmp_path: Path) -> None:
    root = _checkout(tmp_path)

    with pytest.raises(TiledProposalPatchError, match="requires"):
        apply_tiled_proposal_patch(root)

    assert (root / "tools" / "inference.py").read_text(encoding="utf-8") == INFERENCE


def test_tiled_patch_check_mode_does_not_modify_source(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    apply_proposal_export_patch(root)
    before = (root / "tools" / "inference.py").read_text(encoding="utf-8")

    summary = apply_tiled_proposal_patch(root, check_only=True)

    assert summary.needs_update
    assert (root / "tools" / "inference.py").read_text(encoding="utf-8") == before
