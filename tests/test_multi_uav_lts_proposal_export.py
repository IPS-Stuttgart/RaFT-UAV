from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.proposal_export import (
    ProposalExportPatchError,
    apply_proposal_export_patch,
)


INFERENCE = (
    "import os\n"
    "import copy\n\n\n"
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

CURRENT_INFERENCE = INFERENCE.replace(
    "    res_list = []\n",
    "    if opt.save_score_sidecar:\n"
    "        os.makedirs(opt.save_score_sidecar, exist_ok=True)\n"
    "    res_list = []\n"
    "    score_sidecar_rows = []\n",
)


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "BoT-SORT"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "inference.py").write_text(INFERENCE, encoding="utf-8")
    return root


def _checkout_current(tmp_path: Path) -> Path:
    root = tmp_path / "BoT-SORT"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "inference.py").write_text(
        CURRENT_INFERENCE,
        encoding="utf-8",
    )
    return root


def test_applies_proposal_export_patch_idempotently(tmp_path: Path) -> None:
    root = _checkout(tmp_path)

    first = apply_proposal_export_patch(root)
    second = apply_proposal_export_patch(root)

    source = (root / "tools" / "inference.py").read_text(encoding="utf-8")
    assert first.changed
    assert first.needs_update
    assert not second.changed
    assert not second.needs_update
    assert "--proposal-output-dir" in source
    assert "proposal_conf_thres" in source
    assert "proposal_det.detach().cpu().tolist()" in source
    assert "proposal_handle.write" in source
    assert "proposal_handle.close()" in source
    compile(source, "inference.py", "exec")
    assert (root / "tools" / "inference.py.raft-uav-proposal-original").is_file()


def test_check_mode_does_not_modify_external_source(tmp_path: Path) -> None:
    root = _checkout(tmp_path)

    summary = apply_proposal_export_patch(root, check_only=True)

    assert summary.needs_update
    assert (root / "tools" / "inference.py").read_text(encoding="utf-8") == INFERENCE
    assert not (root / "tools" / "inference.py.raft-uav-proposal-original").exists()


def test_patch_accepts_current_score_sidecar_initialization(tmp_path: Path) -> None:
    root = _checkout_current(tmp_path)

    summary = apply_proposal_export_patch(root)

    source = (root / "tools" / "inference.py").read_text(encoding="utf-8")
    assert summary.changed
    assert "score_sidecar_rows = []" in source
    assert source.index("proposal_handle = None") < source.index(
        "score_sidecar_rows = []"
    )
    compile(source, "inference.py", "exec")


def test_source_drift_fails_before_modification(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    inference_path = root / "tools" / "inference.py"
    inference_path.write_text("pass\n", encoding="utf-8")

    with pytest.raises(ProposalExportPatchError, match="initialization"):
        apply_proposal_export_patch(root)

    assert inference_path.read_text(encoding="utf-8") == "pass\n"
