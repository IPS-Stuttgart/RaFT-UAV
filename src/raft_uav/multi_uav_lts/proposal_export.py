"""Patch the external LTS inference script to export low-threshold proposals."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


class ProposalExportPatchError(RuntimeError):
    """Raised when the expected external inference source cannot be patched safely."""


@dataclass(frozen=True)
class ProposalExportPatchSummary:
    botsort_root: str
    inference_path: str
    check_only: bool
    needs_update: bool
    changed: bool
    backup_path: str | None


def apply_proposal_export_patch(
    botsort_root: Path,
    *,
    check_only: bool = False,
    create_backup: bool = True,
) -> ProposalExportPatchSummary:
    """Add an opt-in proposal export path to the external inference script."""

    root = botsort_root.resolve()
    inference_path = root / "tools" / "inference.py"
    if not inference_path.is_file():
        raise FileNotFoundError(inference_path)
    original = inference_path.read_text(encoding="utf-8")
    patched = _patch_inference(original)
    changed = patched != original
    backup_path: Path | None = None
    if changed and not check_only:
        if create_backup:
            backup_path = inference_path.with_suffix(
                inference_path.suffix + ".raft-uav-proposal-original"
            )
            if not backup_path.exists():
                backup_path.write_text(original, encoding="utf-8")
        temporary = inference_path.with_suffix(
            inference_path.suffix + ".raft-uav-proposal-tmp"
        )
        temporary.write_text(patched, encoding="utf-8")
        temporary.replace(inference_path)
    return ProposalExportPatchSummary(
        botsort_root=str(root),
        inference_path=str(inference_path),
        check_only=check_only,
        needs_update=changed,
        changed=changed,
        backup_path=str(backup_path) if backup_path is not None else None,
    )


def _patch_inference(text: str) -> str:
    text = _replace_once(
        text,
        "    os.makedirs(opt.save_path_answer, exist_ok=True)\n    res_list = []",
        "    os.makedirs(opt.save_path_answer, exist_ok=True)\n"
        "    res_list = []\n"
        "    proposal_handle = None\n"
        "    proposal_file = None\n"
        "    if opt.proposal_output_dir:\n"
        "        os.makedirs(opt.proposal_output_dir, exist_ok=True)\n"
        "        proposal_name, proposal_extension = os.path.splitext(foldern)\n"
        "        proposal_base_name = (\n"
        "            proposal_name if proposal_extension else foldern\n"
        "        )\n"
        "        proposal_file = os.path.join(\n"
        "            opt.proposal_output_dir, f\"{proposal_base_name}.txt\"\n"
        "        )\n"
        "        proposal_handle = open(\n"
        "            proposal_file, \"w\", encoding=\"utf-8\", buffering=1\n"
        "        )",
        label="proposal output initialization",
    )
    old_nms = (
        "        from ultralytics.utils.ops import non_max_suppression\n"
        "        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, "
        "classes=opt.classes, agnostic=opt.agnostic_nms)[0]  # Keep only the first "
        "image's detections"
    )
    new_nms = (
        "        from ultralytics.utils.ops import non_max_suppression\n"
        "        proposal_pred = None\n"
        "        if opt.proposal_output_dir:\n"
        "            proposal_conf_thres = (\n"
        "                opt.conf_thres\n"
        "                if opt.proposal_conf_thres is None\n"
        "                else opt.proposal_conf_thres\n"
        "            )\n"
        "            proposal_iou_thres = (\n"
        "                opt.iou_thres\n"
        "                if opt.proposal_iou_thres is None\n"
        "                else opt.proposal_iou_thres\n"
        "            )\n"
        "            proposal_input = (\n"
        "                pred.clone() if hasattr(pred, \"clone\") else copy.deepcopy(pred)\n"
        "            )\n"
        "            proposal_pred = non_max_suppression(\n"
        "                proposal_input,\n"
        "                proposal_conf_thres,\n"
        "                proposal_iou_thres,\n"
        "                classes=opt.classes,\n"
        "                agnostic=opt.agnostic_nms,\n"
        "            )[0]\n"
        "        pred = non_max_suppression(\n"
        "            pred,\n"
        "            opt.conf_thres,\n"
        "            opt.iou_thres,\n"
        "            classes=opt.classes,\n"
        "            agnostic=opt.agnostic_nms,\n"
        "        )[0]  # Keep only the first image's detections"
    )
    text = _replace_once(text, old_nms, new_nms, label="proposal NMS branch")
    export_block = (
        "        if proposal_pred is not None and proposal_pred.numel() != 0:\n"
        "            proposal_frame = im0s[0] if isinstance(im0s, list) else im0s\n"
        "            proposal_det = proposal_pred.clone()\n"
        "            proposal_det[:, :4] = scale_coords(\n"
        "                img.shape[2:], proposal_det[:, :4], proposal_frame.shape\n"
        "            )\n"
        "            for proposal_id, proposal in enumerate(\n"
        "                proposal_det.detach().cpu().tolist(), start=1\n"
        "            ):\n"
        "                x1, y1, x2, y2, score, class_index = proposal[:6]\n"
        "                width = float(x2 - x1)\n"
        "                height = float(y2 - y1)\n"
        "                if width <= 0.0 or height <= 0.0:\n"
        "                    continue\n"
        "                proposal_handle.write(\n"
        "                    \",\".join(\n"
        "                        map(\n"
        "                            str,\n"
        "                            [\n"
        "                                idx + 1,\n"
        "                                proposal_id,\n"
        "                                float(x1),\n"
        "                                float(y1),\n"
        "                                width,\n"
        "                                height,\n"
        "                                float(score),\n"
        "                                int(class_index) + 1,\n"
        "                                1.0,\n"
        "                            ],\n"
        "                        )\n"
        "                    )\n"
        "                    + \"\\n\"\n"
        "                )\n\n"
    )
    text = _replace_once(
        text,
        "        ################\n        # first frame use gt, no need to detect",
        export_block
        + "        ################\n        # first frame use gt, no need to detect",
        label="proposal row export",
    )
    write_block = (
        "    if proposal_handle is not None:\n"
        "        proposal_handle.close()\n"
        "        print('.proposal.txt saved to: {}'.format(proposal_file))\n\n"
    )
    text = _replace_once(
        text,
        "    if is_video_file(opt.source):\n        print('.mp4 saved to: {}'.format(save_dir))",
        write_block
        + "    if is_video_file(opt.source):\n"
        "        print('.mp4 saved to: {}'.format(save_dir))",
        label="proposal file write",
    )
    old_answer_option = (
        "    parser.add_argument('--save_path_answer', type=str, default=None, "
        "help='Path to save the label files. If not set, \"_label\" is appended "
        "to source.')"
    )
    new_answer_option = (
        old_answer_option
        + "\n"
        + "    parser.add_argument('--proposal-output-dir', type=str, default=None,\n"
        + "                        help='optional directory for low-threshold proposals')\n"
        + "    parser.add_argument('--proposal-conf-thres', type=float, default=None,\n"
        + "                        help='confidence threshold for proposal export')\n"
        + "    parser.add_argument('--proposal-iou-thres', type=float, default=None,\n"
        + "                        help='NMS IoU threshold for proposal export')"
    )
    text = _replace_once(
        text,
        old_answer_option,
        new_answer_option,
        label="proposal export CLI",
    )
    validation = (
        "    for option_name in ('proposal_conf_thres', 'proposal_iou_thres'):\n"
        "        option_value = getattr(opt, option_name)\n"
        "        if option_value is not None and not 0.0 <= option_value <= 1.0:\n"
        "            parser.error(\n"
        "                '--' + option_name.replace('_', '-') + ' must be in [0, 1]'\n"
        "            )\n\n"
    )
    text = _replace_once(
        text,
        "    opt.jde = False\n    opt.ablation = False",
        validation + "    opt.jde = False\n    opt.ablation = False",
        label="proposal threshold validation",
    )
    return text


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count == 1:
        return text.replace(old, new, 1)
    if count > 1:
        raise ProposalExportPatchError(
            f"{label}: expected one source block, found {count}"
        )
    raise ProposalExportPatchError(f"{label}: expected source block was not found")


def write_summary(summary: ProposalExportPatchSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("botsort_root", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    summary = apply_proposal_export_patch(
        args.botsort_root,
        check_only=args.check,
        create_backup=not args.no_backup,
    )
    if args.output_json:
        write_summary(summary, args.output_json)
    print(f"proposal_export_needs_update={summary.needs_update}")
    if args.check and summary.needs_update:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
