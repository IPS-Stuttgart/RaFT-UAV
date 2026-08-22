"""Patch the external LTS inference script with proposal-only tiled detector passes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


class TiledProposalPatchError(RuntimeError):
    """Raised when the proposal-exported upstream source cannot be patched safely."""


@dataclass(frozen=True)
class TiledProposalPatchSummary:
    botsort_root: str
    inference_path: str
    check_only: bool
    needs_update: bool
    changed: bool
    backup_path: str | None


def apply_tiled_proposal_patch(
    botsort_root: Path,
    *,
    check_only: bool = False,
    create_backup: bool = True,
) -> TiledProposalPatchSummary:
    """Add an opt-in tiled proposal pass without changing normal tracker detections."""

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
                inference_path.suffix + ".raft-uav-tiled-proposal-original"
            )
            if not backup_path.exists():
                backup_path.write_text(original, encoding="utf-8")
        temporary = inference_path.with_suffix(
            inference_path.suffix + ".raft-uav-tiled-proposal-tmp"
        )
        temporary.write_text(patched, encoding="utf-8")
        temporary.replace(inference_path)
    return TiledProposalPatchSummary(
        botsort_root=str(root),
        inference_path=str(inference_path),
        check_only=check_only,
        needs_update=changed,
        changed=changed,
        backup_path=str(backup_path) if backup_path is not None else None,
    )


def _patch_inference(text: str) -> str:
    if "--proposal-output-dir" not in text or "proposal_handle.write" not in text:
        raise TiledProposalPatchError(
            "tiled proposal export requires the low-threshold proposal patch first"
        )
    text = _replace_once(
        text,
        "from numpy import random",
        "import numpy as np\nfrom numpy import random",
        label="NumPy import",
    )
    tile_block = (
        "        if (\n"
        "            opt.proposal_output_dir\n"
        "            and opt.proposal_tile_size > 0\n"
        "            and not os.path.isdir(weights[0])\n"
        "        ):\n"
        "            tile_frame = im0s[0] if isinstance(im0s, list) else im0s\n"
        "            tile_frame_height, tile_frame_width = tile_frame.shape[:2]\n"
        "            tile_size = min(\n"
        "                opt.proposal_tile_size, tile_frame_height, tile_frame_width\n"
        "            )\n"
        "            tile_step = max(\n"
        "                1, int(round(tile_size * (1.0 - opt.proposal_tile_overlap)))\n"
        "            )\n"
        "            tile_x_starts = list(\n"
        "                range(0, max(1, tile_frame_width - tile_size + 1), tile_step)\n"
        "            )\n"
        "            tile_y_starts = list(\n"
        "                range(0, max(1, tile_frame_height - tile_size + 1), tile_step)\n"
        "            )\n"
        "            final_x_start = tile_frame_width - tile_size\n"
        "            final_y_start = tile_frame_height - tile_size\n"
        "            if not tile_x_starts or tile_x_starts[-1] != final_x_start:\n"
        "                tile_x_starts.append(final_x_start)\n"
        "            if not tile_y_starts or tile_y_starts[-1] != final_y_start:\n"
        "                tile_y_starts.append(final_y_start)\n"
        "            tile_rows = []\n"
        "            for tile_y in tile_y_starts:\n"
        "                for tile_x in tile_x_starts:\n"
        "                    tile_crop = tile_frame[\n"
        "                        tile_y : tile_y + tile_size,\n"
        "                        tile_x : tile_x + tile_size,\n"
        "                    ]\n"
        "                    tile_input = cv2.resize(\n"
        "                        tile_crop, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR\n"
        "                    )\n"
        "                    tile_input = tile_input[:, :, ::-1].transpose(2, 0, 1)\n"
        "                    tile_input = np.ascontiguousarray(tile_input)\n"
        "                    tile_tensor = torch.from_numpy(tile_input).to(device)\n"
        "                    tile_tensor = (\n"
        "                        tile_tensor.half() if half else tile_tensor.float()\n"
        "                    )\n"
        "                    tile_tensor /= 255.0\n"
        "                    tile_tensor = tile_tensor.unsqueeze(0)\n"
        "                    tile_raw = model(tile_tensor, augment=opt.augment)[0]\n"
        "                    tile_det = non_max_suppression(\n"
        "                        tile_raw,\n"
        "                        proposal_conf_thres,\n"
        "                        proposal_iou_thres,\n"
        "                        classes=opt.classes,\n"
        "                        agnostic=opt.agnostic_nms,\n"
        "                    )[0]\n"
        "                    if tile_det is None or tile_det.numel() == 0:\n"
        "                        continue\n"
        "                    tile_det = tile_det.clone()\n"
        "                    tile_det[:, :4] = scale_coords(\n"
        "                        tile_tensor.shape[2:], tile_det[:, :4], tile_crop.shape\n"
        "                    )\n"
        "                    for tile_proposal in tile_det.detach().cpu().tolist():\n"
        "                        x1, y1, x2, y2, score, class_index = tile_proposal[:6]\n"
        "                        x1 += tile_x\n"
        "                        x2 += tile_x\n"
        "                        y1 += tile_y\n"
        "                        y2 += tile_y\n"
        "                        width = float(x2 - x1)\n"
        "                        height = float(y2 - y1)\n"
        "                        if width <= 0.0 or height <= 0.0:\n"
        "                            continue\n"
        "                        tile_rows.append(\n"
        "                            (\n"
        "                                float(score),\n"
        "                                float(x1),\n"
        "                                float(y1),\n"
        "                                width,\n"
        "                                height,\n"
        "                                int(class_index) + 1,\n"
        "                            )\n"
        "                        )\n"
        "            tile_rows.sort(key=lambda row: (-row[0], row[2], row[1]))\n"
        "            for tile_offset, tile_row in enumerate(\n"
        "                tile_rows[: opt.proposal_tile_max_per_frame], start=1\n"
        "            ):\n"
        "                score, x1, y1, width, height, class_id = tile_row\n"
        "                proposal_handle.write(\n"
        "                    \",\".join(\n"
        "                        map(\n"
        "                            str,\n"
        "                            [\n"
        "                                idx + 1,\n"
        "                                1000000 + tile_offset,\n"
        "                                x1,\n"
        "                                y1,\n"
        "                                width,\n"
        "                                height,\n"
        "                                score,\n"
        "                                class_id,\n"
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
        tile_block
        + "        ################\n        # first frame use gt, no need to detect",
        label="tiled proposal inference block",
    )
    old_option = (
        "    parser.add_argument('--proposal-iou-thres', type=float, default=None,\n"
        "                        help='NMS IoU threshold for proposal export')"
    )
    new_option = (
        old_option
        + "\n"
        + "    parser.add_argument('--proposal-tile-size', type=int, default=0,\n"
        + "                        help='square crop size for extra proposal-only inference')\n"
        + "    parser.add_argument('--proposal-tile-overlap', type=float, default=0.25,\n"
        + "                        help='fractional overlap between proposal tiles')\n"
        + "    parser.add_argument('--proposal-tile-max-per-frame', type=int, default=256,\n"
        + "                        help='maximum extra tiled proposals per frame')"
    )
    text = _replace_once(
        text,
        old_option,
        new_option,
        label="tiled proposal CLI",
    )
    validation = (
        "    if opt.proposal_tile_size < 0:\n"
        "        parser.error('--proposal-tile-size must be non-negative')\n"
        "    if not 0.0 <= opt.proposal_tile_overlap < 1.0:\n"
        "        parser.error('--proposal-tile-overlap must be in [0, 1)')\n"
        "    if opt.proposal_tile_max_per_frame <= 0:\n"
        "        parser.error('--proposal-tile-max-per-frame must be positive')\n\n"
    )
    text = _replace_once(
        text,
        "    opt.jde = False\n    opt.ablation = False",
        validation + "    opt.jde = False\n    opt.ablation = False",
        label="tiled proposal validation",
    )
    return text


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count == 1:
        return text.replace(old, new, 1)
    if count > 1:
        raise TiledProposalPatchError(
            f"{label}: expected one source block, found {count}"
        )
    raise TiledProposalPatchError(f"{label}: expected source block was not found")


def write_summary(summary: TiledProposalPatchSummary, path: Path) -> None:
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
    summary = apply_tiled_proposal_patch(
        args.botsort_root,
        check_only=args.check,
        create_backup=not args.no_backup,
    )
    if args.output_json:
        write_summary(summary, args.output_json)
    print(f"tiled_proposal_needs_update={summary.needs_update}")
    if args.check and summary.needs_update:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
