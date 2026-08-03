"""Apply deterministic LTS fixes to the external YOLOv12-BoT-SORT checkout."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


class UpstreamPatchError(RuntimeError):
    """Raised when the expected upstream source layout cannot be patched safely."""


@dataclass(frozen=True)
class PatchedFile:
    path: str
    changed: bool
    backup_path: str | None


@dataclass(frozen=True)
class UpstreamFixSummary:
    botsort_root: str
    check_only: bool
    needs_update: bool
    changed_file_count: int
    files: tuple[PatchedFile, ...]


@dataclass(frozen=True)
class _PreparedPatch:
    path: Path
    original: str
    patched: str

    @property
    def changed(self) -> bool:
        return self.patched != self.original


def apply_upstream_fixes(
    botsort_root: Path,
    *,
    check_only: bool = False,
    create_backups: bool = True,
) -> UpstreamFixSummary:
    """Patch the known upstream inference and tracker files idempotently."""

    root = botsort_root.resolve()
    targets = (
        (root / "tools" / "inference.py", _patch_inference),
        (root / "tracker" / "mc_bot_sort.py", _patch_tracker),
    )
    prepared: list[_PreparedPatch] = []
    for path, patcher in targets:
        if not path.is_file():
            raise FileNotFoundError(path)
        original = path.read_text(encoding="utf-8")
        prepared.append(_PreparedPatch(path, original, patcher(original)))

    results: list[PatchedFile] = []
    for patch in prepared:
        backup_path: Path | None = None
        if patch.changed and not check_only:
            if create_backups:
                backup_path = patch.path.with_suffix(
                    patch.path.suffix + ".raft-uav-original"
                )
                if not backup_path.exists():
                    backup_path.write_text(patch.original, encoding="utf-8")
            temporary = patch.path.with_suffix(patch.path.suffix + ".raft-uav-tmp")
            temporary.write_text(patch.patched, encoding="utf-8")
            temporary.replace(patch.path)
        results.append(
            PatchedFile(
                path=str(patch.path),
                changed=patch.changed,
                backup_path=str(backup_path) if backup_path is not None else None,
            )
        )
    return UpstreamFixSummary(
        botsort_root=str(root),
        check_only=check_only,
        needs_update=any(result.changed for result in results),
        changed_file_count=sum(result.changed for result in results),
        files=tuple(results),
    )


def _patch_inference(text: str) -> str:
    text = _replace_once(
        text,
        "import torch\nimport torch.backends.cudnn as cudnn",
        "import torch\nimport numpy as np\nimport torch.backends.cudnn as cudnn",
        label="NumPy import",
    )
    text = _replace_once(
        text,
        "        else:\n\n            # ===== Detect from Files =====",
        "        else:\n"
        "            # Empty detector frames must advance the tracker clock and loss ages.\n"
        "            tracker.update(np.empty((0, 6), dtype=np.float32), im0s)\n\n"
        "            # ===== Detect from Files =====",
        label="empty-frame tracker update",
    )
    old_option = (
        '    parser.add_argument("--fuse-score", dest="mot20", default=False, '
        'action="store_true",\n'
        '                        help="fuse score and iou for association")'
    )
    new_option = (
        '    parser.add_argument("--fuse-score", dest="fuse_score", default=True, '
        'action="store_true",\n'
        '                        help="fuse detection score and IoU during association")\n'
        '    parser.add_argument("--no-fuse-score", dest="fuse_score", '
        'action="store_false",\n'
        '                        help="disable detection-score fusion during association")'
    )
    text = _replace_once(text, old_option, new_option, label="score-fusion CLI")
    text = _replace_once(
        text,
        "    opt.jde = False\n    opt.ablation = False",
        "    opt.jde = False\n"
        "    opt.ablation = False\n"
        "    # Preserve BoT-SORT's legacy mot20 field while exposing clear CLI semantics.\n"
        "    opt.mot20 = not opt.fuse_score",
        label="score-fusion compatibility assignment",
    )
    text = _replace_once(
        text,
        "        if opt.with_pos and idx == 0:",
        "        frame_initial_track_ids = None\n"
        "        if opt.with_pos and idx == 0:",
        label="initial-track ID frame state",
    )
    text = _replace_once(
        text,
        '            prior_box = []\n\n            with open(gt_path, "r") as file:',
        '            prior_box = []\n'
        '            frame_initial_track_ids = []\n\n'
        '            with open(gt_path, "r") as file:',
        label="initial-track ID collection",
    )
    text = _replace_once(
        text,
        "                    obj_id = int(values[0])  # Extract ID",
        "                    obj_id = int(values[1])  # Extract object ID\n"
        "                    frame_initial_track_ids.append(obj_id)",
        label="initial-track ID append",
    )
    text = _replace_once(
        text,
        "                online_targets, slosts_targets = tracker.update(detections, im0)",
        "                online_targets, slosts_targets = tracker.update(\n"
        "                    detections,\n"
        "                    im0,\n"
        "                    initial_track_ids=frame_initial_track_ids,\n"
        "                )",
        label="initial-track ID tracker call",
    )
    for index in range(4):
        old = f"round(tlwh[{index}], 2)"
        new = f"float(tlwh[{index}])"
        if old in text:
            text = text.replace(old, new)
        elif new not in text:
            raise UpstreamPatchError(
                f"cannot locate coordinate precision expression for tlwh[{index}]"
            )
    return text


def _patch_tracker(text: str) -> str:
    text = _replace_once(
        text,
        "    def activate(self, kalman_filter, frame_id):\n"
        '        """Start a new tracklet"""\n'
        "        self.kalman_filter = kalman_filter\n"
        "        self.track_id = self.next_id()",
        "    def activate(self, kalman_filter, frame_id, forced_track_id=None):\n"
        '        """Start a tracklet, optionally with a benchmark-supplied ID."""\n'
        "        self.kalman_filter = kalman_filter\n"
        "        if forced_track_id is None:\n"
        "            self.track_id = self.next_id()\n"
        "        else:\n"
        "            if isinstance(forced_track_id, (bool, np.bool_)) or not isinstance(\n"
        "                forced_track_id, (int, np.integer)\n"
        "            ):\n"
        '                raise ValueError("forced_track_id must be a positive integer")\n'
        "            forced_track_id = int(forced_track_id)\n"
        "            if forced_track_id <= 0:\n"
        '                raise ValueError("forced_track_id must be a positive integer")\n'
        "            self.track_id = forced_track_id\n"
        "            BaseTrack._count = max(BaseTrack._count, forced_track_id)",
        label="forced track activation",
    )
    text = _replace_once(
        text,
        "    def update(self, output_results, img):\n"
        "        self.frame_id += 1\n"
        "        activated_starcks = []",
        "    def update(self, output_results, img, initial_track_ids=None):\n"
        "        self.frame_id += 1\n"
        "        if initial_track_ids is not None:\n"
        "            if self.frame_id != 1:\n"
        "                raise ValueError(\n"
        '                    "initial_track_ids may only be supplied on tracker frame 1"\n'
        "                )\n"
        "            initial_track_ids = list(initial_track_ids)\n"
        "            if len(initial_track_ids) != len(output_results):\n"
        "                raise ValueError(\n"
        '                    "initial_track_ids must match the number of input detections"\n'
        "                )\n"
        "            normalized_track_ids = []\n"
        "            for track_id in initial_track_ids:\n"
        "                if isinstance(track_id, (bool, np.bool_)) or not isinstance(\n"
        "                    track_id, (int, np.integer)\n"
        "                ):\n"
        "                    raise ValueError(\n"
        '                        "initial_track_ids must contain positive integers"\n'
        "                    )\n"
        "                track_id = int(track_id)\n"
        "                if track_id <= 0:\n"
        "                    raise ValueError(\n"
        '                        "initial_track_ids must contain positive integers"\n'
        "                    )\n"
        "                normalized_track_ids.append(track_id)\n"
        "            if len(set(normalized_track_ids)) != len(normalized_track_ids):\n"
        '                raise ValueError("initial_track_ids must be unique")\n'
        "            initial_track_ids = np.asarray(\n"
        "                normalized_track_ids, dtype=np.int64\n"
        "            )\n"
        "        initial_track_ids_keep = None\n"
        "        activated_starcks = []",
        label="initial-track ID validation",
    )
    text = _replace_once(
        text,
        "            features_keep = features[remain_inds]\n"
        "        else:\n"
        "            bboxes = []",
        "            features_keep = features[remain_inds]\n"
        "            if initial_track_ids is not None:\n"
        "                if not np.all(lowest_inds) or not np.all(remain_inds):\n"
        "                    raise ValueError(\n"
        '                        "every initialized track must pass the tracking thresholds"\n'
        "                    )\n"
        "                initial_track_ids_keep = initial_track_ids\n"
        "        else:\n"
        "            bboxes = []",
        label="initial-track ID threshold preservation",
    )
    text = _replace_once(
        text,
        "        else:\n"
        "            detections = []\n\n"
        "        ''' Add newly detected tracklets to tracked_stracks'''",
        "        else:\n"
        "            detections = []\n\n"
        "        if initial_track_ids_keep is not None:\n"
        "            for detection, forced_track_id in zip(\n"
        "                detections, initial_track_ids_keep\n"
        "            ):\n"
        "                detection.forced_track_id = int(forced_track_id)\n\n"
        "        ''' Add newly detected tracklets to tracked_stracks'''",
        label="initial-track ID attachment",
    )
    text = _replace_once(
        text,
        "            track.activate(self.kalman_filter, self.frame_id)",
        "            track.activate(\n"
        "                self.kalman_filter,\n"
        "                self.frame_id,\n"
        '                forced_track_id=getattr(track, "forced_track_id", None),\n'
        "            )",
        label="forced track activation call",
    )
    text = _replace_once(
        text,
        "        output_stracks = [track for track in self.tracked_stracks]",
        "        output_stracks = [\n"
        "            track for track in self.tracked_stracks if track.is_activated\n"
        "        ]",
        label="confirmed-track output",
    )
    return text


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    old_count = text.count(old)
    if old_count == 1:
        return text.replace(old, new, 1)
    if old_count > 1:
        raise UpstreamPatchError(f"{label}: expected one source block, found {old_count}")
    raise UpstreamPatchError(f"{label}: expected source block was not found")


def write_summary(summary: UpstreamFixSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("botsort_root", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    summary = apply_upstream_fixes(
        args.botsort_root,
        check_only=args.check,
        create_backups=not args.no_backup,
    )
    if args.output_json:
        write_summary(summary, args.output_json)
    print(f"upstream_fix_needs_update={summary.needs_update}")
    print(f"upstream_fix_changed_file_count={summary.changed_file_count}")
    if args.check and summary.needs_update:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
