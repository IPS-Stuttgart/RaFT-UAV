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
    results: list[PatchedFile] = []
    for path, patcher in targets:
        if not path.is_file():
            raise FileNotFoundError(path)
        original = path.read_text(encoding="utf-8")
        patched = patcher(original)
        changed = patched != original
        backup_path: Path | None = None
        if changed and not check_only:
            if create_backups:
                backup_path = path.with_suffix(path.suffix + ".raft-uav-original")
                if not backup_path.exists():
                    backup_path.write_text(original, encoding="utf-8")
            temporary = path.with_suffix(path.suffix + ".raft-uav-tmp")
            temporary.write_text(patched, encoding="utf-8")
            temporary.replace(path)
        results.append(
            PatchedFile(
                path=str(path),
                changed=changed,
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
