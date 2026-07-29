"""Patch the upstream Multi-UAV LTS BoT-SORT checkout for competition use.

The competition baseline lives in a separate checkout. This module applies a
small, version-guarded patch at run time instead of vendoring that repository.
The patch is deliberately idempotent and fails closed when expected upstream
anchors are absent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ._upstream_patch_common import _PATCH_MARKER, UpstreamPatchError
from ._upstream_patch_inference import _patch_inference
from ._upstream_patch_mc import _patch_mc_bot_sort


_ASSOCIATION_TEMPLATE_PATH = Path(__file__).with_name(
    "_upstream_association_template.py"
)
_ASSOCIATION_HELPER = _ASSOCIATION_TEMPLATE_PATH.read_text(encoding="utf-8")


@dataclass(frozen=True)
class PatchedFile:
    path: str
    action: str
    before_sha256: str | None
    after_sha256: str


@dataclass(frozen=True)
class UpstreamPatchReport:
    botsort_root: str
    changed: bool
    dry_run: bool
    files: tuple[PatchedFile, ...]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".raft-uav.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def apply_upstream_tracker_patch(
    botsort_root: Path,
    *,
    dry_run: bool = False,
) -> UpstreamPatchReport:
    """Apply the supported competition patch to an upstream ``BoT-SORT`` root."""

    root = Path(botsort_root).expanduser().resolve()
    targets = {
        root / "tracker" / "mc_bot_sort.py": _patch_mc_bot_sort,
        root / "tools" / "inference.py": _patch_inference,
    }
    records: list[PatchedFile] = []
    pending: list[tuple[Path, str]] = []

    for path, patch_function in targets.items():
        if not path.is_file():
            raise UpstreamPatchError(f"missing supported upstream file: {path}")
        original = path.read_text(encoding="utf-8")
        patched = patch_function(original)
        action = "updated" if patched != original else "unchanged"
        records.append(
            PatchedFile(
                path=str(path.relative_to(root)),
                action=action,
                before_sha256=_sha256(original),
                after_sha256=_sha256(patched),
            )
        )
        if patched != original:
            pending.append((path, patched))

    helper_path = root / "tracker" / "raft_uav_association.py"
    helper_original = (
        helper_path.read_text(encoding="utf-8") if helper_path.exists() else None
    )
    if helper_original == _ASSOCIATION_HELPER:
        helper_action = "unchanged"
    else:
        helper_action = "created" if helper_original is None else "updated"
        pending.append((helper_path, _ASSOCIATION_HELPER))
    records.append(
        PatchedFile(
            path=str(helper_path.relative_to(root)),
            action=helper_action,
            before_sha256=(
                None if helper_original is None else _sha256(helper_original)
            ),
            after_sha256=_sha256(_ASSOCIATION_HELPER),
        )
    )

    if not dry_run:
        for path, content in pending:
            _write_text(path, content)

    return UpstreamPatchReport(
        botsort_root=str(root),
        changed=bool(pending),
        dry_run=dry_run,
        files=tuple(records),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("botsort_root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    report = apply_upstream_tracker_patch(args.botsort_root, dry_run=args.dry_run)
    payload = asdict(report)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
