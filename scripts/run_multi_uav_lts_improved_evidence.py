#!/usr/bin/env python3
"""Run the public evidence gate with guarded association-improvement candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import run_multi_uav_lts_public_evidence as evidence

IMPROVED_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "graph_common_motion",
        (
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
        ),
    ),
    (
        "graph_interpolate_one",
        ("--interpolate-max-gap", "1"),
    ),
    (
        "graph_common_motion_interpolate",
        (
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
            "--interpolate-max-gap",
            "1",
        ),
    ),
    (
        "graph_guarded_border_birth",
        (
            "--birth-require-border-entry",
            "--birth-min-inward-motion",
            "0.0",
        ),
    ),
    (
        "graph_common_motion_guarded_birth",
        (
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
            "--birth-require-border-entry",
            "--birth-min-inward-motion",
            "0.0",
        ),
    ),
)


def _proposal_source_cache_key(
    payload: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_sequences: int,
    expected_frames: int,
    device: str,
) -> str:
    """Hash detector-generation inputs without invalidating on graph-only edits."""
    settings = {
        "schema": "raft-uav-multi-uav-lts-proposal-source-key-v3",
        "upstream_revision": payload["upstream"]["revision"],
        "archive_md5": payload["dataset"]["archive_md5"],
        "detector_sha256": validated["detector_sha256"],
        "reid_model_sha256": validated["reid_model_sha256"],
        "cython_bbox_version": "0.1.5",
        "faiss_cpu_version": "1.8.0.post1",
        "expected_sequences": expected_sequences,
        "expected_frames": expected_frames,
        "device": device,
        "img_size": 1920,
        "proposal_conf_thres": 0.001,
        "proposal_iou_thres": 0.95,
        "device_independent": False,
    }
    source_root = repo_root / "src" / "raft_uav"
    package_root = source_root / "multi_uav_lts"
    source_paths = [
        repo_root / "scripts" / "run_multi_uav_lts_official_baseline.py",
        source_root / "__init__.py",
        *sorted((source_root / "numeric").rglob("*.py")),
        package_root / "__init__.py",
        package_root / "_records.py",
        package_root / "cli.py",
        package_root / "improved_baseline.py",
        package_root / "proposal_baseline.py",
        package_root / "proposal_export.py",
        package_root / "upstream_fixes.py",
        *sorted((package_root / "_records").rglob("*.py")),
        *sorted((package_root / "cli").rglob("*.py")),
    ]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"proposal cache source missing: {rendered}")

    hasher = hashlib.sha256()
    hasher.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for path in source_paths:
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def main() -> int:
    evidence.CANDIDATES = (*evidence.CANDIDATES, *IMPROVED_CANDIDATES)
    evidence._baseline_cache_key = _proposal_source_cache_key
    return evidence.main()


if __name__ == "__main__":
    raise SystemExit(main())
