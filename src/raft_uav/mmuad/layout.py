"""MMUAD/UG2+ dataset-layout inspection helpers.

The official challenge archives have appeared in multiple exported and raw
forms.  These helpers do not parse native packets; they inventory a local tree
so that the next adapter can be written from evidence rather than guesses.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


POINT_CLOUD_SUFFIXES = {".pcd", ".ply", ".las", ".laz", ".bin"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BAG_SUFFIXES = {".bag", ".db3", ".mcap"}
TABLE_SUFFIXES = {".csv", ".txt", ".tsv"}
CALIBRATION_NAMES = {
    "calibration.json",
    "calib.json",
    "extrinsics.json",
    "intrinsics.json",
    "camera_info.json",
}
TRUTH_TOKENS = ("truth", "ground_truth", "gt", "leica", "label")
CANDIDATE_TOKENS = ("candidate", "detection", "track", "points", "point_cloud")


@dataclass(frozen=True)
class LayoutFile:
    """One file discovered during layout inspection."""

    path: Path
    relative_path: str
    suffix: str
    size_bytes: int
    category: str


def inspect_mmuad_layout(root: Path, *, max_files_per_category: int = 25) -> dict[str, Any]:
    """Return a JSON-serializable summary of a local MMUAD-style tree."""

    root = Path(root)
    files = [_classify_file(path, root) for path in root.rglob("*") if path.is_file()]
    suffix_counts = Counter(item.suffix for item in files)
    category_counts = Counter(item.category for item in files)
    examples: dict[str, list[str]] = defaultdict(list)
    for item in files:
        if len(examples[item.category]) < max_files_per_category:
            examples[item.category].append(item.relative_path)

    sequence_candidates = _sequence_candidates(files)
    summary: dict[str, Any] = {
        "schema": "raft-uav-mmuad-layout-inspection-v1",
        "root": str(root),
        "file_count": len(files),
        "total_size_bytes": int(sum(item.size_bytes for item in files)),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "examples": dict(sorted(examples.items())),
        "sequence_candidates": sequence_candidates,
        "recommendations": _layout_recommendations(category_counts, sequence_candidates),
    }
    return summary


def write_layout_report(summary: dict[str, Any], path: Path) -> Path:
    """Write a layout inspection report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _classify_file(path: Path, root: Path) -> LayoutFile:
    suffix = path.suffix.lower()
    name = path.name.lower()
    rel = path.relative_to(root).as_posix()
    if suffix in BAG_SUFFIXES:
        category = "rosbag_or_recording"
    elif suffix in POINT_CLOUD_SUFFIXES:
        category = "point_cloud"
    elif suffix in IMAGE_SUFFIXES:
        category = "image"
    elif name in CALIBRATION_NAMES or "calib" in name or "extrinsic" in name:
        category = "calibration"
    elif suffix in TABLE_SUFFIXES and any(token in name for token in TRUTH_TOKENS):
        category = "truth_or_label"
    elif suffix in TABLE_SUFFIXES and any(token in name for token in CANDIDATE_TOKENS):
        category = "candidate_or_point_table"
    elif suffix == ".json":
        category = "json_metadata"
    elif suffix in TABLE_SUFFIXES:
        category = "table_other"
    else:
        category = "other"
    return LayoutFile(
        path=path,
        relative_path=rel,
        suffix=suffix or "<none>",
        size_bytes=int(path.stat().st_size),
        category=category,
    )


def _sequence_candidates(files: list[LayoutFile]) -> list[dict[str, Any]]:
    grouped: dict[str, list[LayoutFile]] = defaultdict(list)
    for item in files:
        parts = Path(item.relative_path).parts
        key = parts[0] if len(parts) > 1 else "."
        grouped[key].append(item)
    rows: list[dict[str, Any]] = []
    for sequence_id, members in sorted(grouped.items()):
        counts = Counter(item.category for item in members)
        rows.append(
            {
                "sequence_id": sequence_id,
                "file_count": len(members),
                "categories": dict(sorted(counts.items())),
                "has_candidates_or_points": bool(
                    counts.get("candidate_or_point_table", 0)
                    or counts.get("point_cloud", 0)
                    or counts.get("rosbag_or_recording", 0)
                ),
                "has_truth_or_labels": bool(counts.get("truth_or_label", 0)),
                "has_calibration": bool(counts.get("calibration", 0)),
            }
        )
    return rows


def _layout_recommendations(
    category_counts: Counter[str],
    sequence_candidates: list[dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []
    if category_counts.get("rosbag_or_recording", 0):
        recommendations.append(
            "ROS bag / recording files found: add a native rosbag extraction adapter "
            "or export candidate CSV/PCD files before tracking."
        )
    if category_counts.get("point_cloud", 0):
        recommendations.append(
            "Point-cloud files found: the current ASCII PCD/PLY path can be used for "
            "smoke tests; binary/native formats still need explicit parsers."
        )
    if category_counts.get("image", 0):
        recommendations.append(
            "Image files found: camera detections must be generated by an external detector "
            "and exported as candidate CSV before this tracker can use them."
        )
    missing_calibration = [
        row["sequence_id"] for row in sequence_candidates
        if row["has_candidates_or_points"] and not row["has_calibration"]
    ]
    if missing_calibration:
        recommendations.append(
            "Some candidate/point sequences have no calibration file; verify coordinates "
            "are already in a shared/world frame or add calibration exports."
        )
    if not category_counts.get("truth_or_label", 0):
        recommendations.append(
            "No obvious truth/label files found; tracking can run, but metrics will be absent."
        )
    return recommendations
