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

from raft_uav.mmuad.io import JSON_TABLE_SUFFIXES


POINT_CLOUD_SUFFIXES = {".pcd", ".ply", ".las", ".laz", ".bin"}
NUMPY_SUFFIXES = {".npy", ".npz"}
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
CLASS_TOKENS = ("class", "uav_type", "category")
CANDIDATE_TOKENS = (
    "candidate",
    "detection",
    "track",
    "trajectory",
    "trajectories",
    "tracking",
    "result",
)
POINT_CLOUD_TOKENS = ("points", "point_cloud", "cloud", "lidar", "livox")
MODALITY_DIR_TOKENS = (
    "camera",
    "cam",
    "class",
    "classes",
    "detections",
    "ground_truth",
    "gt",
    "image",
    "images",
    "label",
    "labels",
    "lidar",
    "livox",
    "livox_avia",
    "point_cloud",
    "points",
    "radar",
    "tracking_results",
    "tracks",
    "trajectory",
    "truth",
    "uav_type",
)
SPLIT_DIR_TOKENS = ("dev", "test", "train", "training", "val", "valid", "validation")


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
    parent_text = " ".join(part.lower() for part in Path(rel).parts[:-1])
    if suffix in BAG_SUFFIXES:
        category = "rosbag_or_recording"
    elif suffix == ".json" and "topic_map" in name:
        category = _topic_map_category(path)
    elif suffix in POINT_CLOUD_SUFFIXES:
        category = "point_cloud"
    elif suffix in IMAGE_SUFFIXES:
        category = "image"
    elif name in CALIBRATION_NAMES or "calib" in name or "extrinsic" in name:
        category = "calibration"
    elif suffix in TABLE_SUFFIXES | JSON_TABLE_SUFFIXES | NUMPY_SUFFIXES and any(
        token in name or token in parent_text for token in CLASS_TOKENS
    ):
        category = "class_or_label"
    elif suffix in TABLE_SUFFIXES | JSON_TABLE_SUFFIXES | NUMPY_SUFFIXES and any(
        token in name or token in parent_text for token in TRUTH_TOKENS
    ):
        category = "truth_or_label"
    elif suffix in TABLE_SUFFIXES | JSON_TABLE_SUFFIXES | NUMPY_SUFFIXES and any(
        token in name or token in parent_text for token in POINT_CLOUD_TOKENS
    ):
        category = "candidate_or_point_table"
    elif suffix in TABLE_SUFFIXES | JSON_TABLE_SUFFIXES | NUMPY_SUFFIXES and any(
        token in name or token in parent_text for token in CANDIDATE_TOKENS
    ):
        category = "candidate_or_point_table"
    elif suffix in JSON_TABLE_SUFFIXES:
        category = "json_metadata"
    elif suffix in TABLE_SUFFIXES:
        category = "table_other"
    elif suffix in NUMPY_SUFFIXES:
        category = "numpy_other"
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
        key = _sequence_key(item.relative_path)
        grouped[key].append(item)
    rows: list[dict[str, Any]] = []
    for sequence_id, members in sorted(grouped.items()):
        counts = Counter(item.category for item in members)
        topic_map_export_files = [
            item for item in members if item.category == "topic_map_export"
        ]
        has_topic_map_truth = any(
            _topic_map_has_truth_export(item.path) for item in topic_map_export_files
        )
        rows.append(
            {
                "sequence_id": sequence_id,
                "file_count": len(members),
                "categories": dict(sorted(counts.items())),
                "has_topic_map_export": bool(topic_map_export_files),
                "has_native_topic_map": bool(counts.get("topic_map_native", 0)),
                "has_candidates_or_points": bool(
                    counts.get("candidate_or_point_table", 0)
                    or counts.get("point_cloud", 0)
                    or counts.get("rosbag_or_recording", 0)
                    or counts.get("topic_map_export", 0)
                ),
                "has_truth_or_labels": bool(counts.get("truth_or_label", 0) or has_topic_map_truth),
                "has_class_labels": bool(counts.get("class_or_label", 0)),
                "has_calibration": bool(counts.get("calibration", 0)),
            }
        )
    return rows


def _sequence_key(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) <= 1:
        return "."
    first = _normalized_dir_name(parts[0])
    if first in MODALITY_DIR_TOKENS:
        return "."
    if first in SPLIT_DIR_TOKENS and len(parts) > 2:
        second = _normalized_dir_name(parts[1])
        if second not in MODALITY_DIR_TOKENS:
            return parts[1]
    return parts[0]


def _normalized_dir_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


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
    if category_counts.get("topic_map_export", 0):
        recommendations.append(
            "Exported topic-map JSON files found: sequence-root mode can load "
            "their referenced CSV/TSV/TXT/JSON or NumPy exports."
        )
    if category_counts.get("topic_map_native", 0):
        recommendations.append(
            "Native-only topic-map JSON files found: use them with "
            "--rosbag-path and --topic-map-json for explicit native ROS extraction."
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


def _topic_map_category(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "json_metadata"
    exports = payload.get("exports", [])
    if not isinstance(exports, list):
        return "json_metadata"
    if any(isinstance(item, dict) and item.get("path") for item in exports):
        return "topic_map_export"
    if exports:
        return "topic_map_native"
    return "json_metadata"


def _topic_map_has_truth_export(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for export in payload.get("exports", []):
        if not isinstance(export, dict):
            continue
        kind = str(export.get("kind", "")).lower()
        export_path = str(export.get("path", "")).lower()
        if kind == "truth" or kind.endswith("_truth") or any(
            token in export_path for token in TRUTH_TOKENS
        ):
            return True
    return False
