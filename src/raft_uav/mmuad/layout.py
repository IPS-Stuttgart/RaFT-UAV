"""MMUAD/UG2+ dataset-layout inspection helpers.

The official challenge archives have appeared in multiple exported and raw
forms.  These helpers do not parse native packets; they inventory a local tree
so that the next adapter can be written from evidence rather than guesses.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any
import zipfile

from raft_uav.mmuad.io import DELIMITED_TABLE_SUFFIXES, JSON_TABLE_SUFFIXES, data_file_suffix
from raft_uav.mmuad.rosbag_bridge import load_topic_map_payload, load_topic_map_payload_text


POINT_CLOUD_SUFFIXES = {".pcd", ".ply", ".las", ".laz", ".bin"}
NUMPY_SUFFIXES = {".npy", ".npz"}
YAML_SUFFIXES = {".yaml", ".yml"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AUDIO_SUFFIXES = {".wav", ".flac", ".aac", ".mp3"}
BAG_SUFFIXES = {".bag", ".db3", ".mcap"}
ZIP_SUFFIXES = {".zip"}
TAR_SUFFIXES = {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}
ARCHIVE_SUFFIXES = ZIP_SUFFIXES | TAR_SUFFIXES
TABLE_SUFFIXES = DELIMITED_TABLE_SUFFIXES
CALIBRATION_NAMES = {
    f"{stem}{suffix}"
    for stem in (
        "calibration",
        "calib",
        "extrinsics",
        "intrinsics",
        "camera_info",
        "camera_calibration",
        "camera_intrinsics",
    )
    for suffix in (".json", ".yaml", ".yml")
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
POINT_CLOUD_TOKENS = (
    "points",
    "point_cloud",
    "cloud",
    "pcl",
    "lidar",
    "livox",
    "radar_enhance_pcl",
)
MODALITY_DIR_TOKENS = (
    "camera",
    "cam",
    "audio",
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
    "mic",
    "microphone",
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
    topic_map_has_truth_export: bool = False


def inspect_mmuad_layout(root: Path, *, max_files_per_category: int = 25) -> dict[str, Any]:
    """Return a JSON-serializable summary of a local MMUAD-style tree."""

    root = Path(root)
    base = root.parent if root.is_file() else root
    files: list[LayoutFile] = []
    archives: list[dict[str, Any]] = []
    for path in _iter_layout_files(root):
        if _is_supported_archive(path):
            archive_summary, archive_files = _inspect_archive(path, base)
            archives.append(archive_summary)
            files.extend(archive_files)
        else:
            files.append(_classify_file(path, base))
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
        "archive_count": len(archives),
        "archive_member_count": int(sum(row["member_count"] for row in archives)),
        "archives": archives,
        "sequence_candidates": sequence_candidates,
        "recommendations": _layout_recommendations(category_counts, sequence_candidates, archives),
    }
    return summary


def write_layout_report(summary: dict[str, Any], path: Path) -> Path:
    """Write a layout inspection report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _classify_file(path: Path, root: Path) -> LayoutFile:
    suffix = data_file_suffix(path)
    name = path.name.lower()
    rel = path.relative_to(root).as_posix()
    parent_text = " ".join(part.lower() for part in Path(rel).parts[:-1])
    category, topic_map_has_truth = _category_for_logical_file(
        suffix=suffix,
        name=name,
        parent_text=parent_text,
        topic_map_path=path,
    )
    return LayoutFile(
        path=path,
        relative_path=rel,
        suffix=suffix or "<none>",
        size_bytes=int(path.stat().st_size),
        category=category,
        topic_map_has_truth_export=topic_map_has_truth,
    )


def _classify_archive_member(
    archive_path: Path,
    root: Path,
    *,
    member_name: str,
    size_bytes: int,
    topic_map_text: str | None = None,
) -> LayoutFile:
    suffix = data_file_suffix(Path(member_name))
    logical = _normalize_archive_member_name(member_name)
    name = PurePosixPath(logical).name.lower()
    parent_text = " ".join(part.lower() for part in PurePosixPath(logical).parts[:-1])
    category, topic_map_has_truth = _category_for_logical_file(
        suffix=suffix,
        name=name,
        parent_text=parent_text,
        topic_map_name=f"{archive_path.name}::{logical}",
        topic_map_text=topic_map_text,
    )
    try:
        archive_rel = archive_path.relative_to(root).as_posix()
    except ValueError:
        archive_rel = archive_path.name
    return LayoutFile(
        path=archive_path,
        relative_path=f"{archive_rel}::{logical}",
        suffix=suffix or "<none>",
        size_bytes=int(size_bytes),
        category=category,
        topic_map_has_truth_export=topic_map_has_truth,
    )


def _category_for_logical_file(
    *,
    suffix: str,
    name: str,
    parent_text: str,
    topic_map_path: Path | None = None,
    topic_map_name: str | None = None,
    topic_map_text: str | None = None,
) -> tuple[str, bool]:
    topic_map_has_truth = False
    if suffix in BAG_SUFFIXES:
        category = "rosbag_or_recording"
    elif suffix in {".json", ".yaml", ".yml"} and "topic_map" in name:
        if topic_map_text is not None:
            category, topic_map_has_truth = _topic_map_category_from_text(
                topic_map_text,
                name=topic_map_name or name,
                suffix=suffix,
            )
        elif topic_map_path is not None:
            category = _topic_map_category(topic_map_path)
            topic_map_has_truth = _topic_map_has_truth_export(topic_map_path)
        else:
            category = "json_metadata"
    elif suffix in POINT_CLOUD_SUFFIXES:
        category = "point_cloud"
    elif suffix in IMAGE_SUFFIXES:
        category = "image"
    elif suffix in AUDIO_SUFFIXES:
        category = "audio"
    elif name in CALIBRATION_NAMES or "calib" in name or "extrinsic" in name:
        category = "calibration"
    elif suffix in (
        TABLE_SUFFIXES | JSON_TABLE_SUFFIXES | NUMPY_SUFFIXES | YAML_SUFFIXES
    ) and any(token in name or token in parent_text for token in CLASS_TOKENS):
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
    return category, topic_map_has_truth


def _iter_layout_files(root: Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    return [path for path in root.rglob("*") if path.is_file()]


def _sequence_candidates(files: list[LayoutFile]) -> list[dict[str, Any]]:
    grouped: dict[str, list[LayoutFile]] = defaultdict(list)
    for item in files:
        key = _sequence_key(item.relative_path)
        grouped[key].append(item)
    rows: list[dict[str, Any]] = []
    for sequence_id, members in sorted(grouped.items()):
        counts = Counter(item.category for item in members)
        topic_map_export_files = [item for item in members if item.category == "topic_map_export"]
        has_topic_map_truth = any(item.topic_map_has_truth_export for item in topic_map_export_files)
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
    logical_path = _logical_member_path(relative_path)
    parts = PurePosixPath(logical_path).parts
    if len(parts) <= 1:
        return "."

    parent_parts = parts[:-1]
    candidate_indices = [
        index
        for index, part in enumerate(parent_parts)
        if _normalized_dir_name(part) not in MODALITY_DIR_TOKENS + SPLIT_DIR_TOKENS
    ]
    if not candidate_indices:
        return "."
    return parent_parts[candidate_indices[-1]]


def _normalized_dir_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def _logical_member_path(relative_path: str) -> str:
    if "::" not in relative_path:
        return relative_path
    return relative_path.split("::", 1)[1]


def _normalize_archive_member_name(name: str) -> str:
    return str(PurePosixPath(str(name).replace("\\", "/")))


def _is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _archive_kind(path: Path) -> str:
    name = path.name.lower()
    if any(name.endswith(suffix) for suffix in ZIP_SUFFIXES):
        return "zip"
    if any(name.endswith(suffix) for suffix in TAR_SUFFIXES):
        return "tar"
    return "unknown"


def _inspect_archive(archive_path: Path, root: Path) -> tuple[dict[str, Any], list[LayoutFile]]:
    kind = _archive_kind(archive_path)
    if kind == "zip":
        files = _inspect_zip_archive(archive_path, root)
    elif kind == "tar":
        files = _inspect_tar_archive(archive_path, root)
    else:
        files = []
    try:
        archive_rel = archive_path.relative_to(root).as_posix()
    except ValueError:
        archive_rel = archive_path.name
    summary = {
        "path": archive_rel,
        "format": kind,
        "member_count": len(files),
        "total_uncompressed_size_bytes": int(sum(item.size_bytes for item in files)),
    }
    return summary, files


def _inspect_zip_archive(archive_path: Path, root: Path) -> list[LayoutFile]:
    rows: list[LayoutFile] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = _normalize_archive_member_name(info.filename)
            topic_map_text = None
            if _is_topic_map_member(name):
                with archive.open(info) as handle:
                    topic_map_text = handle.read().decode("utf-8")
            rows.append(
                _classify_archive_member(
                    archive_path,
                    root,
                    member_name=name,
                    size_bytes=int(info.file_size),
                    topic_map_text=topic_map_text,
                )
            )
    return rows


def _inspect_tar_archive(archive_path: Path, root: Path) -> list[LayoutFile]:
    rows: list[LayoutFile] = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        for info in archive.getmembers():
            if not info.isfile():
                continue
            name = _normalize_archive_member_name(info.name)
            topic_map_text = None
            if _is_topic_map_member(name):
                handle = archive.extractfile(info)
                if handle is not None:
                    with handle:
                        topic_map_text = handle.read().decode("utf-8")
            rows.append(
                _classify_archive_member(
                    archive_path,
                    root,
                    member_name=name,
                    size_bytes=int(info.size),
                    topic_map_text=topic_map_text,
                )
            )
    return rows


def _is_topic_map_member(name: str) -> bool:
    path = PurePosixPath(name)
    suffix = data_file_suffix(Path(path.name))
    return suffix in {".json", ".yaml", ".yml"} and "topic_map" in path.name.lower()


def _layout_recommendations(
    category_counts: Counter[str],
    sequence_candidates: list[dict[str, Any]],
    archives: list[dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []
    if archives:
        recommendations.append(
            "Archive files found: ZIP/TAR members were inventoried without extraction; "
            "extract or point sequence-root/native ROS tools at the selected sequence files "
            "before running tracking."
        )
    if category_counts.get("rosbag_or_recording", 0):
        recommendations.append(
            "ROS bag / recording files found: use native ROS extraction with a "
            "topic-map template, or export candidate CSV/PCD files before tracking."
        )
    if category_counts.get("point_cloud", 0):
        recommendations.append(
            "Point-cloud files found: exported CSV/JSON/NumPy/PCD/PLY/BIN files can "
            "be clustered for smoke tests; native packet formats still need explicit "
            "parsers."
        )
    if category_counts.get("image", 0):
        recommendations.append(
            "Image files found: camera detections must be generated by an external detector "
            "and exported as candidate CSV before this tracker can use them."
        )
    if category_counts.get("audio", 0):
        recommendations.append(
            "Audio files found: acoustic detections must be generated by an external "
            "preprocessor and exported as candidate tables before tracking."
        )
    if category_counts.get("topic_map_export", 0):
        recommendations.append(
            "Exported topic-map JSON/YAML files found: sequence-root mode can load "
            "their referenced CSV/TSV/TXT/JSON or NumPy exports."
        )
    if category_counts.get("topic_map_native", 0):
        recommendations.append(
            "Native-only topic-map JSON/YAML files found: use them with "
            "--rosbag-path and --topic-map-file/--topic-map-json for explicit "
            "native ROS extraction."
        )
    missing_calibration = [
        row["sequence_id"]
        for row in sequence_candidates
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
        payload = load_topic_map_payload(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return "json_metadata"
    exports = payload.get("exports", [])
    if not isinstance(exports, list):
        return "json_metadata"
    if any(isinstance(item, dict) and item.get("path") for item in exports):
        return "topic_map_export"
    if exports:
        return "topic_map_native"
    return "json_metadata"


def _topic_map_category_from_text(
    text: str,
    *,
    name: str,
    suffix: str,
) -> tuple[str, bool]:
    try:
        payload = load_topic_map_payload_text(text, name=name, suffix=suffix)
    except (json.JSONDecodeError, ValueError):
        return "json_metadata", False
    exports = payload.get("exports", [])
    if not isinstance(exports, list):
        return "json_metadata", False
    has_truth = _topic_map_payload_has_truth_export(payload)
    if any(isinstance(item, dict) and item.get("path") for item in exports):
        return "topic_map_export", has_truth
    if exports:
        return "topic_map_native", False
    return "json_metadata", False


def _topic_map_has_truth_export(path: Path) -> bool:
    try:
        payload = load_topic_map_payload(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return _topic_map_payload_has_truth_export(payload)


def _topic_map_payload_has_truth_export(payload: dict[str, Any]) -> bool:
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
