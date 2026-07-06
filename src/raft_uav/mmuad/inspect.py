"""Dataset-layout inspection helpers for MMUAD/UG2+ experiments.

The official MMUAD archive layout is not assumed here.  These helpers crawl an
unpacked/exported directory, classify files by conservative filename/suffix
rules, infer timestamps from filenames when possible, and write reviewable
CSV/JSON reports.  The report is intended to guide the next native parser patch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any
import zipfile

import pandas as pd

from raft_uav.mmuad.io import (
    DELIMITED_TABLE_SUFFIXES,
    JSON_TABLE_SUFFIXES,
    data_file_suffix,
)
from raft_uav.mmuad.rosbag_bridge import load_topic_map_payload, load_topic_map_payload_text

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AUDIO_SUFFIXES = {".wav", ".flac", ".aac", ".mp3"}
POINT_SUFFIXES = {".pcd", ".ply", ".las", ".laz", ".bin"}
NUMPY_SUFFIXES = {".npy", ".npz"}
YAML_SUFFIXES = {".yaml", ".yml"}
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
TRUTH_NAMES = {
    "truth.csv",
    "truth.npy",
    "truth.npz",
    "ground_truth.csv",
    "ground_truth.npy",
    "ground_truth.npz",
    "ground_truth.json",
    "gt.csv",
    "gt.npy",
    "gt.npz",
    "gt.json",
    "labels.csv",
    "labels.json",
    "truth.json",
}
TRUTH_HINTS = ("truth", "ground_truth", "gt", "label")
CLASS_HINTS = ("class", "uav_type", "category")
CANDIDATE_HINTS = (
    "candidate",
    "detection",
    "tracklet",
    "cluster",
    "trajectory",
    "trajectories",
    "tracking",
    "result",
)
RADAR_HINTS = ("radar", "mmwave", "mmw", "ti_")
LIDAR_HINTS = ("lidar", "livox", "mid360", "avia", "point", "cloud", "pcd", "ply")
POINT_CLOUD_HINTS = ("point", "cloud", "pcd", "ply", "pcl", "radar_enhance_pcl")
CAMERA_HINTS = ("camera", "cam", "fisheye", "image", "rgb", "left", "right")
AUDIO_HINTS = ("audio", "mic", "microphone", "wav")
MODALITY_DIR_HINTS = (
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
_FILENAME_NUMBER_TOKEN_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


@dataclass(frozen=True)
class InspectedFile:
    """One classified file in an exported/raw MMUAD sequence."""

    sequence_id: str
    relative_path: str
    suffix: str
    category: str
    modality: str
    inferred_time_s: float | None
    size_bytes: int
    topic_map_has_truth_export: bool = False
    archive_path: str | None = None


def inspect_sequence_root(
    root: Path,
    *,
    sequence_glob: str = "*",
    recursive: bool = True,
) -> dict[str, Any]:
    """Inspect an MMUAD-like root and return a serializable layout report."""

    root = Path(root)
    records: list[InspectedFile] = []
    archives: list[dict[str, Any]] = []
    if root.is_file():
        sequence_dirs: list[Path] = []
        if _is_supported_archive(root):
            archive_summary, archive_records = _inspect_archive(root, root.parent)
            archives.append(archive_summary)
            records.extend(archive_records)
    else:
        sequence_dirs = _discover_sequence_dirs(root, sequence_glob=sequence_glob)
        for sequence_dir in sequence_dirs:
            records.extend(_inspect_sequence(sequence_dir, recursive=recursive))
        for archive_path in sorted(path for path in root.rglob("*") if path.is_file()):
            if not _is_supported_archive(archive_path):
                continue
            archive_summary, archive_records = _inspect_archive(archive_path, root)
            archives.append(archive_summary)
            records.extend(archive_records)
    file_rows = [record.__dict__ for record in records]
    sequence_reports = _summarize_by_sequence(file_rows)
    category_counts = Counter(row["category"] for row in file_rows)
    modality_counts = Counter(row["modality"] for row in file_rows)
    return {
        "schema": "raft-uav-mmuad-layout-report-v1",
        "root": str(root),
        "sequence_count": len(sequence_reports),
        "file_count": len(file_rows),
        "category_counts": dict(sorted(category_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "archive_count": len(archives),
        "archive_member_count": int(sum(row["member_count"] for row in archives)),
        "archives": archives,
        "sequences": sequence_reports,
        "files": file_rows,
    }


def write_layout_report(report: dict[str, Any], *, json_path: Path, csv_path: Path | None = None) -> None:
    """Write an MMUAD layout report to JSON and optional flat CSV."""

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report.get("files", [])).to_csv(csv_path, index=False)


def classify_mmuad_file(path: Path) -> tuple[str, str, float | None]:
    """Return ``(category, modality, inferred_time_s)`` for a file path."""

    return _classify_logical_file(path)


def _infer_time_s_from_filename(path: Path) -> float:
    """Infer a frame timestamp from the last numeric filename token."""

    stem = Path(path).stem
    tokens: list[str] = []
    for match in _FILENAME_NUMBER_TOKEN_RE.finditer(stem):
        token = match.group(0)
        if token[0] in "+-" and match.start() > 0 and stem[match.start() - 1].isalnum():
            token = token[1:]
        tokens.append(token)
    if not tokens:
        return 0.0
    return float(tokens[-1])


def _classify_logical_file(
    path: Path,
    *,
    topic_map_text: str | None = None,
    topic_map_name: str | None = None,
) -> tuple[str, str, float | None]:
    suffix = data_file_suffix(path)
    name = path.name.lower()
    stem = path.stem.lower()
    parent = path.parent.name.lower()
    modality = _infer_modality(" ".join((stem, parent)))
    inferred_time_s = None
    if (
        suffix in IMAGE_SUFFIXES | AUDIO_SUFFIXES | POINT_SUFFIXES | NUMPY_SUFFIXES
        or modality in {"radar", "lidar", "camera", "audio"}
    ):
        inferred_time_s = _infer_time_s_from_filename(path)
    if name in CALIBRATION_NAMES:
        return "calibration", modality, None
    if suffix in {".json", ".yaml", ".yml"} and "topic_map" in name:
        if topic_map_text is not None:
            return (
                _topic_map_category_from_text(
                    topic_map_text,
                    name=topic_map_name or path.as_posix(),
                    suffix=suffix,
                ),
                "ros",
                None,
            )
        return _topic_map_category(path), "ros", None
    if suffix in (
        NUMPY_SUFFIXES | TABLE_SUFFIXES | JSON_TABLE_SUFFIXES | YAML_SUFFIXES
    ) and any(hint in stem or hint in parent for hint in CLASS_HINTS):
        return "class_label", modality, inferred_time_s
    if name in TRUTH_NAMES or any(hint in stem or hint in parent for hint in TRUTH_HINTS):
        return "truth", modality, None
    if suffix in NUMPY_SUFFIXES:
        if any(hint in stem or hint in parent for hint in LIDAR_HINTS + POINT_CLOUD_HINTS):
            return "point_cloud", modality if modality != "unknown" else "lidar", inferred_time_s
        if any(hint in stem or hint in parent for hint in CANDIDATE_HINTS):
            return "candidate", modality, inferred_time_s
        return "numpy", modality, inferred_time_s
    if suffix in IMAGE_SUFFIXES:
        return "image", "camera", inferred_time_s
    if suffix in AUDIO_SUFFIXES:
        return "audio", "audio", inferred_time_s
    if suffix in POINT_SUFFIXES:
        return "point_cloud", modality if modality != "unknown" else "lidar", inferred_time_s
    if suffix in TABLE_SUFFIXES | JSON_TABLE_SUFFIXES:
        if "point" in stem or "cloud" in stem or modality == "lidar":
            return "point_cloud_csv", "lidar", None
        if any(hint in stem or hint in parent for hint in CANDIDATE_HINTS):
            return "candidate", modality, None
        if suffix != ".json" and modality == "radar":
            return "radar_csv", "radar", None
        return "csv" if suffix == ".csv" else "metadata", modality, None
    if suffix in JSON_TABLE_SUFFIXES | YAML_SUFFIXES | {".toml", ".txt"}:
        return "metadata", modality, None
    if suffix in {".bag", ".db3", ".mcap"}:
        return "ros_recording", modality, None
    return "other", modality, inferred_time_s


def _discover_sequence_dirs(root: Path, *, sequence_glob: str) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return []
    children = [path for path in sorted(root.glob(sequence_glob)) if path.is_dir()]
    if not children:
        return [root]
    direct_sequences = [
        child
        for child in children
        if not _is_modality_dir(child) and _directory_has_sequence_data(child)
    ]
    if direct_sequences:
        return direct_sequences
    nested_sequences: list[Path] = []
    for child in children:
        if _is_modality_dir(child):
            continue
        nested_sequences.extend(
            grandchild
            for grandchild in sorted(child.glob(sequence_glob))
            if grandchild.is_dir()
            and not _is_modality_dir(grandchild)
            and _directory_has_sequence_data(grandchild)
        )
    if nested_sequences:
        return nested_sequences
    if _directory_has_sequence_data(root):
        return [root]
    return children


def _directory_has_sequence_data(path: Path) -> bool:
    for item in path.iterdir():
        if item.is_file():
            category, _modality, _time = classify_mmuad_file(item)
            if category != "other":
                return True
        elif item.is_dir() and _is_modality_dir(item):
            for nested in item.rglob("*"):
                if not nested.is_file():
                    continue
                category, _modality, _time = classify_mmuad_file(nested)
                if category != "other":
                    return True
    return False


def _is_modality_dir(path: Path) -> bool:
    normalized = path.name.lower().replace("-", "_").replace(" ", "_")
    return normalized in MODALITY_DIR_HINTS


def _inspect_sequence(sequence_dir: Path, *, recursive: bool) -> list[InspectedFile]:
    iterator = sequence_dir.rglob("*") if recursive else sequence_dir.glob("*")
    records: list[InspectedFile] = []
    for path in sorted(iterator):
        if not path.is_file():
            continue
        if _is_supported_archive(path):
            continue
        category, modality, time_s = classify_mmuad_file(path)
        records.append(
            InspectedFile(
                sequence_id=sequence_dir.name,
                relative_path=path.relative_to(sequence_dir).as_posix(),
                suffix=path.suffix.lower(),
                category=category,
                modality=modality,
                inferred_time_s=time_s,
                size_bytes=path.stat().st_size,
                topic_map_has_truth_export=(
                    category == "topic_map_export"
                    and _topic_map_has_truth_export(path)
                ),
            )
        )
    return records


def _summarize_by_sequence(file_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in file_rows:
        by_sequence.setdefault(str(row["sequence_id"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for sequence_id, rows in sorted(by_sequence.items()):
        categories = Counter(row["category"] for row in rows)
        modalities = Counter(row["modality"] for row in rows)
        time_values = [row["inferred_time_s"] for row in rows if row.get("inferred_time_s") is not None]
        topic_map_truth = any(
            row["category"] == "topic_map_export"
            and bool(row.get("topic_map_has_truth_export"))
            for row in rows
        )
        missing: list[str] = []
        if categories.get("truth", 0) == 0 and not topic_map_truth:
            missing.append("truth")
        if categories.get("calibration", 0) == 0:
            missing.append("calibration")
        if not any(
            categories.get(name, 0)
            for name in (
                "candidate",
                "point_cloud",
                "point_cloud_csv",
                "radar_csv",
                "topic_map_export",
            )
        ):
            missing.append("candidate_or_point_cloud")
        summaries.append(
            {
                "sequence_id": sequence_id,
                "file_count": len(rows),
                "category_counts": dict(sorted(categories.items())),
                "modality_counts": dict(sorted(modalities.items())),
                "missing_for_tracking_smoke": missing,
                "time_min_s": min(time_values) if time_values else None,
                "time_max_s": max(time_values) if time_values else None,
                "time_count": len(time_values),
            }
        )
    return summaries


def _infer_modality(stem: str) -> str:
    if any(hint in stem for hint in RADAR_HINTS):
        return "radar"
    if any(hint in stem for hint in LIDAR_HINTS):
        return "lidar"
    if any(hint in stem for hint in CAMERA_HINTS):
        return "camera"
    if any(hint in stem for hint in AUDIO_HINTS):
        return "audio"
    return "unknown"


def _topic_map_category(path: Path) -> str:
    try:
        payload = load_topic_map_payload(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return "metadata"
    exports = payload.get("exports", [])
    if not isinstance(exports, list):
        return "metadata"
    if any(isinstance(item, dict) and item.get("path") for item in exports):
        return "topic_map_export"
    if exports:
        return "topic_map_native"
    return "metadata"


def _topic_map_category_from_text(text: str, *, name: str, suffix: str) -> str:
    try:
        payload = load_topic_map_payload_text(text, name=name, suffix=suffix)
    except (json.JSONDecodeError, ValueError):
        return "metadata"
    exports = payload.get("exports", [])
    if not isinstance(exports, list):
        return "metadata"
    if any(isinstance(item, dict) and item.get("path") for item in exports):
        return "topic_map_export"
    if exports:
        return "topic_map_native"
    return "metadata"


def _topic_map_has_truth_export(path: Path) -> bool:
    try:
        payload = load_topic_map_payload(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return _topic_map_payload_has_truth_export(payload)


def _topic_map_text_has_truth_export(text: str, *, name: str, suffix: str) -> bool:
    try:
        payload = load_topic_map_payload_text(text, name=name, suffix=suffix)
    except (json.JSONDecodeError, ValueError):
        return False
    return _topic_map_payload_has_truth_export(payload)


def _topic_map_payload_has_truth_export(payload: dict[str, Any]) -> bool:
    for export in payload.get("exports", []):
        if not isinstance(export, dict):
            continue
        kind = str(export.get("kind", "")).lower()
        export_path = str(export.get("path", "")).lower()
        if kind == "truth" or kind.endswith("_truth") or any(
            token in export_path for token in TRUTH_HINTS
        ):
            return True
    return False


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


def _inspect_archive(archive_path: Path, root: Path) -> tuple[dict[str, Any], list[InspectedFile]]:
    kind = _archive_kind(archive_path)
    if kind == "zip":
        rows = _inspect_zip_archive(archive_path, root)
    elif kind == "tar":
        rows = _inspect_tar_archive(archive_path, root)
    else:
        rows = []
    try:
        archive_rel = archive_path.relative_to(root).as_posix()
    except ValueError:
        archive_rel = archive_path.name
    summary = {
        "path": archive_rel,
        "format": kind,
        "member_count": len(rows),
        "total_uncompressed_size_bytes": int(sum(row.size_bytes for row in rows)),
    }
    return summary, rows


def _inspect_zip_archive(archive_path: Path, root: Path) -> list[InspectedFile]:
    rows: list[InspectedFile] = []
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
                _archive_member_record(
                    archive_path,
                    root,
                    member_name=name,
                    size_bytes=int(info.file_size),
                    topic_map_text=topic_map_text,
                )
            )
    return rows


def _inspect_tar_archive(archive_path: Path, root: Path) -> list[InspectedFile]:
    rows: list[InspectedFile] = []
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
                _archive_member_record(
                    archive_path,
                    root,
                    member_name=name,
                    size_bytes=int(info.size),
                    topic_map_text=topic_map_text,
                )
            )
    return rows


def _archive_member_record(
    archive_path: Path,
    root: Path,
    *,
    member_name: str,
    size_bytes: int,
    topic_map_text: str | None,
) -> InspectedFile:
    logical = _normalize_archive_member_name(member_name)
    sequence_id = _sequence_id_from_logical_path(logical, default=_archive_sequence_id(archive_path))
    category, modality, time_s = _classify_logical_file(
        Path(logical),
        topic_map_text=topic_map_text,
        topic_map_name=f"{archive_path.name}::{logical}",
    )
    try:
        archive_rel = archive_path.relative_to(root).as_posix()
    except ValueError:
        archive_rel = archive_path.name
    topic_map_has_truth = False
    if category == "topic_map_export" and topic_map_text is not None:
        topic_map_has_truth = _topic_map_text_has_truth_export(
            topic_map_text,
            name=f"{archive_path.name}::{logical}",
            suffix=data_file_suffix(Path(logical)),
        )
    return InspectedFile(
        sequence_id=sequence_id,
        relative_path=f"{archive_rel}::{logical}",
        suffix=data_file_suffix(Path(logical)) or "<none>",
        category=category,
        modality=modality,
        inferred_time_s=time_s,
        size_bytes=int(size_bytes),
        topic_map_has_truth_export=topic_map_has_truth,
        archive_path=archive_rel,
    )


def _sequence_id_from_logical_path(logical_path: str, *, default: str) -> str:
    parts = PurePosixPath(logical_path).parts
    if len(parts) <= 1:
        return default
    parent_parts = parts[:-1]
    candidate_indices = [
        index
        for index, part in enumerate(parent_parts)
        if _normalized_dir_name(part) not in MODALITY_DIR_HINTS
    ]
    if not candidate_indices:
        return default
    return parent_parts[candidate_indices[-1]]


def _normalized_dir_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def _archive_sequence_id(path: Path) -> str:
    name = path.name
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _normalize_archive_member_name(name: str) -> str:
    return str(PurePosixPath(str(name).replace("\\", "/")))


def _is_topic_map_member(name: str) -> bool:
    path = PurePosixPath(name)
    suffix = data_file_suffix(Path(path.name))
    return suffix in {".json", ".yaml", ".yml"} and "topic_map" in path.name.lower()
