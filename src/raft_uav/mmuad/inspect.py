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
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad.io import infer_time_s_from_filename

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
POINT_SUFFIXES = {".pcd", ".ply", ".las", ".laz"}
NUMPY_SUFFIXES = {".npy", ".npz"}
TABLE_SUFFIXES = {".csv", ".tsv", ".txt"}
JSON_TABLE_SUFFIXES = {".json"}
CALIBRATION_NAMES = {
    "calibration.json",
    "calib.json",
    "extrinsics.json",
    "calibration.yaml",
    "calib.yaml",
    "extrinsics.yaml",
    "calibration.yml",
    "calib.yml",
    "extrinsics.yml",
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
CAMERA_HINTS = ("camera", "cam", "fisheye", "image", "rgb", "left", "right")
AUDIO_HINTS = ("audio", "mic", "microphone", "wav")
MODALITY_DIR_HINTS = (
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


def inspect_sequence_root(
    root: Path,
    *,
    sequence_glob: str = "*",
    recursive: bool = True,
) -> dict[str, Any]:
    """Inspect an MMUAD-like root and return a serializable layout report."""

    root = Path(root)
    sequence_dirs = _discover_sequence_dirs(root, sequence_glob=sequence_glob)
    records: list[InspectedFile] = []
    for sequence_dir in sequence_dirs:
        records.extend(_inspect_sequence(sequence_dir, recursive=recursive))
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

    suffix = path.suffix.lower()
    name = path.name.lower()
    stem = path.stem.lower()
    parent = path.parent.name.lower()
    modality = _infer_modality(" ".join((stem, parent)))
    inferred_time_s = None
    if (
        suffix in IMAGE_SUFFIXES | POINT_SUFFIXES | NUMPY_SUFFIXES
        or modality in {"radar", "lidar", "camera"}
    ):
        inferred_time_s = infer_time_s_from_filename(path)
    if name in CALIBRATION_NAMES:
        return "calibration", modality, None
    if suffix == ".json" and "topic_map" in name:
        return _topic_map_category(path), "ros", None
    if suffix in NUMPY_SUFFIXES | TABLE_SUFFIXES | JSON_TABLE_SUFFIXES and any(
        hint in stem or hint in parent for hint in CLASS_HINTS
    ):
        return "class_label", modality, inferred_time_s
    if name in TRUTH_NAMES or any(hint in stem or hint in parent for hint in TRUTH_HINTS):
        return "truth", modality, None
    if suffix in NUMPY_SUFFIXES:
        if any(hint in stem or hint in parent for hint in LIDAR_HINTS):
            return "point_cloud", modality if modality != "unknown" else "lidar", inferred_time_s
        if any(hint in stem or hint in parent for hint in CANDIDATE_HINTS):
            return "candidate", modality, inferred_time_s
        return "numpy", modality, inferred_time_s
    if suffix in IMAGE_SUFFIXES:
        return "image", "camera", inferred_time_s
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
    if suffix in {".json", ".yaml", ".yml", ".toml", ".txt"}:
        return "metadata", modality, None
    if suffix in {".bag", ".db3", ".mcap"}:
        return "ros_recording", modality, None
    return "other", modality, inferred_time_s


def _discover_sequence_dirs(root: Path, *, sequence_glob: str) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
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
