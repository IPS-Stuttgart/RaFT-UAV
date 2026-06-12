"""ROS-bag bridge helpers for MMUAD exported data.

The helpers avoid depending on ROS at import time.  They can inspect ROS2
``metadata.yaml`` directories, optionally call ``rosbag info --yaml`` for ROS1
bags when the command exists, and load normalized topic exports via a topic-map
JSON.  This is a bridge toward native support; it is not a binary message
parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import re
import shutil
import subprocess

import pandas as pd

from raft_uav.mmuad.io import (
    load_candidate_file,
    load_point_cloud_file_as_candidates,
    load_truth_file,
    merge_candidate_frames,
    point_rows_to_candidates,
)
from raft_uav.mmuad.schema import (
    CandidateFrame,
    TruthFrame,
    normalize_candidate_columns,
    normalize_truth_columns,
)


@dataclass(frozen=True)
class TopicExportBundle:
    """Normalized candidates/truth loaded from topic-map exports."""

    candidates: CandidateFrame
    truth: TruthFrame | None
    manifest: dict[str, Any]


def inspect_rosbag(path: Path) -> dict[str, Any]:
    """Inspect a ROS bag path without requiring ROS Python packages."""

    path = Path(path)
    if path.is_dir():
        metadata = path / "metadata.yaml"
        if metadata.exists():
            return _inspect_ros2_metadata(metadata)
        return {
            "path": str(path),
            "kind": "directory",
            "metadata_yaml": False,
            "files": [
                str(item.relative_to(path))
                for item in sorted(path.rglob("*"))
                if item.is_file()
            ][:200],
            "recommendation": "No metadata.yaml found; export topics to CSV and use --topic-map-json.",
        }
    if path.suffix.lower() == ".bag":
        return _inspect_ros1_bag(path)
    return {
        "path": str(path),
        "kind": "unknown",
        "suffix": path.suffix.lower(),
        "recommendation": "Unsupported bag path. Use layout inspection or exported CSV topic maps.",
    }


def write_topic_map_template(report: dict[str, Any], path: Path) -> Path:
    """Write a topic-map JSON template from an inspection report."""

    topics = report.get("topics", [])
    exports = []
    for idx, topic in enumerate(topics):
        name = str(topic.get("name", topic.get("topic", f"topic_{idx}")))
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip("/")).strip("_") or f"topic_{idx}"
        kind = _infer_topic_map_kind(topic)
        exports.append(
            {
                "topic": name,
                "kind": kind,
                "path": f"exports/{safe}.csv",
                "source": safe if not _is_truth_kind(kind) else None,
                "sequence_id": report.get(
                    "sequence_id",
                    Path(str(report.get("path", "sequence"))).stem,
                ),
                "column_aliases": {
                    "stamp": "time_s",
                    "timestamp": "time_s",
                    "x": "x_m",
                    "y": "y_m",
                    "z": "z_m",
                },
            }
        )
    payload = {
        "schema": "raft-uav-mmuad-topic-map-v1",
        "sequence_id": report.get(
            "sequence_id",
            Path(str(report.get("path", "sequence"))).stem,
        ),
        "description": "Edit paths and aliases to point at CSV exports of ROS topics.",
        "exports": exports,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_topic_map_exports(path: Path, *, base_dir: Path | None = None) -> TopicExportBundle:
    """Load normalized candidates/truth from a topic-map JSON."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = Path(base_dir) if base_dir is not None else path.parent
    default_sequence_id = str(payload.get("sequence_id", base.name))
    candidate_frames: list[CandidateFrame] = []
    truth_frames: list[TruthFrame] = []
    loaded: list[dict[str, Any]] = []
    for spec in payload.get("exports", []):
        export_path = base / str(spec["path"])
        if not export_path.exists():
            loaded.append({"path": str(export_path), "status": "missing"})
            continue
        kind = str(spec.get("kind", "candidate"))
        sequence_id = str(spec.get("sequence_id", default_sequence_id))
        if _is_truth_kind(kind):
            truth_frame = _load_topic_truth_export(export_path, spec, sequence_id=sequence_id)
            truth_frames.append(truth_frame)
            row_count = len(truth_frame.rows)
        else:
            candidate_frame = _load_topic_candidate_export(export_path, spec, sequence_id=sequence_id)
            candidate_frames.append(candidate_frame)
            row_count = len(candidate_frame.rows)
        loaded.append(
            {
                "path": str(export_path),
                "kind": kind,
                "status": "loaded",
                "rows": int(row_count),
            }
        )
    if not candidate_frames:
        raise ValueError(f"topic map {path} did not load any candidate exports")
    candidates = merge_candidate_frames(candidate_frames)
    truth = None
    if truth_frames:
        truth_rows = pd.concat([frame.rows for frame in truth_frames], ignore_index=True)
        truth = TruthFrame(normalize_truth_columns(truth_rows))
    return TopicExportBundle(candidates, truth, {"topic_map": str(path), "loaded_exports": loaded})


def _infer_topic_map_kind(topic: dict[str, Any]) -> str:
    name = str(topic.get("name", topic.get("topic", ""))).lower()
    msg_type = str(topic.get("type", topic.get("msgtype", ""))).lower()
    truth_like = any(token in name for token in ("truth", "ground", "gt", "label", "mocap"))
    if "pointcloud2" in msg_type:
        return "pointcloud2_candidate"
    if msg_type.endswith("posestamped") or "pose_stamped" in msg_type:
        return "pose_truth" if truth_like else "pose_candidate"
    if msg_type.endswith("odometry"):
        return "odometry_truth" if truth_like else "odometry_candidate"
    return "truth" if truth_like else "candidate"


def _is_truth_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized == "truth" or normalized.endswith("_truth")


def _load_topic_truth_export(path: Path, spec: dict[str, Any], *, sequence_id: str) -> TruthFrame:
    if _is_table_export(path):
        frame = _read_topic_table(path)
        frame = _apply_aliases(frame, spec)
        if "sequence_id" not in frame.columns:
            frame["sequence_id"] = sequence_id
        return TruthFrame(normalize_truth_columns(frame))
    return load_truth_file(path, default_sequence_id=sequence_id)


def _load_topic_candidate_export(
    path: Path,
    spec: dict[str, Any],
    *,
    sequence_id: str,
) -> CandidateFrame:
    source = str(spec.get("source") or spec.get("topic") or "candidate")
    kind = str(spec.get("kind", "candidate")).strip().lower()
    if kind == "pointcloud2_candidate":
        return _load_topic_pointcloud_export(
            path,
            spec,
            sequence_id=sequence_id,
            source=source,
        )
    if _is_table_export(path):
        frame = _read_topic_table(path)
        frame = _apply_aliases(frame, spec)
        if "sequence_id" not in frame.columns:
            frame["sequence_id"] = sequence_id
        if "source" not in frame.columns:
            frame["source"] = source
        for column in ("track_id", "std_xy_m", "std_z_m", "confidence", "class_name"):
            if column not in frame.columns and spec.get(column) is not None:
                frame[column] = spec.get(column)
        return CandidateFrame(normalize_candidate_columns(frame))
    frame = load_candidate_file(path, default_sequence_id=sequence_id, source=source)
    rows = frame.rows.copy()
    for column in ("track_id", "std_xy_m", "std_z_m", "confidence", "class_name"):
        if spec.get(column) is not None:
            rows[column] = spec.get(column)
    return CandidateFrame(
        normalize_candidate_columns(rows, default_sequence_id=sequence_id)
    )


def _load_topic_pointcloud_export(
    path: Path,
    spec: dict[str, Any],
    *,
    sequence_id: str,
    source: str,
) -> CandidateFrame:
    voxel_size_m = float(spec.get("voxel_size_m", spec.get("voxel_size", 0.75)))
    min_points = int(spec.get("min_cluster_points", spec.get("min_points", 3)))
    if _is_table_export(path):
        frame = _apply_aliases(_read_topic_table(path), spec)
        if "sequence_id" not in frame.columns:
            frame["sequence_id"] = sequence_id
        if "time_s" not in frame.columns and spec.get("time_s") is not None:
            frame["time_s"] = spec["time_s"]
        return point_rows_to_candidates(
            frame,
            source=source,
            voxel_size_m=voxel_size_m,
            min_points=min_points,
        )
    return load_point_cloud_file_as_candidates(
        path,
        source=source,
        sequence_id=sequence_id,
        voxel_size_m=voxel_size_m,
        min_points=min_points,
    )


def _read_topic_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _apply_aliases(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    aliases = spec.get("column_aliases", {}) or {}
    return frame.rename(columns={str(key): str(value) for key, value in aliases.items()})


def _is_table_export(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".tsv", ".txt"}


def _inspect_ros2_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    topics: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("topic_metadata:") or stripped.startswith("- topic_metadata:"):
            if current:
                topics.append(current)
            current = {}
            continue
        if current is not None and stripped.startswith("name:"):
            current["name"] = stripped.split(":", 1)[1].strip().strip("'\"")
        elif current is not None and stripped.startswith("type:"):
            current["type"] = stripped.split(":", 1)[1].strip().strip("'\"")
        elif current is not None and stripped.startswith("message_count:"):
            try:
                current["message_count"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                current["message_count"] = stripped.split(":", 1)[1].strip()
    if current:
        topics.append(current)
    root = path.parent
    db_files = sorted(str(item.relative_to(root)) for item in root.rglob("*.db3"))
    mcap_files = sorted(str(item.relative_to(root)) for item in root.rglob("*.mcap"))
    return {
        "path": str(root),
        "kind": "ros2_bag_directory",
        "metadata_yaml": str(path),
        "topics": topics,
        "db3_files": db_files,
        "mcap_files": mcap_files,
        "recommendation": "Export relevant topics to CSV, then run with --topic-map-json.",
    }


def _inspect_ros1_bag(path: Path) -> dict[str, Any]:
    if shutil.which("rosbag") is None:
        return {
            "path": str(path),
            "kind": "ros1_bag",
            "rosbag_cli_available": False,
            "topics": [],
            "recommendation": "Install ROS/rosbag or export topics to CSV and use --topic-map-json.",
        }
    completed = subprocess.run(
        ["rosbag", "info", "--yaml", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    topics: list[dict[str, Any]] = []
    for match in re.finditer(r"topic:\s*([^\n]+).*?type:\s*([^\n]+).*?messages:\s*(\d+)", completed.stdout, re.S):
        topics.append(
            {
                "name": match.group(1).strip(),
                "type": match.group(2).strip(),
                "message_count": int(match.group(3)),
            }
        )
    return {
        "path": str(path),
        "kind": "ros1_bag",
        "rosbag_cli_available": True,
        "returncode": completed.returncode,
        "topics": topics,
        "raw_yaml": completed.stdout if completed.returncode == 0 else completed.stderr,
        "recommendation": "Export relevant topics to CSV, then run with --topic-map-json.",
    }
