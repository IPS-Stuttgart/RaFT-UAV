"""ROS-bag bridge helpers for MMUAD exported data.

The helpers avoid depending on ROS at import time.  They can inspect ROS2
``metadata.yaml`` directories, optionally call ``rosbag info --yaml`` for ROS1
bags when the command exists, and load normalized CSV exports via a topic-map
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

from raft_uav.mmuad.io import merge_candidate_frames
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame, normalize_candidate_columns, normalize_truth_columns


@dataclass(frozen=True)
class TopicExportBundle:
    """Normalized candidates/truth loaded from topic-map CSV exports."""

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
            "files": [str(item.relative_to(path)) for item in sorted(path.rglob("*")) if item.is_file()][:200],
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
        kind = "truth" if any(token in name.lower() for token in ("truth", "gt", "ground")) else "candidate"
        exports.append(
            {
                "topic": name,
                "kind": kind,
                "path": f"exports/{safe}.csv",
                "source": safe if kind == "candidate" else None,
                "sequence_id": report.get("sequence_id", Path(str(report.get("path", "sequence"))).stem),
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
        "sequence_id": report.get("sequence_id", Path(str(report.get("path", "sequence"))).stem),
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
        csv_path = base / str(spec["path"])
        if not csv_path.exists():
            loaded.append({"path": str(csv_path), "status": "missing"})
            continue
        kind = str(spec.get("kind", "candidate"))
        frame = pd.read_csv(csv_path)
        aliases = spec.get("column_aliases", {}) or {}
        frame = frame.rename(columns={str(k): str(v) for k, v in aliases.items()})
        sequence_id = str(spec.get("sequence_id", default_sequence_id))
        if "sequence_id" not in frame.columns:
            frame["sequence_id"] = sequence_id
        if kind == "truth":
            normalized = normalize_truth_columns(frame)
            truth_frames.append(TruthFrame(normalized))
        else:
            if "source" not in frame.columns:
                frame["source"] = spec.get("source") or spec.get("topic") or "candidate"
            if "track_id" not in frame.columns and spec.get("track_id") is not None:
                frame["track_id"] = spec.get("track_id")
            if "std_xy_m" not in frame.columns and spec.get("std_xy_m") is not None:
                frame["std_xy_m"] = spec.get("std_xy_m")
            if "std_z_m" not in frame.columns and spec.get("std_z_m") is not None:
                frame["std_z_m"] = spec.get("std_z_m")
            if "confidence" not in frame.columns and spec.get("confidence") is not None:
                frame["confidence"] = spec.get("confidence")
            if "class_name" not in frame.columns and spec.get("class_name") is not None:
                frame["class_name"] = spec.get("class_name")
            normalized = normalize_candidate_columns(frame)
            candidate_frames.append(CandidateFrame(normalized))
        loaded.append({"path": str(csv_path), "kind": kind, "status": "loaded", "rows": int(len(frame))})
    if not candidate_frames:
        raise ValueError(f"topic map {path} did not load any candidate exports")
    candidates = merge_candidate_frames(candidate_frames)
    truth = None
    if truth_frames:
        truth_rows = pd.concat([frame.rows for frame in truth_frames], ignore_index=True)
        truth = TruthFrame(normalize_truth_columns(truth_rows))
    return TopicExportBundle(candidates, truth, {"topic_map": str(path), "loaded_exports": loaded})


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
