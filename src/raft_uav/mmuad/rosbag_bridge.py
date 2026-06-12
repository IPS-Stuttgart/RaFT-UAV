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

from raft_uav.coordinates import LocalENUProjector
from raft_uav.mmuad.io import (
    load_candidate_file,
    load_point_cloud_file_as_candidates,
    load_truth_file,
    merge_candidate_frames,
    point_rows_to_candidates,
    read_json_table_export,
)
from raft_uav.mmuad.radar import radar_polar_frame_to_candidates
from raft_uav.mmuad.schema import (
    CandidateFrame,
    TruthFrame,
    normalize_candidate_columns,
    normalize_time_column_aliases,
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
        entry = {
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
        if _is_geodetic_kind(kind):
            entry["enu_origin_lla"] = "LAT,LON,ALT"
        exports.append(entry)
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
        if _is_radar_polar_kind(kind):
            candidate_frame = _load_topic_radar_polar_export(
                export_path,
                spec,
                sequence_id=sequence_id,
                source=str(spec.get("source") or spec.get("topic") or "radar_polar"),
            )
            candidate_frames.append(candidate_frame)
            row_count = len(candidate_frame.rows)
        elif _is_geodetic_kind(kind):
            if _is_truth_kind(kind):
                truth_frame = _load_topic_geodetic_truth_export(
                    export_path,
                    spec,
                    sequence_id=sequence_id,
                )
                truth_frames.append(truth_frame)
                row_count = len(truth_frame.rows)
            else:
                candidate_frame = _load_topic_geodetic_candidate_export(
                    export_path,
                    spec,
                    sequence_id=sequence_id,
                    source=str(spec.get("source") or spec.get("topic") or "geodetic"),
                )
                candidate_frames.append(candidate_frame)
                row_count = len(candidate_frame.rows)
        elif _is_truth_kind(kind):
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
    compact_type = msg_type.replace("_", "")
    truth_like = any(token in name for token in ("truth", "ground", "gt", "label", "mocap"))
    if "pointcloud2" in msg_type:
        return "pointcloud2_candidate"
    if compact_type.endswith("navsatfix"):
        return "navsatfix_truth" if truth_like else "navsatfix_candidate"
    if compact_type.endswith("geoposestamped") or compact_type.endswith("geopose"):
        return "geopose_truth" if truth_like else "geopose_candidate"
    if compact_type.endswith("geopointstamped") or compact_type.endswith("geopoint"):
        return "geopoint_truth" if truth_like else "geopoint_candidate"
    if "detection3darray" in compact_type:
        return "detection3d_array_truth" if truth_like else "detection3d_array_candidate"
    if "detection3d" in compact_type:
        return "detection3d_truth" if truth_like else "detection3d_candidate"
    if compact_type.endswith("markerarray"):
        return "marker_array_truth" if truth_like else "marker_array_candidate"
    if compact_type.endswith("marker"):
        return "marker_truth" if truth_like else "marker_candidate"
    if compact_type.endswith("multidofjointtrajectory"):
        return (
            "multidof_joint_trajectory_truth"
            if truth_like
            else "multidof_joint_trajectory_candidate"
        )
    if compact_type.endswith("multidofjointstate"):
        return "multidof_joint_state_truth" if truth_like else "multidof_joint_state_candidate"
    if "tfmessage" in msg_type or msg_type.endswith("/tfmessage"):
        return "tf_truth" if truth_like else "tf_candidate"
    if msg_type.endswith("/path") or msg_type.endswith("msg/path"):
        return "path_truth" if truth_like else "path_candidate"
    if msg_type.endswith("posearray") or "pose_array" in msg_type:
        return "pose_array_truth" if truth_like else "pose_array_candidate"
    if (
        compact_type.endswith("posewithcovariancestamped")
        or compact_type.endswith("posewithcovariance")
        or compact_type.endswith("/pose")
        or compact_type.endswith("msg/pose")
    ):
        return "pose_truth" if truth_like else "pose_candidate"
    if msg_type.endswith("posestamped") or "pose_stamped" in msg_type:
        return "pose_truth" if truth_like else "pose_candidate"
    if msg_type.endswith("pointstamped") or "point_stamped" in msg_type:
        return "point_truth" if truth_like else "point_candidate"
    if msg_type.endswith("transformstamped") or "transform_stamped" in msg_type:
        return "transform_truth" if truth_like else "transform_candidate"
    if msg_type.endswith("odometry"):
        return "odometry_truth" if truth_like else "odometry_candidate"
    return "truth" if truth_like else "candidate"


def _is_truth_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized == "truth" or normalized.endswith("_truth")


def _is_geodetic_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized.startswith(("navsatfix_", "geopoint_", "geopose_"))


def _is_radar_polar_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized in {
        "radar_polar",
        "radar_polar_candidate",
        "polar_radar",
        "polar_radar_candidate",
    }


_LATITUDE_ALIASES = ("latitude_deg", "latitude", "lat_deg", "lat")
_LONGITUDE_ALIASES = (
    "longitude_deg",
    "longitude",
    "lon_deg",
    "long_deg",
    "lon",
    "lng",
    "long",
)
_ALTITUDE_ALIASES = ("altitude_m", "altitude", "alt_m", "alt", "height_m", "height")


def _first_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> object | None:
    lower = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        original = lower.get(alias)
        if original is not None:
            return original
    return None


def _projector_from_spec(spec: dict[str, Any]) -> LocalENUProjector:
    raw = spec.get("enu_origin_lla", spec.get("origin_lla"))
    if raw is not None:
        latitude, longitude, altitude = _parse_lla_values(raw)
        return LocalENUProjector(latitude, longitude, altitude)
    latitude = spec.get("origin_latitude_deg", spec.get("origin_latitude"))
    longitude = spec.get("origin_longitude_deg", spec.get("origin_longitude"))
    altitude = spec.get("origin_altitude_m", spec.get("origin_altitude"))
    if latitude is None or longitude is None or altitude is None:
        raise ValueError(
            "geodetic topic exports require enu_origin_lla or "
            "origin_latitude_deg/origin_longitude_deg/origin_altitude_m"
        )
    return LocalENUProjector(float(latitude), float(longitude), float(altitude))


def _parse_lla_values(value: Any) -> tuple[float, float, float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, dict):
        return (
            float(value.get("latitude_deg", value.get("latitude"))),
            float(value.get("longitude_deg", value.get("longitude"))),
            float(value.get("altitude_m", value.get("altitude"))),
        )
    else:
        try:
            parts = list(value)
        except TypeError as exc:
            raise ValueError("enu_origin_lla must be LAT,LON,ALT") from exc
    if len(parts) != 3:
        raise ValueError("enu_origin_lla must contain LAT,LON,ALT")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except (TypeError, ValueError) as exc:
        raise ValueError("enu_origin_lla must contain numeric LAT,LON,ALT") from exc


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


def _load_topic_radar_polar_export(
    path: Path,
    spec: dict[str, Any],
    *,
    sequence_id: str,
    source: str,
) -> CandidateFrame:
    if not _is_table_export(path):
        raise ValueError(f"radar polar topic exports must be table files: {path}")
    frame = _apply_aliases(_read_topic_table(path), spec)
    return radar_polar_frame_to_candidates(
        frame,
        source=source,
        sequence_id=sequence_id,
        azimuth_convention=str(
            spec.get(
                "azimuth_convention",
                spec.get("radar_azimuth_convention", "north-clockwise"),
            )
        ),
        angle_unit=str(spec.get("angle_unit", spec.get("radar_angle_unit", "deg"))),
        range_std_m=float(
            spec.get("range_std_m", spec.get("radar_polar_range_std_m", 2.0))
        ),
        angle_std_deg=float(
            spec.get("angle_std_deg", spec.get("radar_polar_angle_std_deg", 2.0))
        ),
        z_std_m=float(spec.get("z_std_m", spec.get("radar_polar_z_std_m", 5.0))),
    )


def _load_topic_geodetic_truth_export(
    path: Path,
    spec: dict[str, Any],
    *,
    sequence_id: str,
) -> TruthFrame:
    rows = _load_topic_geodetic_rows(path, spec, sequence_id=sequence_id, source=None)
    return TruthFrame(normalize_truth_columns(rows, default_sequence_id=sequence_id))


def _load_topic_geodetic_candidate_export(
    path: Path,
    spec: dict[str, Any],
    *,
    sequence_id: str,
    source: str,
) -> CandidateFrame:
    rows = _load_topic_geodetic_rows(path, spec, sequence_id=sequence_id, source=source)
    for column in ("track_id", "std_xy_m", "std_z_m", "confidence", "class_name"):
        if column not in rows.columns and spec.get(column) is not None:
            rows[column] = spec.get(column)
    return CandidateFrame(
        normalize_candidate_columns(rows, default_sequence_id=sequence_id)
    )


def _load_topic_geodetic_rows(
    path: Path,
    spec: dict[str, Any],
    *,
    sequence_id: str,
    source: str | None,
) -> pd.DataFrame:
    if not _is_table_export(path):
        raise ValueError(f"geodetic topic exports must be table files: {path}")
    frame = normalize_time_column_aliases(_apply_aliases(_read_topic_table(path), spec))
    latitude_col = _first_column(frame, _LATITUDE_ALIASES)
    longitude_col = _first_column(frame, _LONGITUDE_ALIASES)
    altitude_col = _first_column(frame, _ALTITUDE_ALIASES)
    if latitude_col is None or longitude_col is None:
        raise ValueError("geodetic topic exports require latitude/longitude columns")
    if altitude_col is None and spec.get("altitude_m") is None:
        raise ValueError("geodetic topic exports require altitude_m or altitude")
    if "time_s" not in frame.columns:
        raise ValueError("geodetic topic exports require time_s/timestamp columns")
    latitude = pd.to_numeric(frame[latitude_col], errors="coerce")
    longitude = pd.to_numeric(frame[longitude_col], errors="coerce")
    if altitude_col is not None:
        altitude = pd.to_numeric(frame[altitude_col], errors="coerce")
    else:
        altitude = pd.Series(float(spec["altitude_m"]), index=frame.index)
    projector = _projector_from_spec(spec)
    enu = projector.transform_many(
        latitude.to_numpy(dtype=float),
        longitude.to_numpy(dtype=float),
        altitude.to_numpy(dtype=float),
    )
    rows = frame.copy()
    if "sequence_id" not in rows.columns:
        rows["sequence_id"] = sequence_id
    if source is not None and "source" not in rows.columns:
        rows["source"] = source
    rows["x_m"] = enu[:, 0]
    rows["y_m"] = enu[:, 1]
    rows["z_m"] = enu[:, 2]
    rows["latitude_deg"] = latitude
    rows["longitude_deg"] = longitude
    rows["altitude_m"] = altitude
    return rows


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
    if path.suffix.lower() == ".json":
        return read_json_table_export(
            path,
            preferred=(
                "points",
                "point_cloud",
                "pointcloud",
                "candidates",
                "detections",
                "truth",
                "ground_truth",
                "gt",
                "poses",
                "trajectory",
                "trajectories",
                "rows",
                "data",
            ),
        )
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _apply_aliases(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    aliases = spec.get("column_aliases", {}) or {}
    return frame.rename(columns={str(key): str(value) for key, value in aliases.items()})


def _is_table_export(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".tsv", ".txt", ".json"}


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
