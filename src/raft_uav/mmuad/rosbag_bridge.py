"""ROS-bag bridge helpers for MMUAD exported data.

The helpers avoid depending on ROS at import time.  They can inspect ROS2
``metadata.yaml`` directories, optionally call ``rosbag info --yaml`` for ROS1
bags when the command exists, and load normalized topic exports via a topic-map
JSON/YAML file.  This is a bridge toward native support; it is not a binary
message parser.
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
from raft_uav.mmuad.camera import (
    camera_detection_frame_to_candidates,
    load_camera_models_from_files,
)
from raft_uav.mmuad.io import (
    JSON_TABLE_SUFFIXES,
    DELIMITED_TABLE_SUFFIXES,
    data_file_suffix,
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

ROS2_RECORDING_FILE_SUFFIXES = {".db3", ".mcap"}


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
            "recommendation": (
                "No metadata.yaml found; export topics to CSV and use "
                "--topic-map-file/--topic-map-json."
            ),
        }
    if path.suffix.lower() == ".bag":
        return _inspect_ros1_bag(path)
    if path.suffix.lower() in ROS2_RECORDING_FILE_SUFFIXES:
        return _inspect_native_ros_recording_file(path)
    return {
        "path": str(path),
        "kind": "unknown",
        "suffix": path.suffix.lower(),
        "recommendation": "Unsupported bag path. Use layout inspection or exported CSV topic maps.",
    }


def write_topic_map_template(
    report: dict[str, Any],
    path: Path,
    *,
    template_mode: str = "export",
) -> Path:
    """Write a topic-map JSON template from an inspection report."""

    mode = _normalize_topic_map_template_mode(template_mode)
    topics = report.get("topics", [])
    exports = []
    for idx, topic in enumerate(topics):
        name = str(topic.get("name", topic.get("topic", f"topic_{idx}")))
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip("/")).strip("_") or f"topic_{idx}"
        kind = _infer_topic_map_kind(topic)
        entry = {
            "topic": name,
            "kind": kind,
            "source": safe if not _is_truth_kind(kind) else None,
            "sequence_id": report.get(
                "sequence_id",
                Path(str(report.get("path", "sequence"))).stem,
            ),
        }
        if mode == "export":
            entry["path"] = f"exports/{safe}.csv"
            entry["column_aliases"] = {
                "stamp": "time_s",
                "timestamp": "time_s",
                "x": "x_m",
                "y": "y_m",
                "z": "z_m",
            }
        if _is_camera_detection_kind(kind):
            entry["camera_calibration_file"] = "PATH/TO/camera_info.json"
            if mode == "native":
                entry["camera_fixed_depth_m"] = "SET_DEPTH_OR_REMOVE_IF_MESSAGE_HAS_DEPTH"
            else:
                entry["column_aliases"].update(
                    {
                        "center_x": "u_px",
                        "center_y": "v_px",
                        "bbox_center_x": "u_px",
                        "bbox_center_y": "v_px",
                        "score": "confidence",
                    }
                )
        if _is_radar_polar_kind(kind):
            if mode == "native":
                entry["angle_unit"] = "rad"
            else:
                entry["column_aliases"].update(
                    {
                        "range": "range_m",
                        "r": "range_m",
                        "rho": "range_m",
                        "bearing": "azimuth_deg",
                        "az": "azimuth_deg",
                        "azimuth": "azimuth_deg",
                        "el": "elevation_deg",
                    }
                )
        if _is_geodetic_kind(kind):
            entry["enu_origin_lla"] = "LAT,LON,ALT"
        exports.append(entry)
    payload = {
        "schema": "raft-uav-mmuad-topic-map-v1",
        "template_mode": mode,
        "sequence_id": report.get(
            "sequence_id",
            Path(str(report.get("path", "sequence"))).stem,
        ),
        "description": _topic_map_template_description(mode),
        "exports": exports,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _normalize_topic_map_template_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"export", "native"}:
        raise ValueError("template_mode must be 'export' or 'native'")
    return mode


def _topic_map_template_description(mode: str) -> str:
    if mode == "native":
        return (
            "Native ROS extraction template. Edit sources, calibration/depth, "
            "geodetic origins, and topic-specific options, then run with "
            "--native-ros-extract-output-dir and --topic-map-file."
        )
    return "Edit paths and aliases to point at CSV exports of ROS topics."


def load_topic_map_payload(path: Path) -> dict[str, Any]:
    """Load a topic-map metadata file from JSON or YAML."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception:
            payload = json.loads(text)
        else:
            try:
                payload = yaml.safe_load(text)
            except Exception as exc:
                raise ValueError(f"invalid topic map YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"topic map {path} must contain a mapping")
    return payload


def load_topic_map_exports(path: Path, *, base_dir: Path | None = None) -> TopicExportBundle:
    """Load normalized candidates/truth from a topic-map JSON/YAML file."""

    path = Path(path)
    payload = load_topic_map_payload(path)
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
        if _is_camera_detection_kind(kind):
            candidate_frame = _load_topic_camera_detection_export(
                export_path,
                spec,
                sequence_id=sequence_id,
                source=str(spec.get("source") or spec.get("topic") or "camera"),
                base_dir=base,
            )
            candidate_frames.append(candidate_frame)
            row_count = len(candidate_frame.rows)
        elif _is_radar_polar_kind(kind):
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
    if _looks_like_livox_custom_topic(name, msg_type):
        return "livox_custommsg_candidate"
    if "pointcloud2" in msg_type:
        return "pointcloud2_candidate"
    if _looks_like_polar_radar_topic(name, msg_type):
        return "radar_polar_candidate"
    if compact_type.endswith("navsatfix"):
        return "navsatfix_truth" if truth_like else "navsatfix_candidate"
    if compact_type.endswith("geoposestamped") or compact_type.endswith("geopose"):
        return "geopose_truth" if truth_like else "geopose_candidate"
    if compact_type.endswith("geopointstamped") or compact_type.endswith("geopoint"):
        return "geopoint_truth" if truth_like else "geopoint_candidate"
    if compact_type.endswith("camerainfo"):
        return "camera_info_calibration"
    if "detection2darray" in compact_type:
        return "camera_detections_candidate"
    if "detection2d" in compact_type:
        return "camera_detections_candidate"
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
    if msg_type.endswith("/point") or compact_type.endswith("msg/point"):
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


def _is_pointcloud_topic_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized in {
        "pointcloud2_candidate",
        "livox_custom_candidate",
        "livox_custommsg_candidate",
        "livox_custom_pointcloud_candidate",
        "livox_pointcloud_candidate",
    }


def _looks_like_livox_custom_topic(name: str, msg_type: str) -> bool:
    text = f"{name} {msg_type}".lower().replace("_", "").replace("-", "")
    return "livox" in text and (
        "custommsg" in text or "custompoint" in text or "custom" in text
    )


def _looks_like_polar_radar_topic(name: str, msg_type: str) -> bool:
    text = f"{name} {msg_type}".lower().replace("_", "").replace("-", "")
    radar_like = any(token in text for token in ("radar", "mmwave", "mmw"))
    polar_like = any(
        token in text
        for token in (
            "polar",
            "rangeazimuth",
            "rangebearing",
            "rangeangle",
            "spherical",
        )
    )
    return radar_like and polar_like


def _is_camera_detection_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized in {
        "camera_detection",
        "camera_detection_candidate",
        "camera_detections",
        "camera_detections_candidate",
        "image_detection",
        "image_detection_candidate",
        "image_detections",
        "image_detections_candidate",
        "detection2d",
        "detection2d_candidate",
        "detection2d_array",
        "detection2d_array_candidate",
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
    explicit_source = spec.get("source")
    source = str(spec.get("source") or spec.get("topic") or "candidate")
    kind = str(spec.get("kind", "candidate")).strip().lower()
    if _is_pointcloud_topic_kind(kind):
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
        if explicit_source is not None:
            frame["source"] = source
        elif "source" not in frame.columns:
            frame["source"] = source
        for column in ("track_id", "std_xy_m", "std_z_m", "confidence", "class_name"):
            if column not in frame.columns and spec.get(column) is not None:
                frame[column] = spec.get(column)
        return CandidateFrame(
            normalize_candidate_columns(
                frame,
                default_sequence_id=sequence_id,
                default_source=source,
            )
        )
    frame = load_candidate_file(path, default_sequence_id=sequence_id, source=source)
    rows = frame.rows.copy()
    for column in ("track_id", "std_xy_m", "std_z_m", "confidence", "class_name"):
        if spec.get(column) is not None:
            rows[column] = spec.get(column)
    return CandidateFrame(
        normalize_candidate_columns(
            rows,
            default_sequence_id=sequence_id,
            default_source=source,
        )
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


def _load_topic_camera_detection_export(
    path: Path,
    spec: dict[str, Any],
    *,
    sequence_id: str,
    source: str,
    base_dir: Path,
) -> CandidateFrame:
    if not _is_table_export(path):
        raise ValueError(f"camera detection topic exports must be table files: {path}")
    calibration_files = _topic_camera_calibration_files(
        spec,
        export_path=path,
        base_dir=base_dir,
    )
    if not calibration_files:
        raise ValueError(
            "camera detection topic exports require camera_calibration_file "
            "or a nearby camera_info/intrinsics file"
        )
    camera_models = load_camera_models_from_files(
        calibration_files,
        source_hint_from_path=lambda _path: source,
    )
    frame = _apply_aliases(_read_topic_table(path), spec)
    return camera_detection_frame_to_candidates(
        frame,
        camera_models=camera_models,
        source=source,
        sequence_id=sequence_id,
        fixed_depth_m=_optional_float(
            spec,
            "camera_fixed_depth_m",
            "fixed_depth_m",
            "depth_m",
        ),
        std_xy_m=float(spec.get("camera_std_xy_m", spec.get("std_xy_m", 5.0))),
        std_z_m=float(spec.get("camera_std_z_m", spec.get("std_z_m", 10.0))),
    )


def _topic_camera_calibration_files(
    spec: dict[str, Any],
    *,
    export_path: Path,
    base_dir: Path,
) -> list[Path]:
    explicit = _topic_path_values(
        spec,
        (
            "camera_calibration_file",
            "camera_calibration_path",
            "camera_intrinsics_file",
            "camera_intrinsics_path",
            "camera_info_file",
            "camera_info_path",
            "calibration_file",
            "calibration_path",
        ),
        (
            "camera_calibration_files",
            "camera_intrinsics_files",
            "camera_info_files",
            "calibration_files",
        ),
    )
    candidates: list[Path] = []
    for value in explicit:
        candidates.append(
            _resolve_topic_path(
                str(value),
                base_dir=base_dir,
                sibling_dir=export_path.parent,
            )
        )
    if not candidates:
        for directory in (export_path.parent, base_dir):
            for name in _CAMERA_CALIBRATION_FILENAMES:
                candidates.append(directory / name)
    return _unique_existing_paths(candidates)


def _topic_path_values(
    spec: dict[str, Any],
    scalar_keys: tuple[str, ...],
    list_keys: tuple[str, ...],
) -> list[Any]:
    values: list[Any] = []
    for key in scalar_keys:
        value = spec.get(key)
        if value not in (None, ""):
            values.append(value)
    for key in list_keys:
        value = spec.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, (list, tuple)):
            values.extend(item for item in value if item not in (None, ""))
        else:
            values.append(value)
    return values


def _resolve_topic_path(value: str, *, base_dir: Path, sibling_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    sibling = sibling_dir / path
    if sibling.exists():
        return sibling
    return base_dir / path


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen or not Path(path).exists():
            continue
        seen.add(resolved)
        unique.append(Path(path))
    return unique


def _optional_float(spec: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = spec.get(key)
        if value not in (None, ""):
            return float(value)
    return None


_CAMERA_CALIBRATION_FILENAMES = (
    "camera_info.json",
    "camera_info.yaml",
    "camera_info.yml",
    "camera_calibration.json",
    "camera_calibration.yaml",
    "camera_calibration.yml",
    "camera_intrinsics.json",
    "camera_intrinsics.yaml",
    "camera_intrinsics.yml",
    "intrinsics.json",
    "intrinsics.yaml",
    "intrinsics.yml",
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
        normalize_candidate_columns(
            rows,
            default_sequence_id=sequence_id,
            default_source=source,
        )
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
    suffix = data_file_suffix(path)
    if suffix in JSON_TABLE_SUFFIXES:
        return read_json_table_export(
            path,
            preferred=(
                "points",
                "point_cloud",
                "pointcloud",
                "candidates",
                "detections",
                "objects",
                "targets",
                "measurements",
                "returns",
                "predictions",
                "truth",
                "ground_truth",
                "gt",
                "fixes",
                "gps",
                "navsat",
                "navsatfix",
                "geopoints",
                "geoposes",
                "locations",
                "poses",
                "trajectory",
                "trajectories",
                "rows",
                "data",
            ),
        )
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _apply_aliases(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    aliases = spec.get("column_aliases", {}) or {}
    out = frame.copy()
    rename: dict[Any, str] = {}
    for key, value in aliases.items():
        source = str(key)
        target = str(value)
        if source not in out.columns or source == target:
            continue
        if target in out.columns:
            out[target] = out[target].where(out[target].notna(), out[source])
            out = out.drop(columns=[source])
            continue
        rename[source] = target
    return out.rename(columns=rename)


def _is_table_export(path: Path) -> bool:
    return data_file_suffix(path) in DELIMITED_TABLE_SUFFIXES | JSON_TABLE_SUFFIXES


def _inspect_ros2_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    metadata = _load_yaml_mapping(text)
    info = _ros2_bagfile_information(metadata)
    topics = _ros2_topics_from_metadata(info) if info is not None else []
    if not topics:
        topics = _ros2_topics_from_metadata_text(text)
    metadata_fields = (
        _ros2_metadata_fields_from_mapping(info)
        if info is not None
        else _ros2_metadata_fields_from_text(text)
    )
    root = path.parent
    db_files = sorted(str(item.relative_to(root)) for item in root.rglob("*.db3"))
    mcap_files = sorted(str(item.relative_to(root)) for item in root.rglob("*.mcap"))
    report = {
        "path": str(root),
        "kind": "ros2_bag_directory",
        "metadata_yaml": str(path),
        "topics": topics,
        "db3_files": db_files,
        "mcap_files": mcap_files,
        "recommendation": (
            "Export relevant topics to CSV, then run with "
            "--topic-map-file/--topic-map-json."
        ),
    }
    report.update(metadata_fields)
    return report


def _load_yaml_mapping(text: str) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        payload = yaml.safe_load(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _ros2_bagfile_information(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    info = metadata.get("rosbag2_bagfile_information", metadata)
    return info if isinstance(info, dict) else None


def _ros2_topics_from_metadata(info: dict[str, Any]) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    items = info.get("topics_with_message_count", [])
    if not isinstance(items, list):
        return topics
    for item in items:
        if not isinstance(item, dict):
            continue
        topic_metadata = item.get("topic_metadata", {})
        if not isinstance(topic_metadata, dict):
            topic_metadata = {}
        topic: dict[str, Any] = {}
        for source, target in (
            ("name", "name"),
            ("type", "type"),
            ("serialization_format", "serialization_format"),
        ):
            value = topic_metadata.get(source)
            if value is not None:
                topic[target] = str(value)
        if "message_count" in item:
            topic["message_count"] = _maybe_int(item["message_count"])
        if topic:
            topics.append(topic)
    return topics


def _ros2_metadata_fields_from_mapping(info: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in (
        "storage_identifier",
        "serialization_format",
        "compression_format",
        "compression_mode",
    ):
        value = info.get(key)
        if value not in (None, ""):
            fields[key] = str(value)
    if "message_count" in info:
        fields["total_message_count"] = _maybe_int(info["message_count"])
    relative_files = _string_list(info.get("relative_file_paths"))
    if relative_files:
        fields["relative_file_paths"] = relative_files
    duration_s = _duration_seconds(info.get("duration"))
    if duration_s is not None:
        fields["duration_s"] = duration_s
    starting_time_s = _starting_time_seconds(info.get("starting_time"))
    if starting_time_s is not None:
        fields["starting_time_s"] = starting_time_s
    return fields


def _ros2_topics_from_metadata_text(text: str) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    topic_indent = 0
    for line in text.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("topic_metadata:") or stripped.startswith("- topic_metadata:"):
            if current:
                topics.append(current)
            current = {}
            topic_indent = indent
            continue
        if current is not None and indent <= topic_indent and stripped:
            topics.append(current)
            current = None
        if current is not None and stripped.startswith("name:"):
            current["name"] = _yaml_scalar(stripped.split(":", 1)[1])
        elif current is not None and stripped.startswith("type:"):
            current["type"] = _yaml_scalar(stripped.split(":", 1)[1])
        elif current is not None and stripped.startswith("serialization_format:"):
            current["serialization_format"] = _yaml_scalar(stripped.split(":", 1)[1])
        elif current is not None and stripped.startswith("message_count:"):
            current["message_count"] = _maybe_int(_yaml_scalar(stripped.split(":", 1)[1]))
    if current:
        topics.append(current)
    return topics


def _ros2_metadata_fields_from_text(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in (
        "storage_identifier",
        "serialization_format",
        "compression_format",
        "compression_mode",
    ):
        value = _yaml_lowest_indent_scalar(text, key)
        if value not in (None, ""):
            fields[key] = str(value)
    message_count = _yaml_lowest_indent_scalar(text, "message_count")
    if message_count is not None:
        fields["total_message_count"] = _maybe_int(message_count)
    relative_files = _yaml_sequence(text, "relative_file_paths")
    if relative_files:
        fields["relative_file_paths"] = relative_files
    duration_ns = _yaml_nested_int(text, "duration", ("nanoseconds", "nanoseconds_since_epoch"))
    if duration_ns is not None:
        fields["duration_s"] = duration_ns / 1.0e9
    starting_ns = _yaml_nested_int(text, "starting_time", ("nanoseconds_since_epoch", "nanoseconds"))
    if starting_ns is not None:
        fields["starting_time_s"] = starting_ns / 1.0e9
    return fields


def _duration_seconds(value: Any) -> float | None:
    if isinstance(value, dict):
        nanos = value.get("nanoseconds")
        if nanos is not None:
            return float(nanos) / 1.0e9
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _starting_time_seconds(value: Any) -> float | None:
    if isinstance(value, dict):
        nanos = value.get("nanoseconds_since_epoch", value.get("nanoseconds"))
        if nanos is not None:
            return float(nanos) / 1.0e9
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _maybe_int(value: Any) -> int | Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _yaml_scalar(value: str) -> str:
    value = value.strip().strip("'\"")
    if value.startswith("[") and value.endswith("]"):
        return value
    return value


def _yaml_lowest_indent_scalar(text: str, key: str) -> str | None:
    matches: list[tuple[int, str]] = []
    prefix = f"{key}:"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        value = _yaml_scalar(stripped.split(":", 1)[1])
        if value == "":
            continue
        indent = len(line) - len(line.lstrip())
        matches.append((indent, value))
    if not matches:
        return None
    return sorted(matches, key=lambda item: item[0])[0][1]


def _yaml_sequence(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(f"{key}:"):
            continue
        inline_value = _yaml_scalar(stripped.split(":", 1)[1])
        if inline_value.startswith("[") and inline_value.endswith("]"):
            return [
                item.strip().strip("'\"")
                for item in inline_value.strip("[]").split(",")
                if item.strip()
            ]
        section_indent = len(line) - len(line.lstrip())
        values: list[str] = []
        for child in lines[index + 1 :]:
            child_stripped = child.strip()
            if not child_stripped:
                continue
            child_indent = len(child) - len(child.lstrip())
            if child_indent <= section_indent:
                break
            if child_stripped.startswith("-"):
                values.append(_yaml_scalar(child_stripped[1:]))
        return values
    return []


def _yaml_nested_int(text: str, parent: str, child_keys: tuple[str, ...]) -> int | None:
    lines = text.splitlines()
    matches: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(f"{parent}:"):
            continue
        parent_indent = len(line) - len(line.lstrip())
        for child in lines[index + 1 :]:
            child_stripped = child.strip()
            if not child_stripped:
                continue
            child_indent = len(child) - len(child.lstrip())
            if child_indent <= parent_indent:
                break
            for child_key in child_keys:
                if not child_stripped.startswith(f"{child_key}:"):
                    continue
                value = _maybe_int(_yaml_scalar(child_stripped.split(":", 1)[1]))
                if isinstance(value, int):
                    matches.append((parent_indent, value))
        if matches:
            break
    if not matches:
        return None
    return sorted(matches, key=lambda item: item[0])[0][1]


def _inspect_ros1_bag(path: Path) -> dict[str, Any]:
    if shutil.which("rosbag") is None:
        return {
            "path": str(path),
            "kind": "ros1_bag",
            "rosbag_cli_available": False,
            "topics": [],
            "recommendation": (
                "Install ROS/rosbag or export topics to CSV and use "
                "--topic-map-file/--topic-map-json."
            ),
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
        "recommendation": (
            "Export relevant topics to CSV, then run with "
            "--topic-map-file/--topic-map-json."
        ),
    }


def _inspect_native_ros_recording_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    storage_identifier = suffix.lstrip(".")
    try:
        from rosbags.highlevel import AnyReader  # type: ignore[import-not-found]
    except Exception as exc:
        return {
            "path": str(path),
            "kind": "ros2_recording_file",
            "suffix": suffix,
            "storage_identifier": storage_identifier,
            "rosbags_available": False,
            "topics": [],
            "recommendation": (
                "Install the optional 'rosbags' package to inspect native "
                ".db3/.mcap topics, or export topics to CSV and use "
                "--topic-map-file/--topic-map-json."
            ),
            "native_reader_error": str(exc),
        }
    try:
        with AnyReader([path]) as reader:
            topics = [_topic_from_native_connection(connection) for connection in reader.connections]
    except Exception as exc:
        return {
            "path": str(path),
            "kind": "ros2_recording_file",
            "suffix": suffix,
            "storage_identifier": storage_identifier,
            "rosbags_available": True,
            "topics": [],
            "recommendation": (
                "Native reader could not inspect this recording. Export topics "
                "to CSV or inspect the bag with ROS tooling."
            ),
            "native_reader_error": str(exc),
        }
    total_count = _total_topic_message_count(topics)
    report: dict[str, Any] = {
        "path": str(path),
        "kind": "ros2_recording_file",
        "suffix": suffix,
        "storage_identifier": storage_identifier,
        "rosbags_available": True,
        "topics": topics,
        "recommendation": (
            "Use --topic-map-template-json to create a native topic-map "
            "template, then run --native-ros-extract-output-dir with the "
            "edited topic map."
        ),
    }
    if total_count is not None:
        report["total_message_count"] = total_count
    return report


def _topic_from_native_connection(connection: Any) -> dict[str, Any]:
    topic: dict[str, Any] = {
        "name": str(getattr(connection, "topic", "")),
        "type": str(getattr(connection, "msgtype", "")),
    }
    message_count = _native_connection_message_count(connection)
    if message_count is not None:
        topic["message_count"] = message_count
    serialization_format = _native_connection_serialization_format(connection)
    if serialization_format:
        topic["serialization_format"] = serialization_format
    return topic


def _native_connection_message_count(connection: Any) -> int | None:
    for attr in ("msgcount", "message_count", "count"):
        value = getattr(connection, attr, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _native_connection_serialization_format(connection: Any) -> str | None:
    value = getattr(connection, "serialization_format", None)
    if value not in (None, ""):
        return str(value)
    ext = getattr(connection, "ext", None)
    value = getattr(ext, "serialization_format", None)
    if value not in (None, ""):
        return str(value)
    return None


def _total_topic_message_count(topics: list[dict[str, Any]]) -> int | None:
    counts = [topic.get("message_count") for topic in topics]
    if not counts or any(not isinstance(count, int) for count in counts):
        return None
    return int(sum(counts))
