"""Optional native ROS bag extraction toward full MMUAD support.

This module is intentionally optional-dependency safe.  It uses the ``rosbags``
package when available, but the rest of the MMUAD adapter imports without ROS.
The extractor currently supports common message families that appear in UAV
tracking logs:

* ``sensor_msgs/msg/PointCloud2`` -> clustered candidate detections;
* ``sensor_msgs/msg/NavSatFix`` -> geodetic rows projected into local ENU;
* ``geographic_msgs/msg/GeoPointStamped`` / ``GeoPoseStamped`` -> local ENU rows;
* ``vision_msgs/msg/Detection3D`` / ``Detection3DArray`` -> bbox center rows;
* ``visualization_msgs/msg/Marker`` / ``MarkerArray`` -> marker position rows;
* ``geometry_msgs/msg/Pose`` / ``PoseStamped`` /
  ``PoseWithCovariance(Stamped)`` -> truth rows or candidate rows;
* ``geometry_msgs/msg/PoseArray`` -> batched truth rows or candidate rows;
* ``geometry_msgs/msg/PointStamped`` -> truth rows or candidate rows;
* ``geometry_msgs/msg/TransformStamped`` -> truth rows or candidate rows;
* ``tf2_msgs/msg/TFMessage`` -> transform truth rows or candidate rows;
* ``nav_msgs/msg/Path`` -> trajectory truth rows or candidate rows;
* ``nav_msgs/msg/Odometry`` -> truth rows or candidate rows.
* ``sensor_msgs/msg/MultiDOFJointState`` /
  ``trajectory_msgs/msg/MultiDOFJointTrajectory`` -> transform rows.

Unknown topics are recorded in the extraction manifest and skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import pandas as pd

from raft_uav.coordinates import LocalENUProjector
from raft_uav.mmuad.io import merge_candidate_frames
from raft_uav.mmuad.pointcloud2 import pointcloud2_to_candidates
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame, normalize_truth_columns


@dataclass(frozen=True)
class NativeRosExtraction:
    """Result of optional native ROS extraction."""

    candidates: CandidateFrame | None
    truth: TruthFrame | None
    manifest: dict[str, Any]


def extract_native_rosbag_topic_map(
    *,
    bag_path: Path,
    topic_map_json: Path,
    output_dir: Path | None = None,
    voxel_size_m: float = 0.75,
    min_points: int = 3,
) -> NativeRosExtraction:
    """Extract supported ROS topics according to a topic-map JSON.

    The topic map follows the same high-level structure as the CSV export bridge
    but may use native kinds:

    ``pointcloud2_candidate``
        Decode ``sensor_msgs/msg/PointCloud2`` and cluster points.
    ``navsatfix_truth`` / ``geopoint_truth`` / ``geopose_truth`` /
    ``navsatfix_candidate`` / ``geopoint_candidate`` / ``geopose_candidate``
        Project geodetic GPS/geographic positions into local ENU rows. These
        entries require ``enu_origin_lla`` or separate origin latitude,
        longitude, and altitude fields in the topic map.
    ``detection3d_truth`` / ``detection3d_array_truth`` /
    ``detection3d_candidate`` / ``detection3d_array_candidate``
        Convert vision_msgs 3D detection bbox centers into truth/candidate rows.
    ``marker_truth`` / ``marker_array_truth`` /
    ``marker_candidate`` / ``marker_array_candidate``
        Convert visualization marker poses into truth/candidate rows.
    ``pose_truth`` / ``odometry_truth``
        Convert pose, pose-with-covariance, or odometry messages into truth rows.
    ``multidof_joint_state_truth`` / ``multidof_joint_trajectory_truth`` /
    ``multidof_joint_state_candidate`` / ``multidof_joint_trajectory_candidate``
        Convert MultiDOF transforms into truth/candidate rows.
    ``point_truth`` / ``transform_truth`` / ``tf_truth`` / ``path_truth`` /
    ``pose_array_truth``
        Convert position-only messages into truth rows.
    ``pose_candidate`` / ``odometry_candidate`` / ``point_candidate`` /
    ``transform_candidate`` / ``tf_candidate`` / ``path_candidate`` /
    ``pose_array_candidate``
        Convert pose/odometry messages into candidate detections.
    """

    payload = json.loads(Path(topic_map_json).read_text(encoding="utf-8"))
    specs = list(payload.get("exports", []))
    if not specs:
        raise ValueError(f"topic map {topic_map_json} has no exports")
    try:
        from rosbags.highlevel import AnyReader  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised only without dependency
        raise RuntimeError(
            "native ROS extraction requires the optional 'rosbags' package; "
            "install it or export topics to CSV and use --topic-map-json"
        ) from exc

    by_topic = {str(spec["topic"]): spec for spec in specs if "topic" in spec}
    candidate_frames: list[CandidateFrame] = []
    truth_rows: list[dict[str, Any]] = []
    extracted: list[dict[str, Any]] = []
    sequence_id = str(payload.get("sequence_id", Path(bag_path).stem))
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    with AnyReader([Path(bag_path)]) as reader:
        topic_connections = [connection for connection in reader.connections if connection.topic in by_topic]
        for connection, timestamp_ns, rawdata in reader.messages(connections=topic_connections):
            spec = by_topic[connection.topic]
            kind = str(spec.get("kind", "candidate")).strip().lower()
            message = reader.deserialize(rawdata, connection.msgtype)
            time_s = _message_time_s(message, timestamp_ns)
            source = str(spec.get("source") or connection.topic.strip("/").replace("/", "_"))
            try:
                if kind == "pointcloud2_candidate":
                    frame = pointcloud2_to_candidates(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        source=source,
                        voxel_size_m=voxel_size_m,
                        min_points=min_points,
                    )
                    candidate_frames.append(frame)
                    rows = len(frame.rows)
                elif kind in {"navsatfix_truth", "geopoint_truth", "geopose_truth"}:
                    rows_for_message = geodetic_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        projector=_projector_from_spec(spec),
                        frame_id=spec.get("frame_id"),
                    )
                    truth_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif kind in {
                    "navsatfix_candidate",
                    "geopoint_candidate",
                    "geopose_candidate",
                }:
                    rows_for_message = geodetic_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        projector=_projector_from_spec(spec),
                        frame_id=spec.get("frame_id"),
                    )
                    candidate_rows = []
                    for row in rows_for_message:
                        row.update(
                            {
                                "source": source,
                                "track_id": spec.get(
                                    "track_id",
                                    row.get("child_frame_id", row.get("frame_id", source)),
                                ),
                                "std_xy_m": spec.get(
                                    "std_xy_m",
                                    row.get("std_xy_m", 5.0),
                                ),
                                "std_z_m": spec.get(
                                    "std_z_m",
                                    row.get("std_z_m", 10.0),
                                ),
                                "confidence": spec.get("confidence", 1.0),
                                "class_name": spec.get("class_name", "uav"),
                            }
                        )
                        candidate_rows.append(row)
                    if candidate_rows:
                        candidate_frames.append(
                            CandidateFrame(pd.DataFrame.from_records(candidate_rows))
                        )
                    rows = len(candidate_rows)
                elif kind in {"detection3d_truth", "detection3d_array_truth"}:
                    rows_for_message = detection3d_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        frame_id=spec.get("frame_id"),
                    )
                    truth_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif kind in {"detection3d_candidate", "detection3d_array_candidate"}:
                    rows_for_message = detection3d_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        frame_id=spec.get("frame_id"),
                    )
                    candidate_rows = []
                    for row in rows_for_message:
                        row.update(
                            {
                                "source": source,
                                "track_id": spec.get(
                                    "track_id",
                                    row.get("detection_id", source),
                                ),
                                "std_xy_m": spec.get("std_xy_m", 2.0),
                                "std_z_m": spec.get("std_z_m", 5.0),
                                "confidence": spec.get(
                                    "confidence",
                                    row.get("confidence", 1.0),
                                ),
                                "class_name": spec.get(
                                    "class_name",
                                    row.get("class_name", "uav"),
                                ),
                            }
                        )
                        candidate_rows.append(row)
                    if candidate_rows:
                        candidate_frames.append(
                            CandidateFrame(pd.DataFrame.from_records(candidate_rows))
                        )
                    rows = len(candidate_rows)
                elif kind in {"marker_truth", "marker_array_truth"}:
                    rows_for_message = marker_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        frame_id=spec.get("frame_id"),
                    )
                    truth_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif kind in {"marker_candidate", "marker_array_candidate"}:
                    rows_for_message = marker_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        frame_id=spec.get("frame_id"),
                    )
                    candidate_rows = []
                    for row in rows_for_message:
                        row.update(
                            {
                                "source": source,
                                "track_id": spec.get(
                                    "track_id",
                                    row.get(
                                        "marker_track_id",
                                        row.get("marker_id", source),
                                    ),
                                ),
                                "std_xy_m": spec.get("std_xy_m", 2.0),
                                "std_z_m": spec.get("std_z_m", 5.0),
                                "confidence": spec.get(
                                    "confidence",
                                    row.get("confidence", 1.0),
                                ),
                                "class_name": spec.get(
                                    "class_name",
                                    row.get("class_name", "uav"),
                                ),
                            }
                        )
                        candidate_rows.append(row)
                    if candidate_rows:
                        candidate_frames.append(
                            CandidateFrame(pd.DataFrame.from_records(candidate_rows))
                        )
                    rows = len(candidate_rows)
                elif kind in {
                    "multidof_joint_state_truth",
                    "multidof_joint_trajectory_truth",
                }:
                    rows_for_message = multidof_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        frame_id=spec.get("frame_id"),
                    )
                    truth_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif kind in {
                    "multidof_joint_state_candidate",
                    "multidof_joint_trajectory_candidate",
                }:
                    rows_for_message = multidof_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        frame_id=spec.get("frame_id"),
                    )
                    candidate_rows = []
                    for row in rows_for_message:
                        row.update(
                            {
                                "source": source,
                                "track_id": spec.get(
                                    "track_id",
                                    row.get(
                                        "joint_name",
                                        row.get("child_frame_id", source),
                                    ),
                                ),
                                "std_xy_m": spec.get("std_xy_m", 2.0),
                                "std_z_m": spec.get("std_z_m", 5.0),
                                "confidence": spec.get("confidence", 1.0),
                                "class_name": spec.get("class_name", "uav"),
                            }
                        )
                        candidate_rows.append(row)
                    if candidate_rows:
                        candidate_frames.append(
                            CandidateFrame(pd.DataFrame.from_records(candidate_rows))
                        )
                    rows = len(candidate_rows)
                elif kind in {
                    "pose_truth",
                    "odometry_truth",
                    "point_truth",
                    "transform_truth",
                    "tf_truth",
                    "path_truth",
                    "pose_array_truth",
                }:
                    rows_for_message = position_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        child_frame_id=spec.get("child_frame_id"),
                        frame_id=spec.get("frame_id"),
                    )
                    truth_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif kind in {
                    "pose_candidate",
                    "odometry_candidate",
                    "point_candidate",
                    "transform_candidate",
                    "tf_candidate",
                    "path_candidate",
                    "pose_array_candidate",
                }:
                    rows_for_message = position_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        child_frame_id=spec.get("child_frame_id"),
                        frame_id=spec.get("frame_id"),
                    )
                    candidate_rows = []
                    for row in rows_for_message:
                        row.update(
                            {
                                "source": source,
                                "track_id": spec.get(
                                    "track_id",
                                    row.get("child_frame_id", source),
                                ),
                                "std_xy_m": spec.get("std_xy_m", 2.0),
                                "std_z_m": spec.get("std_z_m", 5.0),
                                "confidence": spec.get("confidence", 1.0),
                                "class_name": spec.get("class_name", "uav"),
                            }
                        )
                        candidate_rows.append(row)
                    if candidate_rows:
                        candidate_frames.append(
                            CandidateFrame(pd.DataFrame.from_records(candidate_rows))
                        )
                    rows = len(candidate_rows)
                else:
                    extracted.append({"topic": connection.topic, "kind": kind, "status": "unsupported"})
                    continue
            except Exception as exc:  # pragma: no cover - data-dependent failure details
                extracted.append(
                    {
                        "topic": connection.topic,
                        "kind": kind,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                continue
            extracted.append(
                {
                    "topic": connection.topic,
                    "kind": kind,
                    "status": "extracted",
                    "time_s": time_s,
                    "rows": int(rows),
                }
            )

    candidates = merge_candidate_frames(candidate_frames) if candidate_frames else None
    truth = TruthFrame(normalize_truth_columns(pd.DataFrame.from_records(truth_rows))) if truth_rows else None
    manifest = {
        "schema": "raft-uav-mmuad-native-ros-extraction-v1",
        "bag_path": str(bag_path),
        "topic_map_json": str(topic_map_json),
        "candidate_rows": int(len(candidates.rows)) if candidates is not None else 0,
        "truth_rows": int(len(truth.rows)) if truth is not None else 0,
        "extracted_messages": extracted,
    }
    if output is not None:
        if candidates is not None:
            candidates.rows.to_csv(output / "native_ros_candidates.csv", index=False)
        if truth is not None:
            truth.rows.to_csv(output / "native_ros_truth.csv", index=False)
        (output / "native_ros_extraction_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return NativeRosExtraction(candidates=candidates, truth=truth, manifest=manifest)


def _message_time_s(message: Any, fallback_timestamp_ns: int) -> float:
    stamp_time_s = _message_stamp_time_s(message)
    if stamp_time_s is not None:
        return stamp_time_s
    return float(fallback_timestamp_ns) * 1.0e-9


def _message_stamp_time_s(message: Any) -> float | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        sec = getattr(stamp, "sec", getattr(stamp, "secs", None))
        nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
        if sec is not None:
            return float(sec) + float(nanosec) * 1.0e-9
    return None


def position_message_to_row(message: Any, *, sequence_id: str, time_s: float) -> dict[str, Any]:
    """Convert common position-bearing ROS messages into a normalized row."""

    xyz = _message_position_xyz(message)
    if xyz is None:
        raise ValueError("position-like message has no position/point/translation")
    return {
        "sequence_id": sequence_id,
        "time_s": float(time_s),
        "x_m": xyz[0],
        "y_m": xyz[1],
        "z_m": xyz[2],
    }


def geodetic_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    projector: LocalENUProjector,
    frame_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert NavSatFix/GeoPoint/GeoPose messages into local ENU rows."""

    if not _frame_filter_matches(
        message,
        child_frame_id=None,
        frame_id=frame_id,
    ):
        return []
    point = _geodetic_point_from_message(message)
    if point is None:
        return []
    try:
        latitude = float(getattr(point, "latitude"))
        longitude = float(getattr(point, "longitude"))
        altitude = float(getattr(point, "altitude"))
    except (TypeError, ValueError, AttributeError):
        return []
    if not all(pd.notna(value) for value in (latitude, longitude, altitude)):
        return []
    enu = projector.transform(latitude, longitude, altitude)
    stamp_time_s = _message_stamp_time_s(message)
    row = {
        "sequence_id": sequence_id,
        "time_s": stamp_time_s if stamp_time_s is not None else float(time_s),
        "x_m": float(enu[0]),
        "y_m": float(enu[1]),
        "z_m": float(enu[2]),
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "altitude_m": altitude,
    }
    _add_frame_metadata(row, message)
    _add_navsat_covariance_metadata(row, message)
    return [row]


def detection3d_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    frame_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert vision_msgs Detection3D/Detection3DArray messages into rows."""

    detections = getattr(message, "detections", None)
    if detections is None:
        detections = [message]
    parent_time_s = _message_stamp_time_s(message)
    parent_frame_id = _message_frame_id(message)
    rows: list[dict[str, Any]] = []
    for detection in detections:
        if not _frame_filter_matches(
            detection,
            child_frame_id=None,
            frame_id=frame_id,
            fallback_frame_id=parent_frame_id,
        ):
            continue
        position = _detection3d_position(detection)
        if position is None:
            continue
        detection_time_s = _message_stamp_time_s(detection)
        row = {
            "sequence_id": sequence_id,
            "time_s": (
                detection_time_s
                if detection_time_s is not None
                else parent_time_s
                if parent_time_s is not None
                else float(time_s)
            ),
            "x_m": float(getattr(position, "x")),
            "y_m": float(getattr(position, "y")),
            "z_m": float(getattr(position, "z")),
        }
        _add_frame_metadata(row, detection, fallback_frame_id=parent_frame_id)
        detection_id = getattr(detection, "id", None)
        if detection_id not in (None, ""):
            row["detection_id"] = str(detection_id)
        confidence = _detection3d_confidence(detection)
        if confidence is not None:
            row["confidence"] = float(confidence)
        class_name = _detection3d_class_name(detection)
        if class_name is not None:
            row["class_name"] = class_name
        rows.append(row)
    return rows


def marker_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    frame_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert visualization_msgs Marker/MarkerArray messages into rows."""

    markers = getattr(message, "markers", None)
    if markers is None:
        markers = [message]
    parent_time_s = _message_stamp_time_s(message)
    parent_frame_id = _message_frame_id(message)
    rows: list[dict[str, Any]] = []
    for marker in markers:
        if _marker_action_is_delete(marker):
            continue
        if not _frame_filter_matches(
            marker,
            child_frame_id=None,
            frame_id=frame_id,
            fallback_frame_id=parent_frame_id,
        ):
            continue
        xyz = _marker_position_xyz(marker)
        if xyz is None:
            continue
        marker_time_s = _message_stamp_time_s(marker)
        row = {
            "sequence_id": sequence_id,
            "time_s": (
                marker_time_s
                if marker_time_s is not None
                else parent_time_s
                if parent_time_s is not None
                else float(time_s)
            ),
            "x_m": xyz[0],
            "y_m": xyz[1],
            "z_m": xyz[2],
        }
        _add_frame_metadata(row, marker, fallback_frame_id=parent_frame_id)
        _add_marker_metadata(row, marker)
        rows.append(row)
    return rows


def multidof_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    frame_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert MultiDOFJointState/Trajectory messages into rows."""

    parent_time_s = _message_stamp_time_s(message)
    parent_frame_id = _message_frame_id(message)
    joint_names = _sequence_attr(message, "joint_names")
    trajectory_points = getattr(message, "points", None)
    if trajectory_points is not None:
        rows: list[dict[str, Any]] = []
        for point_index, point in enumerate(trajectory_points):
            point_time_s = _multidof_point_time_s(
                point,
                parent_time_s=parent_time_s,
                fallback_time_s=time_s,
            )
            rows.extend(
                _multidof_transforms_to_rows(
                    getattr(point, "transforms", None),
                    sequence_id=sequence_id,
                    time_s=point_time_s,
                    frame_id=frame_id,
                    fallback_frame_id=parent_frame_id,
                    joint_names=joint_names,
                    point_index=point_index,
                )
            )
        return rows
    return _multidof_transforms_to_rows(
        getattr(message, "transforms", None),
        sequence_id=sequence_id,
        time_s=parent_time_s if parent_time_s is not None else time_s,
        frame_id=frame_id,
        fallback_frame_id=parent_frame_id,
        joint_names=joint_names,
        point_index=None,
    )


def position_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    child_frame_id: str | None = None,
    frame_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a position-bearing message or TFMessage into normalized rows."""

    transforms = getattr(message, "transforms", None)
    if transforms is not None:
        rows: list[dict[str, Any]] = []
        for transform in transforms:
            if not _frame_filter_matches(
                transform,
                child_frame_id=child_frame_id,
                frame_id=frame_id,
            ):
                continue
            transform_time_s = _message_stamp_time_s(transform)
            row = position_message_to_row(
                transform,
                sequence_id=sequence_id,
                time_s=transform_time_s if transform_time_s is not None else time_s,
            )
            _add_frame_metadata(row, transform)
            rows.append(row)
        return rows
    poses = getattr(message, "poses", None)
    if poses is not None:
        rows = []
        message_time_s = _message_stamp_time_s(message)
        message_frame_id = _message_frame_id(message)
        for pose in poses:
            if not _frame_filter_matches(
                pose,
                child_frame_id=child_frame_id,
                frame_id=frame_id,
                fallback_frame_id=message_frame_id,
            ):
                continue
            pose_time_s = _message_stamp_time_s(pose)
            row = position_message_to_row(
                pose,
                sequence_id=sequence_id,
                time_s=(
                    pose_time_s
                    if pose_time_s is not None
                    else message_time_s
                    if message_time_s is not None
                    else time_s
                ),
            )
            _add_frame_metadata(row, pose, fallback_frame_id=message_frame_id)
            rows.append(row)
        return rows
    if not _frame_filter_matches(
        message,
        child_frame_id=child_frame_id,
        frame_id=frame_id,
    ):
        return []
    row = position_message_to_row(message, sequence_id=sequence_id, time_s=time_s)
    _add_frame_metadata(row, message)
    return [row]


def _frame_filter_matches(
    message: Any,
    *,
    child_frame_id: str | None,
    frame_id: str | None,
    fallback_frame_id: Any | None = None,
) -> bool:
    message_child = _message_child_frame_id(message)
    message_frame = _message_frame_id(message)
    if message_frame is None:
        message_frame = fallback_frame_id
    if child_frame_id is not None and str(message_child) != str(child_frame_id):
        return False
    if frame_id is not None and str(message_frame) != str(frame_id):
        return False
    return True


def _add_frame_metadata(
    row: dict[str, Any],
    message: Any,
    *,
    fallback_frame_id: Any | None = None,
) -> None:
    child_frame = _message_child_frame_id(message)
    frame = _message_frame_id(message)
    if frame is None:
        frame = fallback_frame_id
    if child_frame is not None:
        row["child_frame_id"] = str(child_frame)
    if frame is not None:
        row["frame_id"] = str(frame)


def _message_child_frame_id(message: Any) -> Any | None:
    return getattr(message, "child_frame_id", None)


def _message_frame_id(message: Any) -> Any | None:
    header = getattr(message, "header", None)
    return getattr(header, "frame_id", None)


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
            "geodetic native ROS topics require enu_origin_lla or "
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


def _sequence_attr(message: Any, name: str) -> list[Any]:
    value = getattr(message, name, None)
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _multidof_transforms_to_rows(
    transforms: Any,
    *,
    sequence_id: str,
    time_s: float,
    frame_id: str | None,
    fallback_frame_id: Any | None,
    joint_names: list[Any],
    point_index: int | None,
) -> list[dict[str, Any]]:
    if transforms is None:
        return []
    rows: list[dict[str, Any]] = []
    for transform_index, transform in enumerate(transforms):
        if not _frame_filter_matches(
            transform,
            child_frame_id=None,
            frame_id=frame_id,
            fallback_frame_id=fallback_frame_id,
        ):
            continue
        row = position_message_to_row(
            transform,
            sequence_id=sequence_id,
            time_s=time_s,
        )
        _add_frame_metadata(row, transform, fallback_frame_id=fallback_frame_id)
        _add_multidof_metadata(
            row,
            transform_index=transform_index,
            joint_names=joint_names,
            point_index=point_index,
        )
        rows.append(row)
    return rows


def _add_multidof_metadata(
    row: dict[str, Any],
    *,
    transform_index: int,
    joint_names: list[Any],
    point_index: int | None,
) -> None:
    row["multidof_transform_index"] = int(transform_index)
    if point_index is not None:
        row["multidof_point_index"] = int(point_index)
    if transform_index < len(joint_names) and joint_names[transform_index] not in (None, ""):
        row["joint_name"] = str(joint_names[transform_index])


def _multidof_point_time_s(
    point: Any,
    *,
    parent_time_s: float | None,
    fallback_time_s: float,
) -> float:
    point_stamp = _message_stamp_time_s(point)
    if point_stamp is not None:
        return point_stamp
    duration = _duration_time_s(getattr(point, "time_from_start", None))
    base = parent_time_s if parent_time_s is not None else float(fallback_time_s)
    if duration is None:
        return base
    return base + duration


def _duration_time_s(duration: Any | None) -> float | None:
    if duration is None:
        return None
    sec = getattr(duration, "sec", getattr(duration, "secs", None))
    nanosec = getattr(duration, "nanosec", getattr(duration, "nsecs", 0))
    if sec is None:
        return None
    return float(sec) + float(nanosec) * 1.0e-9


def _geodetic_point_from_message(message: Any) -> Any | None:
    if hasattr(message, "latitude") and hasattr(message, "longitude"):
        return message
    position = getattr(message, "position", None)
    if position is not None:
        point = _geodetic_point_from_message(position)
        if point is not None:
            return point
    pose = getattr(message, "pose", None)
    if pose is not None:
        return _geodetic_point_from_message(pose)
    return None


def _add_navsat_covariance_metadata(row: dict[str, Any], message: Any) -> None:
    covariance = getattr(message, "position_covariance", None)
    if covariance is None:
        return
    try:
        values = [float(value) for value in covariance]
    except (TypeError, ValueError):
        return
    if len(values) < 9:
        return
    xy_variance = max(values[0], values[4])
    z_variance = values[8]
    if xy_variance >= 0.0:
        row["std_xy_m"] = float(xy_variance) ** 0.5
    if z_variance >= 0.0:
        row["std_z_m"] = float(z_variance) ** 0.5
    covariance_type = getattr(message, "position_covariance_type", None)
    if covariance_type not in (None, ""):
        row["navsat_covariance_type"] = str(covariance_type)


def _position_from_message(message: Any) -> Any | None:
    point = getattr(message, "point", None)
    if point is not None:
        return point
    transform = getattr(message, "transform", None)
    if transform is not None:
        translation = getattr(transform, "translation", None)
        if translation is not None:
            return translation
    translation = getattr(message, "translation", None)
    if translation is not None:
        return translation
    pose = getattr(message, "pose", message)
    if hasattr(pose, "pose"):
        pose = pose.pose
    return getattr(pose, "position", None)


def _message_position_xyz(message: Any) -> tuple[float, float, float] | None:
    position = _position_from_message(message)
    return _xyz_from_position(position)


def _xyz_from_position(position: Any | None) -> tuple[float, float, float] | None:
    if position is None:
        return None
    try:
        return (
            float(getattr(position, "x")),
            float(getattr(position, "y")),
            float(getattr(position, "z")),
        )
    except (TypeError, ValueError, AttributeError):
        return None


def _detection3d_position(detection: Any) -> Any | None:
    bbox = getattr(detection, "bbox", None)
    center = getattr(bbox, "center", None)
    if center is None:
        return None
    return _position_from_message(center)


def _marker_position_xyz(marker: Any) -> tuple[float, float, float] | None:
    xyz = _message_position_xyz(marker)
    if xyz is not None:
        return xyz
    points = getattr(marker, "points", None)
    if not points:
        return None
    point_rows = [_xyz_from_position(point) for point in points]
    valid = [point for point in point_rows if point is not None]
    if not valid:
        return None
    count = float(len(valid))
    return (
        sum(point[0] for point in valid) / count,
        sum(point[1] for point in valid) / count,
        sum(point[2] for point in valid) / count,
    )


def _marker_action_is_delete(marker: Any) -> bool:
    action = getattr(marker, "action", None)
    if action is None:
        return False
    if isinstance(action, str):
        return action.strip().lower() in {"delete", "deleteall", "delete_all"}
    try:
        return int(action) in {2, 3}
    except (TypeError, ValueError):
        return False


def _add_marker_metadata(row: dict[str, Any], marker: Any) -> None:
    marker_id = getattr(marker, "id", None)
    namespace = getattr(marker, "ns", None)
    if marker_id not in (None, ""):
        row["marker_id"] = str(marker_id)
    if namespace not in (None, ""):
        row["marker_namespace"] = str(namespace)
    track_id = _marker_track_id(marker_id=marker_id, namespace=namespace)
    if track_id is not None:
        row["marker_track_id"] = track_id
    marker_type = getattr(marker, "type", None)
    if marker_type not in (None, ""):
        row["marker_type"] = str(marker_type)
    action = getattr(marker, "action", None)
    if action not in (None, ""):
        row["marker_action"] = str(action)
    text = getattr(marker, "text", None)
    if text not in (None, ""):
        row["class_name"] = str(text)


def _marker_track_id(*, marker_id: Any | None, namespace: Any | None) -> str | None:
    has_marker_id = marker_id not in (None, "")
    has_namespace = namespace not in (None, "")
    if has_marker_id and has_namespace:
        return f"{namespace}:{marker_id}"
    if has_marker_id:
        return str(marker_id)
    if has_namespace:
        return str(namespace)
    return None


def _detection3d_confidence(detection: Any) -> float | None:
    result = _first_detection_result(detection)
    if result is None:
        return None
    score = getattr(result, "score", None)
    if score is not None:
        return float(score)
    hypothesis = getattr(result, "hypothesis", None)
    if hypothesis is not None and getattr(hypothesis, "score", None) is not None:
        return float(hypothesis.score)
    return None


def _detection3d_class_name(detection: Any) -> str | None:
    result = _first_detection_result(detection)
    if result is None:
        return None
    class_id = getattr(result, "class_id", None)
    if class_id not in (None, ""):
        return str(class_id)
    hypothesis = getattr(result, "hypothesis", None)
    if hypothesis is not None:
        hypothesis_id = getattr(hypothesis, "class_id", getattr(hypothesis, "id", None))
        if hypothesis_id not in (None, ""):
            return str(hypothesis_id)
    result_id = getattr(result, "id", None)
    if result_id not in (None, ""):
        return str(result_id)
    return None


def _first_detection_result(detection: Any) -> Any | None:
    results = getattr(detection, "results", None)
    if not results:
        return None
    return results[0]
