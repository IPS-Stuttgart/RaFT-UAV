"""Optional native ROS bag extraction toward full MMUAD support.

This module is intentionally optional-dependency safe.  It uses the ``rosbags``
package when available, but the rest of the MMUAD adapter imports without ROS.
The extractor currently supports common message families that appear in UAV
tracking logs:

* ``sensor_msgs/msg/PointCloud2`` -> clustered candidate detections;
* ``geometry_msgs/msg/PoseStamped`` -> truth rows or candidate rows;
* ``geometry_msgs/msg/PointStamped`` -> truth rows or candidate rows;
* ``geometry_msgs/msg/TransformStamped`` -> truth rows or candidate rows;
* ``tf2_msgs/msg/TFMessage`` -> transform truth rows or candidate rows;
* ``nav_msgs/msg/Path`` -> trajectory truth rows or candidate rows;
* ``nav_msgs/msg/Odometry`` -> truth rows or candidate rows.

Unknown topics are recorded in the extraction manifest and skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import pandas as pd

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
    ``pose_truth`` / ``odometry_truth``
        Convert pose/odometry messages into truth rows.
    ``point_truth`` / ``transform_truth`` / ``tf_truth`` / ``path_truth``
        Convert position-only messages into truth rows.
    ``pose_candidate`` / ``odometry_candidate`` / ``point_candidate`` /
    ``transform_candidate`` / ``tf_candidate`` / ``path_candidate``
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
                elif kind in {
                    "pose_truth",
                    "odometry_truth",
                    "point_truth",
                    "transform_truth",
                    "tf_truth",
                    "path_truth",
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

    position = _position_from_message(message)
    if position is None:
        raise ValueError("position-like message has no position/point/translation")
    return {
        "sequence_id": sequence_id,
        "time_s": float(time_s),
        "x_m": float(getattr(position, "x")),
        "y_m": float(getattr(position, "y")),
        "z_m": float(getattr(position, "z")),
    }


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
        for pose in poses:
            if not _frame_filter_matches(
                pose,
                child_frame_id=child_frame_id,
                frame_id=frame_id,
            ):
                continue
            pose_time_s = _message_stamp_time_s(pose)
            row = position_message_to_row(
                pose,
                sequence_id=sequence_id,
                time_s=pose_time_s if pose_time_s is not None else time_s,
            )
            _add_frame_metadata(row, pose)
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
) -> bool:
    message_child = _message_child_frame_id(message)
    message_frame = _message_frame_id(message)
    if child_frame_id is not None and str(message_child) != str(child_frame_id):
        return False
    if frame_id is not None and str(message_frame) != str(frame_id):
        return False
    return True


def _add_frame_metadata(row: dict[str, Any], message: Any) -> None:
    child_frame = _message_child_frame_id(message)
    frame = _message_frame_id(message)
    if child_frame is not None:
        row["child_frame_id"] = str(child_frame)
    if frame is not None:
        row["frame_id"] = str(frame)


def _message_child_frame_id(message: Any) -> Any | None:
    return getattr(message, "child_frame_id", None)


def _message_frame_id(message: Any) -> Any | None:
    header = getattr(message, "header", None)
    return getattr(header, "frame_id", None)


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
