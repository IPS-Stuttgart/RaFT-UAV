"""Optional native ROS bag extraction toward full MMUAD support.

This module is intentionally optional-dependency safe.  It uses the ``rosbags``
package when available, but the rest of the MMUAD adapter imports without ROS.
The extractor currently supports common message families that appear in UAV
tracking logs:

* ``sensor_msgs/msg/PointCloud2`` -> clustered candidate detections;
* ``geometry_msgs/msg/PoseStamped`` -> truth rows or candidate rows;
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
    ``pose_candidate`` / ``odometry_candidate``
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
            kind = str(spec.get("kind", "candidate"))
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
                elif kind in {"pose_truth", "odometry_truth"}:
                    truth_rows.append(_pose_like_row(message, sequence_id=str(spec.get("sequence_id", sequence_id)), time_s=time_s))
                    rows = 1
                elif kind in {"pose_candidate", "odometry_candidate"}:
                    row = _pose_like_row(message, sequence_id=str(spec.get("sequence_id", sequence_id)), time_s=time_s)
                    row.update(
                        {
                            "source": source,
                            "track_id": spec.get("track_id", source),
                            "std_xy_m": spec.get("std_xy_m", 2.0),
                            "std_z_m": spec.get("std_z_m", 5.0),
                            "confidence": spec.get("confidence", 1.0),
                            "class_name": spec.get("class_name", "uav"),
                        }
                    )
                    candidate_frames.append(CandidateFrame(pd.DataFrame.from_records([row])))
                    rows = 1
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
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        sec = getattr(stamp, "sec", getattr(stamp, "secs", None))
        nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
        if sec is not None:
            return float(sec) + float(nanosec) * 1.0e-9
    return float(fallback_timestamp_ns) * 1.0e-9


def _pose_like_row(message: Any, *, sequence_id: str, time_s: float) -> dict[str, Any]:
    pose = getattr(message, "pose", message)
    if hasattr(pose, "pose"):
        pose = pose.pose
    position = getattr(pose, "position", None)
    if position is None:
        raise ValueError("pose-like message has no position")
    return {
        "sequence_id": sequence_id,
        "time_s": float(time_s),
        "x_m": float(getattr(position, "x")),
        "y_m": float(getattr(position, "y")),
        "z_m": float(getattr(position, "z")),
    }
