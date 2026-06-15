"""Optional native ROS bag extraction toward full MMUAD support.

This module is intentionally optional-dependency safe.  It uses the ``rosbags``
package when available, but the rest of the MMUAD adapter imports without ROS.
The extractor currently supports common message families that appear in UAV
tracking logs:

* ``sensor_msgs/msg/PointCloud2`` / ``PointCloud`` -> clustered candidate
  detections;
* ``livox_ros_driver(2)/msg/CustomMsg`` -> clustered candidate detections;
* ``sensor_msgs/msg/CameraInfo`` -> camera intrinsics for native Detection2D;
* ``sensor_msgs/msg/Image`` / ``CompressedImage`` -> timestamp template rows;
* common ROS audio messages -> timestamp inventory rows;
* ``sensor_msgs/msg/Imu`` -> timestamp/kinematics inventory rows;
* ``geometry_msgs/msg/Twist`` / ``Accel`` -> velocity/acceleration inventory
  rows;
* ``sensor_msgs/msg/LaserScan`` -> range-scan candidate detections;
* common polar/range-azimuth radar messages -> polar radar candidates;
* ``sensor_msgs/msg/NavSatFix`` -> geodetic rows projected into local ENU;
* ``geographic_msgs/msg/GeoPointStamped`` / ``GeoPoseStamped`` -> local ENU rows;
* ``vision_msgs/msg/Detection2D`` / ``Detection2DArray`` -> calibrated camera
  detection candidates;
* ``vision_msgs/msg/Detection3D`` / ``Detection3DArray`` -> bbox center rows;
* ``vision_msgs/msg/BoundingBox3D`` / ``BoundingBox3DArray`` -> bbox center rows;
* common tracked/detected object arrays -> object pose rows;
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

Unsupported and missing topic-map topics are recorded in the extraction
manifest and skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import json

import pandas as pd

from raft_uav.coordinates import LocalENUProjector
from raft_uav.mmuad.calibration import _transform_from_entry
from raft_uav.mmuad.camera import (
    CameraIntrinsics,
    CameraModel,
    camera_detection_frame_to_candidates,
    load_camera_models_from_files,
)
from raft_uav.mmuad.io import merge_candidate_frames, point_rows_to_candidates
from raft_uav.mmuad.pointcloud2 import pointcloud2_to_candidates
from raft_uav.mmuad.radar import radar_polar_frame_to_candidates
from raft_uav.mmuad.rosbag_bridge import load_topic_map_payload
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame, normalize_truth_columns


@dataclass(frozen=True)
class NativeRosExtraction:
    """Result of optional native ROS extraction."""

    candidates: CandidateFrame | None
    truth: TruthFrame | None
    manifest: dict[str, Any]
    image_timestamps: pd.DataFrame | None = None
    audio_timestamps: pd.DataFrame | None = None
    imu_timestamps: pd.DataFrame | None = None
    kinematic_timestamps: pd.DataFrame | None = None


def extract_native_rosbag_topic_map(
    *,
    bag_path: Path,
    topic_map_json: Path,
    output_dir: Path | None = None,
    voxel_size_m: float = 0.75,
    min_points: int = 3,
) -> NativeRosExtraction:
    """Extract supported ROS topics according to a topic-map JSON/YAML file.

    The topic map follows the same high-level structure as the CSV export bridge
    but may use native kinds:

    ``pointcloud2_candidate``
        Decode ``sensor_msgs/msg/PointCloud2`` and cluster points.
    ``pointcloud_candidate`` / ``legacy_pointcloud_candidate``
        Decode legacy ``sensor_msgs/msg/PointCloud`` point arrays and cluster
        them through the same point-row bridge as other point clouds.
    ``livox_custom_candidate`` / ``livox_custommsg_candidate``
        Decode common Livox CustomMsg point arrays and cluster them through the
        same point-row bridge as exported point-cloud files.
    ``radar_polar_candidate`` / ``polar_radar_candidate``
        Decode common range/azimuth message shapes and convert them through
        the same polar radar bridge as exported table rows.  Native ROS angles
        default to radians unless ``angle_unit`` is set in the topic map.
    ``laserscan_candidate`` / ``laser_scan_candidate``
        Decode ``sensor_msgs/msg/LaserScan`` ranges into polar candidate rows.
        LaserScan angles follow the ROS convention of zero forward on +X and
        positive counterclockwise/left by default.  Set
        ``cluster_adjacent_ranges`` to combine contiguous returns into
        centroid candidates.
    ``camera_info`` / ``camera_info_calibration``
        Decode ``sensor_msgs/msg/CameraInfo`` intrinsics for native
        Detection2D back-projection.  Detection2D topics with the same source
        can then omit ``camera_calibration_file`` sidecars.
    ``image_timestamps`` / ``image_timestamp_template``
        Extract native ``sensor_msgs/msg/Image`` or ``CompressedImage`` frame
        timestamps into CSV/template artifacts. This is timestamp inventory
        only; image object detection remains external.
    ``audio_timestamps`` / ``audio_timestamp_inventory``
        Extract native audio message timestamps into CSV inventory artifacts.
        This is timestamp/sample metadata only; acoustic detections remain
        external candidate exports.
    ``imu_timestamps`` / ``imu_timestamp_inventory``
        Extract native ``sensor_msgs/msg/Imu`` timestamps and kinematics into
        CSV inventory artifacts. This is raw sensor metadata only.
    ``kinematic_timestamps`` / ``twist_timestamps`` / ``accel_timestamps``
        Extract native ``geometry_msgs/msg/Twist``/``TwistStamped`` and
        ``geometry_msgs/msg/Accel``/``AccelStamped`` vectors into CSV
        inventory artifacts. This is velocity/acceleration metadata only.
    ``navsatfix_truth`` / ``geopoint_truth`` / ``geopose_truth`` /
    ``navsatfix_candidate`` / ``geopoint_candidate`` / ``geopose_candidate``
        Project geodetic GPS/geographic positions into local ENU rows. These
        entries require ``enu_origin_lla`` or separate origin latitude,
        longitude, and altitude fields in the topic map.
    ``detection3d_truth`` / ``detection3d_array_truth`` /
    ``detection3d_candidate`` / ``detection3d_array_candidate``
        Convert vision_msgs 3D detection bbox centers into truth/candidate rows.
    ``bounding_box3d_truth`` / ``bounding_box3d_array_truth`` /
    ``bounding_box3d_candidate`` / ``bounding_box3d_array_candidate``
        Convert vision_msgs 3D bounding-box centers into truth/candidate rows.
    ``tracked_objects_truth`` / ``tracked_objects_candidate``
        Convert common perception/tracker object arrays with ``objects``,
        ``tracks``, ``detections``, or ``targets`` children into pose rows.
    ``camera_detections_candidate`` / ``detection2d_candidate`` /
    ``detection2d_array_candidate``
        Convert vision_msgs 2D detections into calibrated camera candidates.
        These entries require camera calibration and either per-detection depth
        or ``camera_fixed_depth_m`` / ``fixed_depth_m`` in the topic map.
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

    payload = load_topic_map_payload(topic_map_json)
    specs = list(payload.get("exports", []))
    if not specs:
        raise ValueError(f"topic map {topic_map_json} has no exports")
    try:
        from rosbags.highlevel import AnyReader  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised only without dependency
        raise RuntimeError(
            "native ROS extraction requires the optional 'rosbags' package; "
            "install it or export topics to CSV and use --topic-map-file/--topic-map-json"
        ) from exc

    by_topic = {str(spec["topic"]): spec for spec in specs if "topic" in spec}
    candidate_frames: list[CandidateFrame] = []
    truth_rows: list[dict[str, Any]] = []
    image_timestamp_rows: list[dict[str, Any]] = []
    audio_timestamp_rows: list[dict[str, Any]] = []
    imu_timestamp_rows: list[dict[str, Any]] = []
    kinematic_timestamp_rows: list[dict[str, Any]] = []
    extracted: list[dict[str, Any]] = []
    sequence_id = str(payload.get("sequence_id", Path(bag_path).stem))
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    with AnyReader([Path(bag_path)]) as reader:
        reader_topics = {str(connection.topic) for connection in reader.connections}
        for topic, spec in by_topic.items():
            if topic in reader_topics:
                continue
            extracted.append(
                {
                    "topic": topic,
                    "kind": str(spec.get("kind", "candidate")).strip().lower(),
                    "status": "missing_topic",
                }
            )
        topic_connections = [
            connection for connection in reader.connections if connection.topic in by_topic
        ]
        native_camera_models, camera_info_messages = _camera_models_from_camera_info_topics(
            reader,
            topic_connections=topic_connections,
            by_topic=by_topic,
        )
        extracted.extend(camera_info_messages)
        replay_connections = [
            connection
            for connection in topic_connections
            if not _is_camera_info_kind(by_topic[connection.topic])
        ]
        replay_message_counts = {str(connection.topic): 0 for connection in replay_connections}
        for connection, timestamp_ns, rawdata in reader.messages(connections=replay_connections):
            replay_message_counts[str(connection.topic)] = (
                replay_message_counts.get(str(connection.topic), 0) + 1
            )
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
                elif _is_pointcloud_candidate_kind(kind):
                    point_rows = pointcloud_message_to_points(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                    )
                    if point_rows:
                        frame = point_rows_to_candidates(
                            pd.DataFrame.from_records(point_rows),
                            source=source,
                            voxel_size_m=float(
                                spec.get("voxel_size_m", spec.get("voxel_size", voxel_size_m))
                            ),
                            min_points=int(
                                spec.get(
                                    "min_cluster_points",
                                    spec.get("min_points", min_points),
                                )
                            ),
                            min_confidence=float(spec.get("min_confidence", 0.0)),
                        )
                        candidate_frames.append(frame)
                        rows = len(frame.rows)
                    else:
                        rows = 0
                elif kind in {
                    "livox_custom_candidate",
                    "livox_custommsg_candidate",
                    "livox_custom_pointcloud_candidate",
                    "livox_pointcloud_candidate",
                }:
                    point_rows = livox_custom_message_to_points(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                    )
                    if point_rows:
                        frame = point_rows_to_candidates(
                            pd.DataFrame.from_records(point_rows),
                            source=source,
                            voxel_size_m=float(
                                spec.get("voxel_size_m", spec.get("voxel_size", voxel_size_m))
                            ),
                            min_points=int(
                                spec.get(
                                    "min_cluster_points",
                                    spec.get("min_points", min_points),
                                )
                            ),
                            min_confidence=float(
                                spec.get("min_confidence", 0.0)
                            ),
                        )
                        candidate_frames.append(frame)
                        rows = len(frame.rows)
                    else:
                        rows = 0
                elif _is_laserscan_candidate_kind(kind):
                    angle_unit = str(
                        spec.get(
                            "angle_unit",
                            spec.get("laserscan_angle_unit", "rad"),
                        )
                    )
                    rows_for_message = laserscan_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        angle_unit=angle_unit,
                        cluster_adjacent=_spec_bool(
                            spec,
                            "cluster_adjacent_ranges",
                            "laserscan_cluster_adjacent_ranges",
                            "cluster_adjacent",
                        ),
                        min_cluster_points=_spec_int(
                            spec,
                            "min_cluster_points",
                            "laserscan_min_cluster_points",
                            default=1,
                        ),
                        max_cluster_range_gap_m=_spec_float(
                            spec,
                            "max_cluster_range_gap_m",
                            "laserscan_max_cluster_range_gap_m",
                            default=1.0,
                        ),
                    )
                    if rows_for_message:
                        frame = radar_polar_frame_to_candidates(
                            pd.DataFrame.from_records(rows_for_message),
                            source=source,
                            sequence_id=str(spec.get("sequence_id", sequence_id)),
                            azimuth_convention=str(
                                spec.get(
                                    "azimuth_convention",
                                    spec.get(
                                        "laserscan_azimuth_convention",
                                        "x-forward-left-positive",
                                    ),
                                )
                            ),
                            angle_unit=angle_unit,
                            range_std_m=float(
                                spec.get(
                                    "range_std_m",
                                    spec.get("laserscan_range_std_m", 1.0),
                                )
                            ),
                            angle_std_deg=float(
                                spec.get(
                                    "angle_std_deg",
                                    spec.get("laserscan_angle_std_deg", 0.5),
                                )
                            ),
                            z_std_m=float(
                                spec.get("z_std_m", spec.get("laserscan_z_std_m", 2.0))
                            ),
                        )
                        candidate_frames.append(frame)
                        rows = len(frame.rows)
                    else:
                        rows = 0
                elif kind in {
                    "radar_polar",
                    "radar_polar_candidate",
                    "polar_radar",
                    "polar_radar_candidate",
                }:
                    angle_unit = str(
                        spec.get(
                            "angle_unit",
                            spec.get("radar_polar_angle_unit", "rad"),
                        )
                    )
                    rows_for_message = radar_polar_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        angle_unit=angle_unit,
                    )
                    if rows_for_message:
                        frame = radar_polar_frame_to_candidates(
                            pd.DataFrame.from_records(rows_for_message),
                            source=source,
                            sequence_id=str(spec.get("sequence_id", sequence_id)),
                            azimuth_convention=str(
                                spec.get(
                                    "azimuth_convention",
                                    spec.get(
                                        "radar_polar_azimuth_convention",
                                        "north-clockwise",
                                    ),
                                )
                            ),
                            angle_unit=angle_unit,
                            range_std_m=float(
                                spec.get(
                                    "range_std_m",
                                    spec.get("radar_polar_range_std_m", 2.0),
                                )
                            ),
                            angle_std_deg=float(
                                spec.get(
                                    "angle_std_deg",
                                    spec.get("radar_polar_angle_std_deg", 2.0),
                                )
                            ),
                            z_std_m=float(
                                spec.get(
                                    "z_std_m",
                                    spec.get("radar_polar_z_std_m", 5.0),
                                )
                            ),
                        )
                        candidate_frames.append(frame)
                        rows = len(frame.rows)
                    else:
                        rows = 0
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
                elif kind in {
                    "bounding_box3d_truth",
                    "bounding_box3d_array_truth",
                    "boundingbox3d_truth",
                    "boundingbox3d_array_truth",
                    "bbox3d_truth",
                    "bbox3d_array_truth",
                }:
                    rows_for_message = bounding_box3d_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        frame_id=spec.get("frame_id"),
                    )
                    truth_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif kind in {
                    "bounding_box3d_candidate",
                    "bounding_box3d_array_candidate",
                    "boundingbox3d_candidate",
                    "boundingbox3d_array_candidate",
                    "bbox3d_candidate",
                    "bbox3d_array_candidate",
                }:
                    rows_for_message = bounding_box3d_message_to_rows(
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
                                        "track_id",
                                        row.get("box_id", row.get("box_index", source)),
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
                    "tracked_object_truth",
                    "tracked_objects_truth",
                    "tracked_object_array_truth",
                    "object_array_truth",
                    "detected_objects_truth",
                }:
                    rows_for_message = tracked_objects_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        frame_id=spec.get("frame_id"),
                    )
                    truth_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif kind in {
                    "tracked_object_candidate",
                    "tracked_objects_candidate",
                    "tracked_object_array_candidate",
                    "object_array_candidate",
                    "detected_objects_candidate",
                }:
                    rows_for_message = tracked_objects_message_to_rows(
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
                                        "track_id",
                                        row.get("object_id", row.get("object_index", source)),
                                    ),
                                ),
                                "std_xy_m": spec.get(
                                    "std_xy_m",
                                    row.get("std_xy_m", 2.0),
                                ),
                                "std_z_m": spec.get(
                                    "std_z_m",
                                    row.get("std_z_m", 5.0),
                                ),
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
                    "camera_detection",
                    "camera_detection_candidate",
                    "camera_detections",
                    "camera_detections_candidate",
                    "detection2d",
                    "detection2d_candidate",
                    "detection2d_array",
                    "detection2d_array_candidate",
                    "image_detection",
                    "image_detection_candidate",
                    "image_detections",
                    "image_detections_candidate",
                }:
                    rows_for_message = detection2d_message_to_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        frame_id=spec.get("frame_id"),
                    )
                    if rows_for_message:
                        camera_models = _camera_models_from_spec(
                            spec,
                            bag_path=bag_path,
                            topic_map_json=topic_map_json,
                            source=source,
                            native_camera_models=native_camera_models,
                        )
                        frame = camera_detection_frame_to_candidates(
                            pd.DataFrame.from_records(rows_for_message),
                            camera_models=camera_models,
                            source=source,
                            sequence_id=str(spec.get("sequence_id", sequence_id)),
                            fixed_depth_m=_optional_float(
                                spec,
                                "camera_fixed_depth_m",
                                "fixed_depth_m",
                                "depth_m",
                            ),
                            std_xy_m=float(
                                spec.get("camera_std_xy_m", spec.get("std_xy_m", 5.0))
                            ),
                            std_z_m=float(
                                spec.get("camera_std_z_m", spec.get("std_z_m", 10.0))
                            ),
                        )
                        candidate_frames.append(frame)
                        rows = len(frame.rows)
                    else:
                        rows = 0
                elif _is_image_timestamp_kind(kind):
                    rows_for_message = image_message_to_timestamp_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        topic=str(connection.topic),
                        source=source,
                        message_index=replay_message_counts[str(connection.topic)] - 1,
                    )
                    image_timestamp_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif _is_audio_timestamp_kind(kind):
                    rows_for_message = audio_message_to_timestamp_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        topic=str(connection.topic),
                        source=source,
                        message_index=replay_message_counts[str(connection.topic)] - 1,
                    )
                    audio_timestamp_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif _is_imu_timestamp_kind(kind):
                    rows_for_message = imu_message_to_timestamp_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        topic=str(connection.topic),
                        source=source,
                        message_index=replay_message_counts[str(connection.topic)] - 1,
                    )
                    imu_timestamp_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
                elif _is_kinematic_timestamp_kind(kind):
                    rows_for_message = kinematic_message_to_timestamp_rows(
                        message,
                        sequence_id=str(spec.get("sequence_id", sequence_id)),
                        time_s=time_s,
                        topic=str(connection.topic),
                        source=source,
                        message_index=replay_message_counts[str(connection.topic)] - 1,
                        kind=kind,
                    )
                    kinematic_timestamp_rows.extend(rows_for_message)
                    rows = len(rows_for_message)
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
        for connection in replay_connections:
            topic = str(connection.topic)
            if replay_message_counts.get(topic, 0) > 0:
                continue
            spec = by_topic[connection.topic]
            extracted.append(
                {
                    "topic": connection.topic,
                    "kind": str(spec.get("kind", "candidate")).strip().lower(),
                    "status": "matched_topic_no_messages",
                    "msgtype": str(getattr(connection, "msgtype", "")),
                }
            )

    candidates = merge_candidate_frames(candidate_frames) if candidate_frames else None
    truth = TruthFrame(normalize_truth_columns(pd.DataFrame.from_records(truth_rows))) if truth_rows else None
    image_timestamps = (
        pd.DataFrame.from_records(image_timestamp_rows) if image_timestamp_rows else None
    )
    audio_timestamps = (
        pd.DataFrame.from_records(audio_timestamp_rows) if audio_timestamp_rows else None
    )
    imu_timestamps = (
        pd.DataFrame.from_records(imu_timestamp_rows) if imu_timestamp_rows else None
    )
    kinematic_timestamps = (
        pd.DataFrame.from_records(kinematic_timestamp_rows)
        if kinematic_timestamp_rows
        else None
    )
    manifest = {
        "schema": "raft-uav-mmuad-native-ros-extraction-v1",
        "bag_path": str(bag_path),
        "topic_map_json": str(topic_map_json),
        "candidate_rows": int(len(candidates.rows)) if candidates is not None else 0,
        "truth_rows": int(len(truth.rows)) if truth is not None else 0,
        "image_timestamp_rows": int(len(image_timestamps)) if image_timestamps is not None else 0,
        "audio_timestamp_rows": int(len(audio_timestamps)) if audio_timestamps is not None else 0,
        "imu_timestamp_rows": int(len(imu_timestamps)) if imu_timestamps is not None else 0,
        "kinematic_timestamp_rows": (
            int(len(kinematic_timestamps)) if kinematic_timestamps is not None else 0
        ),
        "extracted_messages": extracted,
    }
    if output is not None:
        if candidates is not None:
            candidates.rows.to_csv(output / "native_ros_candidates.csv", index=False)
        if truth is not None:
            truth.rows.to_csv(output / "native_ros_truth.csv", index=False)
        if image_timestamps is not None:
            image_timestamps.to_csv(output / "native_ros_image_timestamps.csv", index=False)
            _image_timestamp_template_rows(image_timestamps).to_csv(
                output / "native_ros_image_timestamp_template.csv",
                index=False,
            )
            manifest["image_timestamps_csv"] = str(output / "native_ros_image_timestamps.csv")
            manifest["image_timestamp_template_csv"] = str(
                output / "native_ros_image_timestamp_template.csv"
            )
        if audio_timestamps is not None:
            audio_timestamps.to_csv(output / "native_ros_audio_timestamps.csv", index=False)
            manifest["audio_timestamps_csv"] = str(output / "native_ros_audio_timestamps.csv")
        if imu_timestamps is not None:
            imu_timestamps.to_csv(output / "native_ros_imu_timestamps.csv", index=False)
            manifest["imu_timestamps_csv"] = str(output / "native_ros_imu_timestamps.csv")
        if kinematic_timestamps is not None:
            kinematic_timestamps.to_csv(
                output / "native_ros_kinematic_timestamps.csv",
                index=False,
            )
            manifest["kinematic_timestamps_csv"] = str(
                output / "native_ros_kinematic_timestamps.csv"
            )
        (output / "native_ros_extraction_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return NativeRosExtraction(
        candidates=candidates,
        truth=truth,
        manifest=manifest,
        image_timestamps=image_timestamps,
        audio_timestamps=audio_timestamps,
        imu_timestamps=imu_timestamps,
        kinematic_timestamps=kinematic_timestamps,
    )


def _message_time_s(message: Any, fallback_timestamp_ns: int) -> float:
    stamp_time_s = _message_stamp_time_s(message)
    if stamp_time_s is not None:
        return stamp_time_s
    return float(fallback_timestamp_ns) * 1.0e-9


def _is_image_timestamp_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized in {
        "image_timestamp",
        "image_timestamps",
        "image_frame",
        "image_frames",
        "image_frame_timestamp",
        "image_frame_timestamps",
        "image_timestamp_template",
        "compressed_image_timestamp",
        "compressed_image_timestamps",
    }


def _is_audio_timestamp_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized in {
        "audio_timestamp",
        "audio_timestamps",
        "audio_frame",
        "audio_frames",
        "audio_frame_timestamp",
        "audio_frame_timestamps",
        "audio_timestamp_inventory",
        "audio_sample_timestamp",
        "audio_sample_timestamps",
    }


def _is_imu_timestamp_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized in {
        "imu_timestamp",
        "imu_timestamps",
        "imu_frame",
        "imu_frames",
        "imu_frame_timestamp",
        "imu_frame_timestamps",
        "imu_timestamp_inventory",
    }


def _is_kinematic_timestamp_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized in {
        "kinematic_timestamp",
        "kinematic_timestamps",
        "kinematics_timestamp",
        "kinematics_timestamps",
        "kinematic_timestamp_inventory",
        "twist_timestamp",
        "twist_timestamps",
        "twist_timestamp_inventory",
        "velocity_timestamp",
        "velocity_timestamps",
        "velocity_timestamp_inventory",
        "accel_timestamp",
        "accel_timestamps",
        "accel_timestamp_inventory",
        "acceleration_timestamp",
        "acceleration_timestamps",
        "acceleration_timestamp_inventory",
    }


def _is_laserscan_candidate_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized in {
        "laserscan_candidate",
        "laser_scan_candidate",
        "scan_candidate",
        "range_scan_candidate",
        "range_scan",
    }


def _is_pointcloud_candidate_kind(kind: str) -> bool:
    normalized = str(kind).strip().lower()
    return normalized in {
        "pointcloud_candidate",
        "pointcloud1_candidate",
        "legacy_pointcloud_candidate",
        "sensor_msgs_pointcloud_candidate",
    }


def image_message_to_timestamp_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    topic: str,
    source: str,
    message_index: int,
) -> list[dict[str, Any]]:
    """Convert a native ROS image message into timestamp/template metadata."""

    row: dict[str, Any] = {
        "sequence_id": str(sequence_id),
        "time_s": float(time_s),
        "topic": str(topic),
        "source": str(source),
        "message_index": int(message_index),
    }
    frame_id = _message_frame_id(message)
    if frame_id not in (None, ""):
        row["frame_id"] = str(frame_id)
    for output_key, names in {
        "height": ("height",),
        "width": ("width",),
        "encoding": ("encoding",),
        "format": ("format",),
        "step": ("step",),
        "is_bigendian": ("is_bigendian", "is_big_endian"),
    }.items():
        value = _field_value(message, *names)
        if value not in (None, ""):
            row[output_key] = value
    data = _field_value(message, "data")
    data_length = _sequence_length(data)
    if data_length is not None:
        row["data_length"] = data_length
    return [row]


def imu_message_to_timestamp_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    topic: str,
    source: str,
    message_index: int,
) -> list[dict[str, Any]]:
    """Convert a native ROS IMU message into timestamp/kinematics metadata."""

    row: dict[str, Any] = {
        "sequence_id": str(sequence_id),
        "time_s": float(time_s),
        "topic": str(topic),
        "source": str(source),
        "message_index": int(message_index),
    }
    frame_id = _message_frame_id(message)
    if frame_id not in (None, ""):
        row["frame_id"] = str(frame_id)
    _add_vector_components(
        row,
        "orientation",
        _field_value(message, "orientation"),
        components=("x", "y", "z", "w"),
    )
    _add_vector_components(
        row,
        "angular_velocity",
        _field_value(message, "angular_velocity"),
        components=("x", "y", "z"),
        suffix="_rad_s",
    )
    _add_vector_components(
        row,
        "linear_acceleration",
        _field_value(message, "linear_acceleration"),
        components=("x", "y", "z"),
        suffix="_m_s2",
    )
    _add_covariance_diagonal(row, "orientation_covariance", message)
    _add_covariance_diagonal(row, "angular_velocity_covariance", message)
    _add_covariance_diagonal(row, "linear_acceleration_covariance", message)
    return [row]


def kinematic_message_to_timestamp_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    topic: str,
    source: str,
    message_index: int,
    kind: str = "kinematic_timestamps",
) -> list[dict[str, Any]]:
    """Convert native ROS Twist/Accel messages into vector inventory rows."""

    row: dict[str, Any] = {
        "sequence_id": str(sequence_id),
        "time_s": float(time_s),
        "topic": str(topic),
        "source": str(source),
        "message_index": int(message_index),
    }
    frame_id = _message_frame_id(message)
    if frame_id not in (None, ""):
        row["frame_id"] = str(frame_id)
    child_frame_id = _message_child_frame_id(message)
    if child_frame_id not in (None, ""):
        row["child_frame_id"] = str(child_frame_id)

    normalized_kind = str(kind).strip().lower()
    if "accel" in normalized_kind or "acceleration" in normalized_kind:
        _add_accel_components(row, _kinematic_vector_source(message, "accel"))
    elif "twist" in normalized_kind or "velocity" in normalized_kind:
        _add_twist_components(row, _kinematic_vector_source(message, "twist"))
    else:
        twist_source = _kinematic_vector_source(message, "twist")
        accel_source = _kinematic_vector_source(message, "accel")
        if _has_linear_or_angular_vector(twist_source):
            _add_twist_components(row, twist_source)
        if _has_linear_or_angular_vector(accel_source):
            _add_accel_components(row, accel_source)
        if not _has_linear_or_angular_vector(twist_source) and _has_linear_or_angular_vector(
            message
        ):
            _add_twist_components(row, message)

    return [row]


def _kinematic_vector_source(message: Any, kind: str) -> Any:
    if kind == "twist":
        source = _field_value(message, "twist")
        nested = _field_value(source, "twist") if source is not None else None
        if _has_linear_or_angular_vector(nested):
            return nested
        if _has_linear_or_angular_vector(source):
            return source
    if kind == "accel":
        source = _field_value(message, "accel", "acceleration")
        nested = (
            _field_value(source, "accel", "acceleration") if source is not None else None
        )
        if _has_linear_or_angular_vector(nested):
            return nested
        if _has_linear_or_angular_vector(source):
            return source
    return message


def _has_linear_or_angular_vector(value: Any) -> bool:
    if value is None:
        return False
    return _field_value(value, "linear") is not None or _field_value(value, "angular") is not None


def _add_twist_components(row: dict[str, Any], source: Any) -> None:
    _add_vector_components(
        row,
        "linear_velocity",
        _field_value(source, "linear", "linear_velocity"),
        components=("x", "y", "z"),
        suffix="_m_s",
    )
    _add_vector_components(
        row,
        "angular_velocity",
        _field_value(source, "angular", "angular_velocity"),
        components=("x", "y", "z"),
        suffix="_rad_s",
    )


def _add_accel_components(row: dict[str, Any], source: Any) -> None:
    _add_vector_components(
        row,
        "linear_acceleration",
        _field_value(source, "linear", "linear_acceleration"),
        components=("x", "y", "z"),
        suffix="_m_s2",
    )
    _add_vector_components(
        row,
        "angular_acceleration",
        _field_value(source, "angular", "angular_acceleration"),
        components=("x", "y", "z"),
        suffix="_rad_s2",
    )


def audio_message_to_timestamp_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    topic: str,
    source: str,
    message_index: int,
) -> list[dict[str, Any]]:
    """Convert a native ROS audio message into timestamp/sample metadata."""

    row: dict[str, Any] = {
        "sequence_id": str(sequence_id),
        "time_s": float(time_s),
        "topic": str(topic),
        "source": str(source),
        "message_index": int(message_index),
    }
    frame_id = _message_frame_id(message)
    if frame_id not in (None, ""):
        row["frame_id"] = str(frame_id)
    for output_key, names in {
        "encoding": ("encoding", "sample_format", "format"),
        "sample_rate_hz": ("sample_rate", "sample_rate_hz", "rate", "frequency"),
        "channels": ("channels", "channel_count", "num_channels"),
        "layout": ("layout",),
    }.items():
        value = _field_value(message, *names)
        if value not in (None, ""):
            row[output_key] = value
    data = _field_value(message, "data", "samples", "audio", "frames")
    data_length = _sequence_length(data)
    if data_length is not None:
        row["data_length"] = data_length
    sample_count, frame_count, byte_count = _audio_sample_and_frame_counts(
        data,
        data_length,
        channels=_optional_positive_int(row.get("channels")),
        encoding=row.get("encoding"),
    )
    if byte_count is not None:
        row["byte_count"] = byte_count
    if sample_count is not None:
        row["sample_count"] = sample_count
    if frame_count is not None:
        row["frame_count"] = frame_count
    duration_s = _audio_duration_s(
        frame_count=frame_count,
        sample_rate_hz=_optional_positive_float(row.get("sample_rate_hz")),
    )
    if duration_s is not None:
        row["duration_s"] = duration_s
    return [row]


def _add_vector_components(
    row: dict[str, Any],
    prefix: str,
    value: Any,
    *,
    components: tuple[str, ...],
    suffix: str = "",
) -> None:
    if value is None:
        return
    for component in components:
        parsed = _field_float(value, component)
        if parsed is not None and math.isfinite(parsed):
            row[f"{prefix}_{component}{suffix}"] = parsed


def _add_covariance_diagonal(row: dict[str, Any], field_name: str, message: Any) -> None:
    raw = _field_value(message, field_name)
    if raw is None or isinstance(raw, (str, bytes, bytearray, dict)):
        return
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return
    if len(values) < 9:
        return
    prefix = field_name
    for index, label in ((0, "xx"), (4, "yy"), (8, "zz")):
        value = values[index]
        if math.isfinite(value):
            row[f"{prefix}_{label}"] = value


def _sequence_length(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(len(value))
    except TypeError:
        return None


def _audio_sample_and_frame_counts(
    data: Any,
    data_length: int | None,
    *,
    channels: int | None,
    encoding: Any,
) -> tuple[int | None, int | None, int | None]:
    if data_length is None:
        return None, None, None
    byte_count = int(data_length) if isinstance(data, (bytes, bytearray, memoryview)) else None
    if byte_count is not None:
        bytes_per_sample = _audio_bytes_per_sample(encoding)
        if bytes_per_sample is None:
            return None, None, byte_count
        sample_count = int(byte_count) // int(bytes_per_sample)
    else:
        sample_count = int(data_length)
    if channels is None or channels <= 1:
        return sample_count, sample_count, byte_count
    return sample_count, int(sample_count) // int(channels), byte_count


def _audio_bytes_per_sample(encoding: Any) -> int | None:
    normalized = str(encoding or "").strip().lower().replace("-", "").replace("_", "")
    if not normalized:
        return None
    if any(token in normalized for token in ("float64", "f64", "double")):
        return 8
    if any(token in normalized for token in ("float32", "f32", "32")):
        return 4
    if "24" in normalized:
        return 3
    if "16" in normalized:
        return 2
    if "8" in normalized:
        return 1
    return None


def _audio_duration_s(
    *,
    frame_count: int | None,
    sample_rate_hz: float | None,
) -> float | None:
    if frame_count is None or sample_rate_hz is None or sample_rate_hz <= 0.0:
        return None
    return float(frame_count) / float(sample_rate_hz)


def _optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _optional_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _spec_bool(spec: dict[str, Any], *names: str, default: bool = False) -> bool:
    for name in names:
        if name not in spec:
            continue
        value = spec[name]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
        return default
    return default


def _spec_int(spec: dict[str, Any], *names: str, default: int) -> int:
    for name in names:
        if name not in spec:
            continue
        value = spec[name]
        if value in (None, ""):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _spec_float(spec: dict[str, Any], *names: str, default: float) -> float:
    for name in names:
        if name not in spec:
            continue
        value = spec[name]
        if value in (None, ""):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _image_timestamp_template_rows(image_timestamps: pd.DataFrame) -> pd.DataFrame:
    template = (
        image_timestamps[["sequence_id", "time_s"]]
        .dropna(subset=["sequence_id", "time_s"])
        .drop_duplicates()
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )
    template["x_m"] = 0.0
    template["y_m"] = 0.0
    template["z_m"] = 0.0
    return normalize_truth_columns(template)


def camera_info_message_to_model(
    message: Any,
    *,
    source: str,
    spec: dict[str, Any] | None = None,
) -> CameraModel:
    """Convert a ROS ``sensor_msgs/msg/CameraInfo`` message into a camera model."""

    entry = dict(spec or {})
    intrinsics = _camera_info_intrinsics(message)
    return CameraModel(
        source=str(source),
        intrinsics=intrinsics,
        transform_camera_to_world=_transform_from_entry(entry),
        time_offset_s=float(entry.get("time_offset_s", 0.0)),
    )


def _camera_models_from_camera_info_topics(
    reader: Any,
    *,
    topic_connections: list[Any],
    by_topic: dict[str, dict[str, Any]],
) -> tuple[dict[str, CameraModel], list[dict[str, Any]]]:
    connections = [
        connection
        for connection in topic_connections
        if _is_camera_info_kind(by_topic[connection.topic])
    ]
    if not connections:
        return {}, []
    models: dict[str, CameraModel] = {}
    extracted: list[dict[str, Any]] = []
    message_counts = {str(connection.topic): 0 for connection in connections}
    for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
        message_counts[str(connection.topic)] = (
            message_counts.get(str(connection.topic), 0) + 1
        )
        spec = by_topic[connection.topic]
        kind = str(spec.get("kind", "camera_info")).strip().lower()
        message = reader.deserialize(rawdata, connection.msgtype)
        time_s = _message_time_s(message, timestamp_ns)
        source = _camera_info_source(spec, connection=connection, message=message)
        key = source.lower()
        if key in models:
            continue
        try:
            model = camera_info_message_to_model(message, source=source, spec=spec)
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
        models[key] = model
        extracted.append(
            {
                "topic": connection.topic,
                "kind": kind,
                "status": "extracted",
                "time_s": time_s,
                "rows": 1,
                "source": source,
            }
        )
    for connection in connections:
        topic = str(connection.topic)
        if message_counts.get(topic, 0) > 0:
            continue
        spec = by_topic[connection.topic]
        extracted.append(
            {
                "topic": connection.topic,
                "kind": str(spec.get("kind", "camera_info")).strip().lower(),
                "status": "matched_topic_no_messages",
                "msgtype": str(getattr(connection, "msgtype", "")),
            }
        )
    return models, extracted


def _camera_info_source(
    spec: dict[str, Any],
    *,
    connection: Any,
    message: Any,
) -> str:
    source = (
        spec.get("source")
        or spec.get("camera")
        or spec.get("camera_id")
        or _message_frame_id(message)
        or connection.topic.strip("/").replace("/", "_")
    )
    return str(source)


def _is_camera_info_kind(spec: dict[str, Any]) -> bool:
    return str(spec.get("kind", "")).strip().lower() in {
        "camera_info",
        "camera_info_calibration",
        "camera_intrinsics",
        "camera_intrinsics_calibration",
    }


def _camera_info_intrinsics(message: Any) -> CameraIntrinsics:
    matrix = _camera_info_matrix(message, "k", "K", "camera_matrix")
    if matrix is None:
        matrix = _camera_info_matrix(message, "p", "P", "projection_matrix")
    if matrix is None:
        raise ValueError("CameraInfo message needs K/k or P/p intrinsics")
    if len(matrix) >= 12:
        fx = matrix[0]
        fy = matrix[5]
        cx = matrix[2]
        cy = matrix[6]
    elif len(matrix) >= 9:
        fx = matrix[0]
        fy = matrix[4]
        cx = matrix[2]
        cy = matrix[5]
    else:
        raise ValueError("CameraInfo K/P intrinsics must contain at least 9 values")
    if fx == 0.0 or fy == 0.0:
        raise ValueError("CameraInfo intrinsics must have nonzero fx/fy")
    return CameraIntrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


def _camera_info_matrix(message: Any, *names: str) -> list[float] | None:
    for name in names:
        value = getattr(message, name, None)
        if value is None:
            continue
        data = getattr(value, "data", value)
        try:
            values = [float(item) for item in data]
        except (TypeError, ValueError):
            continue
        if values:
            return values
    return None


def livox_custom_message_to_points(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
) -> list[dict[str, Any]]:
    """Convert common Livox CustomMsg point arrays into normalized point rows."""

    parent_time_s = _message_stamp_time_s(message)
    base_time_s = parent_time_s if parent_time_s is not None else float(time_s)
    points = _field_sequence(
        message,
        ("points", "point", "cloud", "pointcloud", "livox_points"),
    )
    if not points:
        return []
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        xyz = _xyz_from_position(point)
        if xyz is None:
            continue
        row: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "time_s": _livox_point_time_s(point, base_time_s=base_time_s),
            "x_m": xyz[0],
            "y_m": xyz[1],
            "z_m": xyz[2],
            "livox_point_index": int(index),
        }
        for key, names in {
            "intensity": ("intensity", "reflectivity", "reflectance"),
            "livox_offset_time": ("offset_time", "offset_time_ns", "time_offset_ns"),
            "livox_tag": ("tag",),
            "livox_line": ("line", "laser_id", "channel"),
        }.items():
            value = _field_value(point, *names)
            if value not in (None, ""):
                row[key] = value
        rows.append(row)
    return rows


def pointcloud_message_to_points(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
) -> list[dict[str, Any]]:
    """Convert legacy sensor_msgs/PointCloud messages into normalized point rows."""

    parent_time_s = _message_stamp_time_s(message)
    base_time_s = parent_time_s if parent_time_s is not None else float(time_s)
    points = _field_sequence(message, ("points", "point", "cloud", "pointcloud"))
    if not points:
        return []
    channels = _pointcloud_channel_values(message)
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        xyz = _xyz_from_position(point)
        if xyz is None:
            continue
        row: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "time_s": float(base_time_s),
            "x_m": xyz[0],
            "y_m": xyz[1],
            "z_m": xyz[2],
            "pointcloud_point_index": int(index),
        }
        for channel_name, values in channels.items():
            if index >= len(values):
                continue
            value = values[index]
            row[f"pointcloud_channel_{channel_name}"] = value
            if channel_name in {"intensity", "reflectivity", "reflectance"}:
                row["intensity"] = value
        rows.append(row)
    return rows


def _pointcloud_channel_values(message: Any) -> dict[str, list[float]]:
    channels = _field_sequence(message, ("channels", "channel"))
    out: dict[str, list[float]] = {}
    for channel_index, channel in enumerate(channels):
        raw_name = _field_value(channel, "name")
        name = _safe_metadata_name(raw_name, default=f"channel_{channel_index}")
        values = _numeric_field_sequence(channel, ("values", "data"))
        if values:
            out[name] = values
    return out


def _safe_metadata_name(value: Any, *, default: str) -> str:
    text = str(value or "").strip().lower()
    safe = "".join(char if char.isalnum() else "_" for char in text).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or default


def _livox_point_time_s(point: Any, *, base_time_s: float) -> float:
    offset = _field_float(point, "offset_time_s", "time_offset_s")
    if offset is not None:
        return float(base_time_s) + offset
    offset_ns = _field_float(point, "offset_time", "offset_time_ns", "time_offset_ns")
    if offset_ns is not None:
        return float(base_time_s) + offset_ns * 1.0e-9
    offset_us = _field_float(point, "offset_time_us", "time_offset_us")
    if offset_us is not None:
        return float(base_time_s) + offset_us * 1.0e-6
    offset_ms = _field_float(point, "offset_time_ms", "time_offset_ms")
    if offset_ms is not None:
        return float(base_time_s) + offset_ms * 1.0e-3
    return float(base_time_s)


def radar_polar_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    angle_unit: str = "rad",
) -> list[dict[str, Any]]:
    """Convert common native polar radar message shapes into table rows."""

    target_angle_unit = _normalize_radar_angle_unit(angle_unit)
    parent_time_s = _message_stamp_time_s(message)
    default_time_s = parent_time_s if parent_time_s is not None else float(time_s)
    array_rows = _radar_parallel_array_rows(
        message,
        sequence_id=sequence_id,
        time_s=default_time_s,
        target_angle_unit=target_angle_unit,
    )
    if array_rows:
        return array_rows
    children = _radar_child_messages(message)
    if children:
        rows: list[dict[str, Any]] = []
        for index, child in enumerate(children):
            child_time_s = _message_stamp_time_s(child)
            row = _radar_polar_row_from_message(
                child,
                sequence_id=sequence_id,
                time_s=child_time_s if child_time_s is not None else default_time_s,
                target_angle_unit=target_angle_unit,
                index=index,
            )
            if row is not None:
                rows.append(row)
        return rows
    row = _radar_polar_row_from_message(
        message,
        sequence_id=sequence_id,
        time_s=default_time_s,
        target_angle_unit=target_angle_unit,
        index=None,
    )
    return [row] if row is not None else []


def laserscan_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    angle_unit: str = "rad",
    cluster_adjacent: bool = False,
    min_cluster_points: int = 1,
    max_cluster_range_gap_m: float | None = 1.0,
) -> list[dict[str, Any]]:
    """Convert a native ROS LaserScan message into polar range rows."""

    target_angle_unit = _normalize_radar_angle_unit(angle_unit)
    parent_time_s = _message_stamp_time_s(message)
    default_time_s = parent_time_s if parent_time_s is not None else float(time_s)
    ranges = _numeric_field_sequence(message, ("ranges", "ranges_m", "range", "range_m"))
    if not ranges:
        return []
    angle_min = _field_float(message, "angle_min", "min_angle", "start_angle")
    if angle_min is None:
        angle_min = 0.0
    angle_increment = _field_float(
        message,
        "angle_increment",
        "angle_step",
        "increment",
        "resolution",
    )
    if angle_increment is None:
        angle_max = _field_float(message, "angle_max", "max_angle", "end_angle")
        if angle_max is not None and len(ranges) > 1:
            angle_increment = (float(angle_max) - float(angle_min)) / float(len(ranges) - 1)
        else:
            angle_increment = 0.0
    range_min = _field_float(message, "range_min", "min_range", "minimum_range")
    range_max = _field_float(message, "range_max", "max_range", "maximum_range")
    time_increment = _field_float(
        message,
        "time_increment",
        "time_increment_s",
        "sample_time_s",
    )
    intensities = _numeric_field_sequence(
        message,
        ("intensities", "intensity", "reflectivity", "reflectivities"),
    )
    valid_returns: list[dict[str, Any]] = []
    for index, range_m in enumerate(ranges):
        if not math.isfinite(range_m) or range_m <= 0.0:
            continue
        if range_min is not None and range_m < range_min:
            continue
        if range_max is not None and range_m > range_max:
            continue
        azimuth_rad = float(angle_min) + float(index) * float(angle_increment)
        row_time_s = (
            float(default_time_s) + float(index) * float(time_increment)
            if time_increment is not None
            else float(default_time_s)
        )
        intensity = (
            float(intensities[index])
            if index < len(intensities) and math.isfinite(intensities[index])
            else None
        )
        valid_returns.append(
            {
                "index": int(index),
                "time_s": row_time_s,
                "range_m": float(range_m),
                "azimuth_rad": azimuth_rad,
                "intensity": intensity,
            }
        )
    if cluster_adjacent:
        return _laserscan_cluster_rows(
            valid_returns,
            sequence_id=sequence_id,
            target_angle_unit=target_angle_unit,
            min_cluster_points=min_cluster_points,
            max_cluster_range_gap_m=max_cluster_range_gap_m,
        )
    rows: list[dict[str, Any]] = []
    for item in valid_returns:
        row: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "time_s": item["time_s"],
            "range_m": item["range_m"],
            "azimuth": _convert_angle_unit(
                float(item["azimuth_rad"]),
                source_unit="rad",
                target_unit=target_angle_unit,
            ),
            "elevation": 0.0,
            "scan_index": int(item["index"]),
        }
        if item["intensity"] is not None:
            row["scan_intensity"] = item["intensity"]
        rows.append(row)
    return rows


def _laserscan_cluster_rows(
    valid_returns: list[dict[str, Any]],
    *,
    sequence_id: str,
    target_angle_unit: str,
    min_cluster_points: int,
    max_cluster_range_gap_m: float | None,
) -> list[dict[str, Any]]:
    min_points = max(int(min_cluster_points), 1)
    max_gap = (
        float(max_cluster_range_gap_m)
        if max_cluster_range_gap_m is not None and math.isfinite(float(max_cluster_range_gap_m))
        else None
    )
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for item in valid_returns:
        starts_new = False
        if previous is not None:
            starts_new = int(item["index"]) != int(previous["index"]) + 1
            if max_gap is not None and not starts_new:
                starts_new = abs(float(item["range_m"]) - float(previous["range_m"])) > max_gap
        if starts_new and current:
            clusters.append(current)
            current = []
        current.append(item)
        previous = item
    if current:
        clusters.append(current)

    rows: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters):
        if len(cluster) < min_points:
            continue
        xs = [
            float(item["range_m"]) * math.cos(float(item["azimuth_rad"]))
            for item in cluster
        ]
        ys = [
            float(item["range_m"]) * math.sin(float(item["azimuth_rad"]))
            for item in cluster
        ]
        x_mean = sum(xs) / float(len(xs))
        y_mean = sum(ys) / float(len(ys))
        range_m = math.hypot(x_mean, y_mean)
        azimuth_rad = math.atan2(y_mean, x_mean)
        start_index = int(cluster[0]["index"])
        end_index = int(cluster[-1]["index"])
        row: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "time_s": sum(float(item["time_s"]) for item in cluster) / float(len(cluster)),
            "range_m": range_m,
            "azimuth": _convert_angle_unit(
                azimuth_rad,
                source_unit="rad",
                target_unit=target_angle_unit,
            ),
            "elevation": 0.0,
            "track_id": f"laserscan:{sequence_id}:{start_index}-{end_index}",
            "confidence": float(len(cluster)),
            "scan_cluster_index": int(cluster_index),
            "scan_start_index": start_index,
            "scan_end_index": end_index,
        }
        intensities = [
            float(item["intensity"]) for item in cluster if item["intensity"] is not None
        ]
        if intensities:
            row["scan_intensity"] = sum(intensities) / float(len(intensities))
        rows.append(row)
    return rows


def _radar_parallel_array_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    target_angle_unit: str,
) -> list[dict[str, Any]]:
    ranges = _numeric_field_sequence(
        message,
        (
            "ranges_m",
            "range_m",
            "ranges",
            "range",
            "r",
            "rho",
            "distances_m",
            "distances",
        ),
    )
    azimuths = _angle_field_sequence(
        message,
        (
            "azimuths_rad",
            "azimuth_rad",
            "bearings_rad",
            "bearing_rad",
            "azimuths_deg",
            "azimuth_deg",
            "bearings_deg",
            "bearing_deg",
            "azimuths",
            "azimuth",
            "bearings",
            "bearing",
            "az",
            "theta",
        ),
        target_angle_unit=target_angle_unit,
    )
    if not ranges or not azimuths:
        return []
    elevations = _angle_field_sequence(
        message,
        (
            "elevations_rad",
            "elevation_rad",
            "elevations_deg",
            "elevation_deg",
            "elevations",
            "elevation",
            "pitch",
            "el",
        ),
        target_angle_unit=target_angle_unit,
    )
    confidences = _numeric_field_sequence(
        message,
        ("confidence", "confidences", "score", "scores", "probability", "probabilities"),
    )
    track_ids = _field_sequence(
        message,
        ("track_ids", "track_id", "ids", "id", "object_ids", "object_id"),
    )
    class_names = _field_sequence(
        message,
        ("class_names", "class_name", "labels", "label", "categories", "category"),
    )
    times = _numeric_field_sequence(
        message,
        ("times_s", "time_s", "timestamps_s", "timestamp_s", "timestamps", "timestamp"),
    )
    count = min(len(ranges), len(azimuths))
    rows: list[dict[str, Any]] = []
    for index in range(count):
        row: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "time_s": times[index] if index < len(times) else float(time_s),
            "range_m": ranges[index],
            "azimuth": azimuths[index],
            "elevation": elevations[index] if index < len(elevations) else 0.0,
            "radar_detection_index": int(index),
        }
        if index < len(confidences):
            row["confidence"] = confidences[index]
        if index < len(track_ids):
            row["track_id"] = str(track_ids[index])
        if index < len(class_names):
            row["class_name"] = str(class_names[index])
        rows.append(row)
    return rows


def _radar_child_messages(message: Any) -> list[Any]:
    for name in (
        "radar_polar",
        "radar_detections",
        "detections",
        "targets",
        "objects",
        "measurements",
        "returns",
        "tracks",
        "points",
    ):
        children = _field_sequence(message, (name,))
        if not children or all(_is_scalar_like(item) for item in children):
            continue
        return children
    return []


def _radar_polar_row_from_message(
    detection: Any,
    *,
    sequence_id: str,
    time_s: float,
    target_angle_unit: str,
    index: int | None,
) -> dict[str, Any] | None:
    range_m = _field_float(
        detection,
        "range_m",
        "range",
        "r",
        "rho",
        "distance_m",
        "distance",
    )
    azimuth = _angle_field_value(
        detection,
        (
            "azimuth_rad",
            "bearing_rad",
            "azimuth_deg",
            "bearing_deg",
            "azimuth",
            "bearing",
            "az",
            "theta",
        ),
        target_angle_unit=target_angle_unit,
    )
    if range_m is None or azimuth is None:
        return None
    elevation = _angle_field_value(
        detection,
        (
            "elevation_rad",
            "elevation_deg",
            "elevation",
            "pitch",
            "el",
        ),
        target_angle_unit=target_angle_unit,
    )
    row: dict[str, Any] = {
        "sequence_id": str(sequence_id),
        "time_s": float(time_s),
        "range_m": range_m,
        "azimuth": azimuth,
        "elevation": elevation if elevation is not None else 0.0,
    }
    if index is not None:
        row["radar_detection_index"] = int(index)
    for key, names in {
        "track_id": ("track_id", "track", "id", "object_id", "target_id"),
        "confidence": ("confidence", "score", "probability", "catprob", "cat_prob"),
        "class_name": ("class_name", "class", "label", "category", "uav_type"),
    }.items():
        value = _field_value(detection, *names)
        if value not in (None, ""):
            row[key] = str(value) if key != "confidence" else float(value)
    return row


def _field_value(value: Any, *names: str) -> Any | None:
    if isinstance(value, dict):
        lower = {str(key).lower(): item for key, item in value.items()}
        for name in names:
            if name.lower() in lower:
                return lower[name.lower()]
        return None
    for name in names:
        item = getattr(value, name, None)
        if item is not None:
            return item
    return None


def _field_float(value: Any, *names: str) -> float | None:
    raw = _field_value(value, *names)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _angle_field_value(
    value: Any,
    names: tuple[str, ...],
    *,
    target_angle_unit: str,
) -> float | None:
    for name in names:
        raw = _field_value(value, name)
        if raw in (None, ""):
            continue
        try:
            angle = float(raw)
        except (TypeError, ValueError):
            continue
        return _convert_angle_unit(
            angle,
            source_unit=_angle_unit_from_name(name) or target_angle_unit,
            target_unit=target_angle_unit,
        )
    return None


def _numeric_field_sequence(value: Any, names: tuple[str, ...]) -> list[float]:
    for name in names:
        raw_values = _field_sequence(value, (name,))
        if not raw_values:
            continue
        numbers: list[float] = []
        for raw in raw_values:
            try:
                numbers.append(float(raw))
            except (TypeError, ValueError):
                numbers = []
                break
        if numbers:
            return numbers
    return []


def _angle_field_sequence(
    value: Any,
    names: tuple[str, ...],
    *,
    target_angle_unit: str,
) -> list[float]:
    for name in names:
        raw_values = _field_sequence(value, (name,))
        if not raw_values:
            continue
        source_unit = _angle_unit_from_name(name) or target_angle_unit
        angles: list[float] = []
        for raw in raw_values:
            try:
                angle = float(raw)
            except (TypeError, ValueError):
                angles = []
                break
            angles.append(
                _convert_angle_unit(
                    angle,
                    source_unit=source_unit,
                    target_unit=target_angle_unit,
                )
            )
        if angles:
            return angles
    return []


def _field_sequence(value: Any, names: tuple[str, ...]) -> list[Any]:
    raw = _field_value(value, *names)
    if raw is None or isinstance(raw, (str, bytes, bytearray, dict)):
        return []
    try:
        items = list(raw)
    except TypeError:
        return []
    return items


def _is_scalar_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return True
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _normalize_radar_angle_unit(unit: str) -> str:
    normalized = str(unit).strip().lower()
    if normalized not in {"deg", "rad"}:
        raise ValueError("native radar angle_unit must be 'deg' or 'rad'")
    return normalized


def _angle_unit_from_name(name: str) -> str | None:
    lowered = str(name).lower()
    if "deg" in lowered:
        return "deg"
    if "rad" in lowered:
        return "rad"
    return None


def _convert_angle_unit(
    value: float,
    *,
    source_unit: str,
    target_unit: str,
) -> float:
    if source_unit == target_unit:
        return float(value)
    if source_unit == "deg" and target_unit == "rad":
        return math.radians(float(value))
    if source_unit == "rad" and target_unit == "deg":
        return math.degrees(float(value))
    raise ValueError("native radar angle units must be 'deg' or 'rad'")


def _message_stamp_time_s(message: Any) -> float | None:
    header = _field_value(message, "header")
    stamp = _field_value(header, "stamp") if header is not None else None
    if stamp is None:
        stamp = _field_value(message, "stamp")
    if stamp is not None:
        sec = _field_value(stamp, "sec", "secs")
        nanosec = _field_value(stamp, "nanosec", "nsecs")
        if nanosec is None:
            nanosec = 0
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


def bounding_box3d_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    frame_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert BoundingBox3D/BoundingBox3DArray-style messages into rows."""

    boxes = _bounding_box3d_children(message)
    if not boxes and _bounding_box3d_position(message) is not None:
        boxes = [message]
    parent_time_s = _message_stamp_time_s(message)
    parent_frame_id = _message_frame_id(message)
    rows: list[dict[str, Any]] = []
    for box_index, box in enumerate(boxes):
        if not _frame_filter_matches(
            box,
            child_frame_id=None,
            frame_id=frame_id,
            fallback_frame_id=parent_frame_id,
        ):
            continue
        position = _bounding_box3d_position(box)
        xyz = _xyz_from_position(position)
        if xyz is None:
            continue
        box_time_s = _message_stamp_time_s(box)
        row = {
            "sequence_id": sequence_id,
            "time_s": (
                box_time_s
                if box_time_s is not None
                else parent_time_s
                if parent_time_s is not None
                else float(time_s)
            ),
            "x_m": xyz[0],
            "y_m": xyz[1],
            "z_m": xyz[2],
            "box_index": int(box_index),
        }
        _add_frame_metadata(row, box, fallback_frame_id=parent_frame_id)
        _add_bounding_box3d_metadata(row, box)
        rows.append(row)
    return rows


def detection2d_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    frame_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert vision_msgs Detection2D/Detection2DArray messages into rows."""

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
        bbox = getattr(detection, "bbox", None)
        center = _detection2d_center(bbox)
        if center is None:
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
            "u_px": center[0],
            "v_px": center[1],
        }
        _add_frame_metadata(row, detection, fallback_frame_id=parent_frame_id)
        _add_detection2d_bbox_geometry(row, bbox, center=center)
        depth_m = _detection2d_depth_m(bbox)
        if depth_m is not None:
            row["depth_m"] = depth_m
        detection_id = getattr(detection, "id", None)
        if detection_id not in (None, ""):
            row["track_id"] = str(detection_id)
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


def tracked_objects_message_to_rows(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    frame_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert common tracked/detected object arrays into normalized pose rows."""

    objects = _tracked_object_children(message)
    if not objects and _tracked_object_pose_source(message) is not None:
        objects = [message]
    parent_time_s = _message_stamp_time_s(message)
    parent_frame_id = _message_frame_id(message)
    rows: list[dict[str, Any]] = []
    for object_index, tracked_object in enumerate(objects):
        if not _frame_filter_matches(
            tracked_object,
            child_frame_id=None,
            frame_id=frame_id,
            fallback_frame_id=parent_frame_id,
        ):
            continue
        source = _tracked_object_pose_source(tracked_object)
        if source is None:
            continue
        xyz = _message_position_xyz(source)
        if xyz is None:
            continue
        object_time_s = _message_stamp_time_s(tracked_object)
        row = {
            "sequence_id": sequence_id,
            "time_s": (
                object_time_s
                if object_time_s is not None
                else parent_time_s
                if parent_time_s is not None
                else float(time_s)
            ),
            "x_m": xyz[0],
            "y_m": xyz[1],
            "z_m": xyz[2],
            "object_index": int(object_index),
        }
        _add_frame_metadata(
            row,
            tracked_object,
            fallback_frame_id=parent_frame_id,
        )
        object_id = _tracked_object_id(tracked_object)
        if object_id is not None:
            row["object_id"] = object_id
            row["track_id"] = object_id
        class_name = _tracked_object_class_name(tracked_object)
        if class_name is not None:
            row["class_name"] = class_name
        confidence = _tracked_object_confidence(tracked_object)
        if confidence is not None:
            row["confidence"] = confidence
        _add_pose_covariance_metadata(row, tracked_object)
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
    return _field_value(message, "child_frame_id")


def _message_frame_id(message: Any) -> Any | None:
    header = _field_value(message, "header")
    if header is None:
        return _field_value(message, "frame_id")
    return _field_value(header, "frame_id")


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


def _camera_models_from_spec(
    spec: dict[str, Any],
    *,
    bag_path: Path,
    topic_map_json: Path,
    source: str,
    native_camera_models: dict[str, CameraModel] | None = None,
):
    models: dict[str, CameraModel] = {}
    if native_camera_models:
        models.update(native_camera_models)
    calibration_files = _camera_calibration_files_from_spec(
        spec,
        bag_path=bag_path,
        topic_map_json=topic_map_json,
    )
    if calibration_files:
        models.update(
            load_camera_models_from_files(
                calibration_files,
                source_hint_from_path=lambda _path: source,
            )
        )
    if not models:
        raise ValueError(
            "native Detection2D topics require camera_calibration_file "
            "or a nearby camera_info/intrinsics file or camera_info topic"
        )
    return models


def _camera_calibration_files_from_spec(
    spec: dict[str, Any],
    *,
    bag_path: Path,
    topic_map_json: Path,
) -> list[Path]:
    values = _spec_path_values(
        spec,
        scalar_keys=(
            "camera_calibration_file",
            "camera_calibration_path",
            "camera_intrinsics_file",
            "camera_intrinsics_path",
            "camera_info_file",
            "camera_info_path",
            "calibration_file",
            "calibration_path",
        ),
        list_keys=(
            "camera_calibration_files",
            "camera_intrinsics_files",
            "camera_info_files",
            "calibration_files",
        ),
    )
    topic_map_dir = Path(topic_map_json).parent
    bag_dir = Path(bag_path) if Path(bag_path).is_dir() else Path(bag_path).parent
    candidates = [
        _resolve_spec_path(str(value), topic_map_dir=topic_map_dir, bag_dir=bag_dir)
        for value in values
    ]
    if not candidates:
        for directory in (topic_map_dir, bag_dir):
            for name in _CAMERA_CALIBRATION_FILENAMES:
                candidates.append(directory / name)
    return _unique_existing_paths(candidates)


def _spec_path_values(
    spec: dict[str, Any],
    *,
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


def _resolve_spec_path(value: str, *, topic_map_dir: Path, bag_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    topic_map_sibling = topic_map_dir / path
    if topic_map_sibling.exists():
        return topic_map_sibling
    return bag_dir / path


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
    if all(_field_value(message, attr) is not None for attr in ("x", "y", "z")):
        return message
    point = _field_value(message, "point")
    if point is not None:
        return point
    position = _field_value(message, "position")
    if position is not None:
        return position
    center = _field_value(message, "center", "centroid")
    if center is not None:
        return center
    transform = _field_value(message, "transform")
    if transform is not None:
        translation = _field_value(transform, "translation")
        if translation is not None:
            return translation
    translation = _field_value(message, "translation")
    if translation is not None:
        return translation
    pose = _field_value(message, "pose")
    if pose is None:
        pose = message
    inner_pose = _field_value(pose, "pose")
    if inner_pose is not None:
        pose = inner_pose
    return _field_value(pose, "position")


def _message_position_xyz(message: Any) -> tuple[float, float, float] | None:
    position = _position_from_message(message)
    return _xyz_from_position(position)


def _xyz_from_position(position: Any | None) -> tuple[float, float, float] | None:
    if position is None:
        return None
    try:
        return (
            float(_field_value(position, "x")),
            float(_field_value(position, "y")),
            float(_field_value(position, "z")),
        )
    except (TypeError, ValueError, AttributeError):
        return None


def _detection3d_position(detection: Any) -> Any | None:
    bbox = getattr(detection, "bbox", None)
    center = getattr(bbox, "center", None)
    if center is None:
        return None
    return _position_from_message(center)


def _bounding_box3d_children(message: Any) -> list[Any]:
    for name in (
        "boxes",
        "bounding_boxes",
        "boundingboxes",
        "bboxes",
        "boxes3d",
        "bounding_boxes3d",
    ):
        values = _field_sequence(message, (name,))
        if values:
            return [value for value in values if value is not None]
    return []


def _bounding_box3d_position(box: Any) -> Any | None:
    for source in (
        _field_value(box, "center"),
        _nested_field_value(box, "bbox", "center"),
        _nested_field_value(box, "bounding_box", "center"),
        _nested_field_value(box, "box", "center"),
        _field_value(box, "pose"),
        box,
    ):
        if source is None:
            continue
        position = _position_from_message(source)
        if _xyz_from_position(position) is not None:
            return position
    return None


def _add_bounding_box3d_metadata(row: dict[str, Any], box: Any) -> None:
    box_id = _tracked_object_id(box)
    if box_id is None:
        box_id = _format_object_identifier(_field_value(box, "box_id", "bbox_id"))
    if box_id is not None:
        row["box_id"] = box_id
        row["track_id"] = box_id
    class_name = _tracked_object_class_name(box)
    if class_name is not None:
        row["class_name"] = class_name
    confidence = _tracked_object_confidence(box)
    if confidence is not None:
        row["confidence"] = confidence
    size = _field_value(box, "size", "dimensions", "scale", "extent")
    for key, nested_names, direct_names in (
        ("box_size_x_m", ("x", "size_x", "length"), ("size_x", "length")),
        ("box_size_y_m", ("y", "size_y", "width"), ("size_y", "width")),
        ("box_size_z_m", ("z", "size_z", "height"), ("size_z", "height")),
    ):
        value = _field_float(size, *nested_names) if size is not None else None
        if value is None:
            value = _field_float(box, *direct_names)
        if value is not None:
            row[key] = value


def _detection2d_center(bbox: Any | None) -> tuple[float, float] | None:
    if bbox is None:
        return None
    center = getattr(bbox, "center", None)
    if center is None:
        return None
    position = getattr(center, "position", None)
    source = position if position is not None else center
    try:
        return (float(getattr(source, "x")), float(getattr(source, "y")))
    except (TypeError, ValueError, AttributeError):
        return None


def _add_detection2d_bbox_geometry(
    row: dict[str, Any],
    bbox: Any | None,
    *,
    center: tuple[float, float],
) -> None:
    size = _detection2d_size(bbox)
    if size is None:
        return
    width, height = size
    row["x1"] = center[0] - width / 2.0
    row["y1"] = center[1] - height / 2.0
    row["x2"] = center[0] + width / 2.0
    row["y2"] = center[1] + height / 2.0


def _detection2d_size(bbox: Any | None) -> tuple[float, float] | None:
    if bbox is None:
        return None
    width = _optional_attr_float(bbox, "size_x", "width", "w")
    height = _optional_attr_float(bbox, "size_y", "height", "h")
    size = getattr(bbox, "size", None)
    if width is None and size is not None:
        width = _optional_attr_float(size, "x", "width")
    if height is None and size is not None:
        height = _optional_attr_float(size, "y", "height")
    if width is None or height is None:
        return None
    return (width, height)


def _detection2d_depth_m(bbox: Any | None) -> float | None:
    if bbox is None:
        return None
    center = getattr(bbox, "center", None)
    if center is None:
        return None
    position = getattr(center, "position", None)
    for source in (position, center):
        if source is None:
            continue
        depth = _optional_attr_float(
            source,
            "z",
            "depth",
            "depth_m",
            "range",
            "range_m",
            "distance",
            "distance_m",
        )
        if depth is not None:
            return depth
    return None


def _optional_attr_float(value: Any, *names: str) -> float | None:
    for name in names:
        raw = getattr(value, name, None)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


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


def _tracked_object_children(message: Any) -> list[Any]:
    for name in (
        "objects",
        "tracked_objects",
        "detected_objects",
        "perception_objects",
        "tracks",
        "detections",
        "targets",
    ):
        values = _field_sequence(message, (name,))
        if values:
            return [value for value in values if value is not None]
    return []


def _tracked_object_pose_source(tracked_object: Any) -> Any | None:
    for path in (
        (),
        ("pose",),
        ("position",),
        ("point",),
        ("center",),
        ("centroid",),
        ("bbox", "center"),
        ("bounding_box", "center"),
        ("box", "center"),
        ("pose_with_covariance",),
        ("pose_with_covariance", "pose"),
        ("kinematics", "pose_with_covariance"),
        ("kinematics", "pose_with_covariance", "pose"),
        ("state", "pose"),
        ("state", "pose", "pose"),
        ("state", "pose_covariance"),
        ("state", "pose_covariance", "pose"),
        ("state", "pose_with_covariance"),
        ("state", "pose_with_covariance", "pose"),
        ("object", "pose"),
        ("object", "pose", "pose"),
    ):
        source = tracked_object if not path else _nested_field_value(tracked_object, *path)
        if source is not None and _message_position_xyz(source) is not None:
            return source
    return None


def _tracked_object_id(tracked_object: Any) -> str | None:
    for path in (
        ("track_id",),
        ("tracking_id",),
        ("object_id",),
        ("object_id", "uuid"),
        ("object_id", "value"),
        ("id",),
        ("id", "uuid"),
        ("id", "value"),
        ("uuid",),
        ("track", "id"),
    ):
        raw = _nested_field_value(tracked_object, *path)
        object_id = _format_object_identifier(raw)
        if object_id is not None:
            return object_id
    return None


def _format_object_identifier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, str):
        return value
    if _is_scalar_like(value):
        return str(value)
    for name in ("uuid", "value", "data", "id"):
        nested = _field_value(value, name)
        if nested is not None and nested is not value:
            formatted = _format_object_identifier(nested)
            if formatted is not None:
                return formatted
    try:
        items = list(value)
    except TypeError:
        return str(value)
    if not items:
        return None
    if all(isinstance(item, int) and 0 <= item <= 255 for item in items):
        return "".join(f"{int(item):02x}" for item in items)
    if len(items) <= 8:
        return ":".join(str(item) for item in items)
    return str(value)


def _tracked_object_class_name(tracked_object: Any) -> str | None:
    for path in (
        ("class_name",),
        ("class_id",),
        ("category",),
        ("label",),
        ("type",),
        ("object_class",),
        ("classification",),
        ("classification", "label"),
        ("classification", "class_id"),
        ("classification", "name"),
        ("hypothesis", "class_id"),
        ("semantic", "label"),
    ):
        label = _classification_label(_nested_field_value(tracked_object, *path))
        if label is not None:
            return label
    for classification in _field_sequence(
        tracked_object,
        ("classifications", "classification_results", "labels"),
    ):
        label = _classification_label(classification)
        if label is not None:
            return label
    return None


def _classification_label(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    if _is_scalar_like(value):
        return str(value)
    hypothesis = _field_value(value, "hypothesis")
    if hypothesis is not None:
        label = _classification_label(hypothesis)
        if label is not None:
            return label
    for name in ("class_name", "class_id", "label", "name", "category", "type", "id"):
        raw = _field_value(value, name)
        if raw not in (None, ""):
            return str(raw)
    return None


def _tracked_object_confidence(tracked_object: Any) -> float | None:
    for path in (
        ("confidence",),
        ("score",),
        ("probability",),
        ("existence_probability",),
        ("tracking_confidence",),
        ("classification", "probability"),
        ("classification", "score"),
        ("hypothesis", "score"),
    ):
        value = _nested_field_float(tracked_object, *path)
        if value is not None:
            return value
    scores = [
        _classification_confidence(classification)
        for classification in _field_sequence(
            tracked_object,
            ("classifications", "classification_results", "labels"),
        )
    ]
    valid_scores = [score for score in scores if score is not None]
    if valid_scores:
        return max(valid_scores)
    return None


def _classification_confidence(value: Any) -> float | None:
    for name in ("confidence", "score", "probability"):
        score = _field_float(value, name)
        if score is not None:
            return score
    hypothesis = _field_value(value, "hypothesis")
    if hypothesis is not None:
        return _classification_confidence(hypothesis)
    return None


def _add_pose_covariance_metadata(row: dict[str, Any], message: Any) -> None:
    for source in _pose_covariance_sources(message):
        values = _numeric_field_sequence(source, ("covariance",))
        if len(values) < 15:
            continue
        xy_variance = max(values[0], values[7])
        z_variance = values[14]
        if xy_variance >= 0.0:
            row["std_xy_m"] = float(xy_variance) ** 0.5
        if z_variance >= 0.0:
            row["std_z_m"] = float(z_variance) ** 0.5
        return


def _pose_covariance_sources(message: Any) -> list[Any]:
    sources: list[Any] = []
    for path in (
        (),
        ("pose",),
        ("pose_with_covariance",),
        ("kinematics", "pose_with_covariance"),
        ("state", "pose_covariance"),
        ("state", "pose_with_covariance"),
    ):
        source = message if not path else _nested_field_value(message, *path)
        if source is not None:
            sources.append(source)
    return sources


def _nested_field_value(value: Any, *path: str) -> Any | None:
    current = value
    for name in path:
        if current is None:
            return None
        current = _field_value(current, name)
    return current


def _nested_field_float(value: Any, *path: str) -> float | None:
    raw = _nested_field_value(value, *path)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _detection3d_confidence(detection: Any) -> float | None:
    result = _first_detection_result(detection)
    if result is None:
        return None
    return _detection_result_score(result)


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
    best = results[0]
    best_score = _detection_result_score(best)
    for result in results[1:]:
        score = _detection_result_score(result)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best = result
            best_score = score
    return best


def _detection_result_score(result: Any) -> float | None:
    score = getattr(result, "score", None)
    if score in (None, ""):
        hypothesis = getattr(result, "hypothesis", None)
        if hypothesis is not None:
            score = getattr(hypothesis, "score", None)
    if score in (None, ""):
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None
