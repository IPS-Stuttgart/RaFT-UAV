"""Sequence discovery/loading helpers for MMUAD-style directory exports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.mmuad.calibration import (
    CalibrationSet,
    load_calibration_auto,
    transform_candidate_frame,
)
from raft_uav.mmuad.camera import (
    load_camera_detections_csv_as_candidates,
    load_camera_models_from_files,
)
from raft_uav.mmuad.io import (
    DISCOVERABLE_DELIMITED_TABLE_SUFFIXES,
    DISCOVERABLE_JSON_TABLE_SUFFIXES,
    JSON_TABLE_SUFFIXES as IO_JSON_TABLE_SUFFIXES,
    data_file_suffix,
    load_candidate_file,
    infer_time_s_from_filename,
    load_point_cloud_file_as_candidates,
    load_truth_file,
    merge_candidate_frames,
    path_matches_suffix,
    read_json_export_payload,
    read_text_export,
)
from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates
from raft_uav.mmuad.rosbag_bridge import load_topic_map_exports, load_topic_map_payload
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame, normalize_truth_columns


TABLE_SUFFIXES = DISCOVERABLE_DELIMITED_TABLE_SUFFIXES
JSON_TABLE_SUFFIXES = DISCOVERABLE_JSON_TABLE_SUFFIXES
JSON_DATA_SUFFIXES = tuple(sorted(IO_JSON_TABLE_SUFFIXES))
YAML_DATA_SUFFIXES = (".yaml", ".yml")
CALIBRATION_SUFFIXES = (".json", ".yaml", ".yml")
TRAJECTORY_SUFFIXES = (".npy", ".npz")
POINT_CLOUD_EXPORT_SUFFIXES = (
    ".pcd",
    ".pcd.gz",
    ".ply",
    ".ply.gz",
    ".las",
    ".las.gz",
    ".laz",
    ".bin",
    ".bin.gz",
)
POINT_FILE_SUFFIXES = (
    *TABLE_SUFFIXES,
    *JSON_TABLE_SUFFIXES,
    ".npy",
    ".npz",
    *POINT_CLOUD_EXPORT_SUFFIXES,
)
CANDIDATE_DIR_TOKENS = (
    "candidate",
    "candidates",
    "detection",
    "detections",
    "track",
    "tracks",
    "trajectory",
    "trajectories",
    "tracking",
    "tracking_results",
    "result",
    "results",
)
POINT_DIR_TOKENS = (
    "point",
    "points",
    "point_cloud",
    "cloud",
    "pcl",
    "lidar",
    "livox",
    "livox_avia",
    "mid360",
    "radar_enhance_pcl",
    "enhance_pcl",
)
TRUTH_DIR_TOKENS = ("truth", "ground_truth", "gt", "label", "labels", "leica")
CLASS_DIR_TOKENS = ("class", "classes", "uav_type", "uav_types", "category", "categories")
CLASS_JSON_CONTAINER_KEYS = ("sequences", "class_map", "classes", "mapping", "items", "labels")
CLASS_JSON_LABEL_ALIASES = (
    "uav_type",
    "class_name",
    "class",
    "label",
    "category",
    "type",
    "uav_class",
)
CLASS_JSON_SEQUENCE_ID_ALIASES = ("sequence_id", "sequence", "seq", "scene", "scene_id", "id", "name")
CLASS_JSON_METADATA_KEYS = ("schema", "version", "description", "metadata", "meta")
CLASS_FILE_SUFFIXES = (
    TABLE_SUFFIXES + JSON_TABLE_SUFFIXES + YAML_DATA_SUFFIXES + TRAJECTORY_SUFFIXES
)
RADAR_DIR_TOKENS = ("radar", "mmwave", "mmw")
CAMERA_DIR_TOKENS = ("camera", "cam", "image", "images")
CAMERA_INTRINSICS_NAMES = {
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
}
MODALITY_DIR_TOKENS = (
    CANDIDATE_DIR_TOKENS
    + POINT_DIR_TOKENS
    + TRUTH_DIR_TOKENS
    + CLASS_DIR_TOKENS
    + CAMERA_DIR_TOKENS
    + RADAR_DIR_TOKENS
)
OFFICIAL_TRACK5_TIMESTAMP_SOURCE_DIRS = {
    "ground-truth": ("ground_truth",),
    "image": ("image",),
    "lidar-360": ("lidar_360",),
    "livox-avia": ("livox_avia",),
    "radar-enhance-pcl": ("radar_enhance_pcl",),
}
OFFICIAL_TRACK5_TIMESTAMP_SOURCES = (
    "ground-truth-or-all",
    "all-modalities",
    *OFFICIAL_TRACK5_TIMESTAMP_SOURCE_DIRS,
)
RADAR_RANGE_ALIASES = ("range_m", "range", "r", "rho", "distance_m")
RADAR_AZIMUTH_ALIASES = ("azimuth_deg", "azimuth", "az", "bearing", "bearing_deg")
CANDIDATE_TIME_ALIASES = (
    "time_s",
    "timestamp_s",
    "stamp_s",
    "timestamp",
    "stamp",
    "time",
    "t",
    "sec",
    "secs",
    "seconds",
    "stamp.sec",
    "header.stamp.sec",
    "timestamp_ns",
    "time_ns",
    "stamp_ns",
    "nanoseconds",
    "timestamp_us",
    "time_us",
    "stamp_us",
    "timestamp_usec",
    "time_usec",
    "stamp_usec",
    "microseconds",
    "timestamp_ms",
    "time_ms",
    "stamp_ms",
    "milliseconds",
)
CANDIDATE_X_ALIASES = (
    "x_m",
    "x",
    "east_m",
    "pos_x",
    "position_x",
    "center_x",
    "bbox_center_x",
    "cx",
    "px",
    "point.x",
    "position.x",
    "pose.position.x",
    "pose.pose.position.x",
    "translation.x",
    "transform.translation.x",
    "center.position.x",
    "bbox.center.position.x",
    "bbox.center.x",
    "location.x",
    "coordinates.x",
)
CANDIDATE_Y_ALIASES = (
    "y_m",
    "y",
    "north_m",
    "pos_y",
    "position_y",
    "center_y",
    "bbox_center_y",
    "cy",
    "py",
    "point.y",
    "position.y",
    "pose.position.y",
    "pose.pose.position.y",
    "translation.y",
    "transform.translation.y",
    "center.position.y",
    "bbox.center.position.y",
    "bbox.center.y",
    "location.y",
    "coordinates.y",
)
CANDIDATE_Z_ALIASES = (
    "z_m",
    "z",
    "up_m",
    "pos_z",
    "position_z",
    "center_z",
    "bbox_center_z",
    "cz",
    "pz",
    "point.z",
    "position.z",
    "pose.position.z",
    "pose.pose.position.z",
    "translation.z",
    "transform.translation.z",
    "center.position.z",
    "bbox.center.position.z",
    "bbox.center.z",
    "location.z",
    "coordinates.z",
)
CAMERA_U_ALIASES = ("u_px", "u", "pixel_x", "center_u", "cx_px")
CAMERA_V_ALIASES = ("v_px", "v", "pixel_y", "center_v", "cy_px")
CAMERA_BBOX_X_ALIASES = ("x1", "xmin", "bbox_x1", "left")
CAMERA_BBOX_Y_ALIASES = ("y1", "ymin", "bbox_y1", "top")
CAMERA_BBOX_X2_ALIASES = ("x2", "xmax", "bbox_x2", "right")
CAMERA_BBOX_Y2_ALIASES = ("y2", "ymax", "bbox_y2", "bottom")
CAMERA_COMPACT_BBOX_ALIASES = (
    "bbox",
    "bbox_xywh",
    "xywh",
    "box_xywh",
    "bbox_xyxy",
    "xyxy",
    "box_xyxy",
)


@dataclass(frozen=True)
class SequencePaths:
    """Paths discovered for one exported sequence."""

    sequence_id: str
    root: Path
    candidate_csvs: tuple[Path, ...]
    candidate_trajectory_files: tuple[Path, ...]
    radar_polar_csvs: tuple[Path, ...]
    camera_detection_csvs: tuple[Path, ...]
    point_cloud_files: tuple[Path, ...]
    topic_map_jsons: tuple[Path, ...]
    truth_file: Path | None
    truth_files: tuple[Path, ...]
    class_files: tuple[Path, ...]
    calibration_file: Path | None
    camera_calibration_files: tuple[Path, ...] = ()


def discover_sequence_paths(root: Path, *, sequence_glob: str = "*") -> list[SequencePaths]:
    """Discover sequence folders in an exported MMUAD-style directory.

    The helper supports normalized/exported files and the public UG2+ Track 5
    sequence folders with ``ground_truth``, ``Image``, ``lidar_360``,
    ``livox_avia``, and ``radar_enhance_pcl`` subdirectories.  It also looks
    for common names such as ``candidates.csv``, ``*_candidates.csv``,
    delimited variants such as ``candidates.tsv`` or ``detections.txt``,
    JSON/JSONL row tables, compact NumPy trajectory tables, point-cloud files,
    exported ROS topic-map JSON/YAML files, ``truth.csv`` / ``truth.npy``, and
    ``calibration.json`` under each sequence folder.  If ``root`` itself holds
    such files, it is treated as a single sequence.
    """

    root = Path(root)
    sequence_dirs = _candidate_sequence_dirs(root, sequence_glob=sequence_glob)
    if sequence_dirs:
        return [_sequence_from_dir(path) for path in sequence_dirs]
    if _looks_like_sequence(root):
        return [_sequence_from_dir(root)]
    return []


def _candidate_sequence_dirs(root: Path, *, sequence_glob: str) -> list[Path]:
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    for child in _non_modality_child_dirs(root):
        _collect_sequence_dirs(
            child,
            root=root,
            sequence_glob=sequence_glob,
            candidates=candidates,
        )
    return _unique_paths(candidates)


def _collect_sequence_dirs(
    path: Path,
    *,
    root: Path,
    sequence_glob: str,
    candidates: list[Path],
) -> None:
    if _is_modality_dir(path):
        return
    if _sequence_dir_matches(path, root=root, sequence_glob=sequence_glob) and _looks_like_sequence(path):
        candidates.append(path)
        return
    for child in _non_modality_child_dirs(path):
        _collect_sequence_dirs(
            child,
            root=root,
            sequence_glob=sequence_glob,
            candidates=candidates,
        )


def _non_modality_child_dirs(path: Path) -> list[Path]:
    return [
        child
        for child in sorted(path.iterdir())
        if child.is_dir() and not _is_modality_dir(child)
    ]


def _sequence_dir_matches(path: Path, *, root: Path, sequence_glob: str) -> bool:
    if fnmatch(path.name, sequence_glob):
        return True
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.as_posix()
    return fnmatch(relative, sequence_glob)


def load_sequence_export(
    paths: SequencePaths,
    *,
    apply_calibration: bool = True,
    voxel_size_m: float = 0.75,
    min_cluster_points: int = 3,
    radar_azimuth_convention: str = "north-clockwise",
    radar_angle_unit: str = "deg",
    radar_polar_range_std_m: float = 2.0,
    radar_polar_angle_std_deg: float = 2.0,
    radar_polar_z_std_m: float = 5.0,
    camera_fixed_depth_m: float | None = None,
    camera_std_xy_m: float = 5.0,
    camera_std_z_m: float = 10.0,
) -> tuple[CandidateFrame, TruthFrame | None, CalibrationSet | None]:
    """Load candidates/truth for one discovered sequence export."""

    candidate_frames = [
        load_candidate_file(
            path,
            default_sequence_id=paths.sequence_id,
            source=_source_from_path(
                path,
                sequence_root=paths.root,
                default=path.stem.replace("_candidates", "-candidates"),
            ),
        )
        for path in paths.candidate_csvs
    ]
    candidate_frames.extend(
        load_candidate_file(
            path,
            default_sequence_id=paths.sequence_id,
            source=_source_from_path(
                path,
                sequence_root=paths.root,
                default=path.stem.replace("_trajectory", "-trajectory"),
            ),
        )
        for path in paths.candidate_trajectory_files
    )
    candidate_frames.extend(
        load_radar_polar_csv_as_candidates(
            path,
            source=_source_from_path(
                path,
                sequence_root=paths.root,
                default=path.stem.replace("_radar_polar", "-radar"),
            ),
            sequence_id=paths.sequence_id,
            azimuth_convention=radar_azimuth_convention,
            angle_unit=radar_angle_unit,
            range_std_m=radar_polar_range_std_m,
            angle_std_deg=radar_polar_angle_std_deg,
            z_std_m=radar_polar_z_std_m,
        )
        for path in paths.radar_polar_csvs
    )
    candidate_frames.extend(
        load_point_cloud_file_as_candidates(
            path,
            source=_source_from_path(
                path,
                sequence_root=paths.root,
                default=path.stem.replace("_points", "-cluster"),
            ),
            sequence_id=paths.sequence_id,
            time_s=_official_point_cloud_frame_time_s(path, sequence_root=paths.root),
            voxel_size_m=voxel_size_m,
            min_points=min_cluster_points,
        )
        for path in paths.point_cloud_files
    )
    truth_frames: list[TruthFrame] = []
    for path in paths.topic_map_jsons:
        bundle = load_topic_map_exports(path, base_dir=path.parent)
        candidate_frames.append(bundle.candidates)
        if bundle.truth is not None:
            truth_frames.append(bundle.truth)
    camera_models_loaded = False
    if paths.camera_detection_csvs:
        camera_calibration_files = paths.camera_calibration_files
        if not camera_calibration_files and paths.calibration_file is not None:
            camera_calibration_files = (paths.calibration_file,)
        if not camera_calibration_files:
            raise ValueError(
                f"camera detections discovered for {paths.root} but no "
                "calibration/intrinsics file exists"
            )
        camera_models = load_camera_models_from_files(
            camera_calibration_files,
            source_hint_from_path=lambda path: _camera_source_hint_from_path(
                path,
                sequence_root=paths.root,
            ),
        )
        camera_models_loaded = bool(camera_models)
        candidate_frames.extend(
            load_camera_detections_csv_as_candidates(
                path,
                camera_models=camera_models,
                default_source=_source_from_path(
                    path,
                    sequence_root=paths.root,
                    default="camera",
                ),
                fixed_depth_m=camera_fixed_depth_m,
                std_xy_m=camera_std_xy_m,
                std_z_m=camera_std_z_m,
            )
            for path in paths.camera_detection_csvs
        )
    if not candidate_frames:
        raise ValueError(f"no candidate or point-cloud files discovered for {paths.root}")
    candidates = merge_candidate_frames(candidate_frames)
    class_label = _sequence_class_label(paths.class_files, sequence_id=paths.sequence_id)
    if class_label is not None:
        rows = candidates.rows.copy()
        if "class_name" not in rows.columns:
            rows["class_name"] = class_label
        else:
            unknown = rows["class_name"].astype(str).str.strip().str.lower().isin(
                {"", "unknown", "nan", "none", "uav", "drone"}
            )
            rows.loc[unknown, "class_name"] = class_label
        candidates = CandidateFrame(rows)
    for path in paths.truth_files or (() if paths.truth_file is None else (paths.truth_file,)):
        truth_frames.append(
            load_truth_file(path, default_sequence_id=paths.sequence_id)
        )
    truth = _merge_truth_frames(truth_frames)
    calibration = None
    if paths.calibration_file is not None:
        try:
            calibration = load_calibration_auto(paths.calibration_file)
        except ValueError as exc:
            if camera_models_loaded and (
                _is_camera_intrinsics_file(paths.calibration_file)
                or _is_camera_only_calibration_error(exc)
            ):
                calibration = None
            else:
                raise
        if calibration is not None and apply_calibration:
            candidates = transform_candidate_frame(candidates, calibration)
    return candidates, truth, calibration


def official_track5_timestamp_template(
    paths: SequencePaths,
    *,
    timestamp_source: str = "ground-truth-or-all",
) -> TruthFrame:
    """Return a timestamp-only template from public Track 5 sequence folders.

    Public UG2+ Track 5 submissions require one row per requested sequence
    timestamp.  Training sequences expose those timestamps through
    ``ground_truth/<ros_timestamp>.npy``; validation/test-style folders may only
    expose sensor frames such as ``Image/<ros_timestamp>.png`` or point clouds.
    The returned zero-valued truth frame is a template for resampling results,
    not a substitute for hidden leaderboard labels.
    """

    timestamps = official_track5_sequence_timestamps(
        paths,
        timestamp_source=timestamp_source,
    )
    rows = pd.DataFrame(
        {
            "sequence_id": [paths.sequence_id] * len(timestamps),
            "time_s": timestamps,
            "x_m": [0.0] * len(timestamps),
            "y_m": [0.0] * len(timestamps),
            "z_m": [0.0] * len(timestamps),
        }
    )
    return TruthFrame(normalize_truth_columns(rows, default_sequence_id=paths.sequence_id))


def official_track5_sequence_timestamps(
    paths: SequencePaths,
    *,
    timestamp_source: str = "ground-truth-or-all",
) -> list[float]:
    """Return sorted unique timestamps from official Track 5 modality folders."""

    if timestamp_source not in OFFICIAL_TRACK5_TIMESTAMP_SOURCES:
        allowed = ", ".join(OFFICIAL_TRACK5_TIMESTAMP_SOURCES)
        raise ValueError(f"unsupported Track 5 timestamp source {timestamp_source!r}; allowed={allowed}")
    if timestamp_source == "ground-truth-or-all":
        timestamps = _timestamps_from_official_dirs(
            paths.root,
            OFFICIAL_TRACK5_TIMESTAMP_SOURCE_DIRS["ground-truth"],
        )
        if timestamps:
            return timestamps
        return _timestamps_from_official_dirs(
            paths.root,
            _all_official_timestamp_dirs(),
        )
    if timestamp_source == "all-modalities":
        return _timestamps_from_official_dirs(paths.root, _all_official_timestamp_dirs())
    return _timestamps_from_official_dirs(
        paths.root,
        OFFICIAL_TRACK5_TIMESTAMP_SOURCE_DIRS[timestamp_source],
    )


def _looks_like_sequence(path: Path) -> bool:
    if not path.is_dir():
        return False
    return bool(
        _candidate_files(path)
        or _candidate_trajectory_files(path)
        or _radar_polar_files(path)
        or _camera_detection_files(path)
        or _point_files(path)
        or _topic_map_files(path)
        or _truth_files(path)
        or _class_files(path)
        or _timestamps_from_official_dirs(path, _all_official_timestamp_dirs())
    )


def _is_camera_only_calibration_error(exc: ValueError) -> bool:
    return "at least one sensor calibration entry" in str(exc)


def _sequence_from_dir(path: Path) -> SequencePaths:
    topic_maps = _topic_map_files(path)
    topic_map_paths = _topic_map_referenced_paths(topic_maps)
    truth_files = _without_paths(_truth_files(path), topic_map_paths)
    truth_file = truth_files[0] if truth_files else None
    class_files = _without_paths(_class_files(path), topic_map_paths)
    camera_calibration_files = _without_paths(_camera_calibration_files(path), topic_map_paths)
    calibration = _first_existing(
        [
            path / "calibration.json",
            path / "calib.json",
            path / "extrinsics.json",
            path / "intrinsics.json",
            path / "camera_info.json",
            path / "camera_calibration.json",
            path / "camera_intrinsics.json",
            path / "calibration.yaml",
            path / "calib.yaml",
            path / "extrinsics.yaml",
            path / "intrinsics.yaml",
            path / "camera_info.yaml",
            path / "camera_calibration.yaml",
            path / "camera_intrinsics.yaml",
            path / "calibration.yml",
            path / "calib.yml",
            path / "extrinsics.yml",
            path / "intrinsics.yml",
            path / "camera_info.yml",
            path / "camera_calibration.yml",
            path / "camera_intrinsics.yml",
            path / "calibration.txt",
            path / "extrinsics.txt",
        ]
    )
    return SequencePaths(
        sequence_id=path.name,
        root=path,
        candidate_csvs=tuple(_without_paths(_candidate_files(path), topic_map_paths)),
        candidate_trajectory_files=tuple(
            _without_paths(_candidate_trajectory_files(path), topic_map_paths)
        ),
        radar_polar_csvs=tuple(_without_paths(_radar_polar_files(path), topic_map_paths)),
        camera_detection_csvs=tuple(
            _without_paths(_camera_detection_files(path), topic_map_paths)
        ),
        point_cloud_files=tuple(_without_paths(_point_files(path), topic_map_paths)),
        topic_map_jsons=tuple(topic_maps),
        truth_file=truth_file,
        truth_files=tuple(truth_files),
        class_files=tuple(class_files),
        calibration_file=calibration,
        camera_calibration_files=tuple(camera_calibration_files),
    )


def _candidate_files(path: Path) -> list[Path]:
    names = [
        path / f"{stem}{suffix}"
        for stem in ("candidates", "detections")
        for suffix in TABLE_SUFFIXES + JSON_TABLE_SUFFIXES
    ]
    files = [item for item in names if item.exists()]
    for suffix in TABLE_SUFFIXES + JSON_TABLE_SUFFIXES:
        files.extend(sorted(path.glob(f"*_candidates{suffix}")))
        files.extend(sorted(path.glob(f"*_detections{suffix}")))
    files.extend(
        _files_under_named_dirs(
            path,
            directory_tokens=CANDIDATE_DIR_TOKENS,
            suffixes=TABLE_SUFFIXES + JSON_TABLE_SUFFIXES,
        )
    )
    files.extend(
        item
        for item in _files_under_sensor_dirs(
            path,
            directory_tokens=RADAR_DIR_TOKENS,
            suffixes=TABLE_SUFFIXES + JSON_TABLE_SUFFIXES,
        )
        if _looks_like_cartesian_candidate_file(item)
    )
    return _unique_paths(
        [
            item
            for item in files
            if not ("radar" in item.stem.lower() and "polar" in item.stem.lower())
            and "camera" not in item.stem.lower()
            and not _relative_path_has_any(
                item,
                root=path,
                tokens=(
                    "topic_map",
                    "calibration",
                    "calib",
                    "extrinsic",
                    "class",
                    "category",
                    "truth",
                    "ground_truth",
                    "gt",
                    "label",
                ),
            )
        ]
    )


def _radar_polar_files(path: Path) -> list[Path]:
    names = [
        path / f"{stem}{suffix}"
        for stem in ("radar_polar", "radar_detections_polar")
        for suffix in TABLE_SUFFIXES + JSON_TABLE_SUFFIXES
    ]
    files = [item for item in names if item.exists()]
    for suffix in TABLE_SUFFIXES + JSON_TABLE_SUFFIXES:
        files.extend(sorted(path.glob(f"*_radar_polar{suffix}")))
        files.extend(sorted(path.glob(f"*_polar_radar{suffix}")))
        files.extend(sorted(path.glob(f"*_radar_detections_polar{suffix}")))
    sensor_files = [
        item
        for item in _files_under_sensor_dirs(
            path,
            directory_tokens=RADAR_DIR_TOKENS,
            suffixes=TABLE_SUFFIXES + JSON_TABLE_SUFFIXES,
        )
        if _looks_like_radar_polar_file(item)
    ]
    return _unique_paths(files + sensor_files)


def _candidate_trajectory_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in TRAJECTORY_SUFFIXES:
        files.extend(
            item
            for item in [
                path / f"candidates{suffix}",
                path / f"detections{suffix}",
                path / f"tracks{suffix}",
                path / f"trajectory{suffix}",
                path / f"trajectories{suffix}",
                path / f"tracking{suffix}",
                path / f"results{suffix}",
            ]
            if item.exists()
        )
        for pattern in (
            f"*candidates*{suffix}",
            f"*detections*{suffix}",
            f"*tracks*{suffix}",
            f"*trajectory*{suffix}",
            f"*trajectories*{suffix}",
            f"*tracking*{suffix}",
            f"*results*{suffix}",
            f"*_candidates{suffix}",
            f"*_detections{suffix}",
            f"*_tracks{suffix}",
            f"*_trajectory{suffix}",
            f"*_trajectories{suffix}",
            f"*_tracking{suffix}",
            f"*_results{suffix}",
        ):
            files.extend(sorted(path.glob(pattern)))
    files.extend(
        _files_under_named_dirs(
            path,
            directory_tokens=CANDIDATE_DIR_TOKENS,
            suffixes=TRAJECTORY_SUFFIXES,
        )
    )
    return _unique_paths(
        [
            item
            for item in files
            if not _relative_path_has_any(
                item,
                root=path,
                tokens=("truth", "ground_truth", "gt", "label"),
            )
            and not _relative_path_has_any(
                item,
                root=path,
                tokens=("point", "points", "cloud", "lidar", "livox"),
            )
        ]
    )


def _camera_detection_files(path: Path) -> list[Path]:
    names = [
        path / f"{stem}{suffix}"
        for stem in ("camera_detections", "image_detections")
        for suffix in TABLE_SUFFIXES + JSON_TABLE_SUFFIXES
    ]
    files = [item for item in names if item.exists()]
    for suffix in TABLE_SUFFIXES + JSON_TABLE_SUFFIXES:
        files.extend(sorted(path.glob(f"*_camera_detections{suffix}")))
        files.extend(sorted(path.glob(f"*_image_detections{suffix}")))
    sensor_files = [
        item
        for item in _files_under_sensor_dirs(
            path,
            directory_tokens=CAMERA_DIR_TOKENS,
            suffixes=TABLE_SUFFIXES + JSON_TABLE_SUFFIXES,
        )
        if _looks_like_camera_detection_file(item)
    ]
    return _unique_paths(files + sensor_files)


def _camera_calibration_files(path: Path) -> list[Path]:
    top_level_names = [
        path / f"{stem}{suffix}"
        for stem in (
            "calibration",
            "calib",
            "extrinsics",
            "intrinsics",
            "camera_info",
            "camera_calibration",
            "camera_intrinsics",
        )
        for suffix in CALIBRATION_SUFFIXES
    ]
    sensor_files = [
        item
        for item in _files_under_sensor_dirs(
            path,
            directory_tokens=CAMERA_DIR_TOKENS,
            suffixes=CALIBRATION_SUFFIXES,
        )
        if _is_camera_intrinsics_file(item)
    ]
    return _unique_paths([item for item in top_level_names if item.exists()] + sensor_files)


def _point_files(path: Path) -> list[Path]:
    names = [
        path / f"{stem}{suffix}"
        for stem in ("points", "point_cloud", "lidar_points")
        for suffix in TABLE_SUFFIXES + JSON_TABLE_SUFFIXES
    ]
    files = [item for item in names if item.exists()]
    for suffix in TABLE_SUFFIXES + JSON_TABLE_SUFFIXES:
        files.extend(sorted(path.glob(f"*_points{suffix}")))
        files.extend(sorted(path.glob(f"*_point_cloud{suffix}")))
    for suffix in (".pcd", ".pcd.gz", ".ply", ".ply.gz", ".las", ".las.gz", ".laz"):
        files.extend(sorted(path.glob(f"*{suffix}")))
    for pattern in ("*points*", "*point_cloud*", "*cloud*", "*lidar*", "*livox*"):
        for suffix in (".bin", ".bin.gz"):
            files.extend(sorted(path.glob(f"{pattern}{suffix}")))
    files.extend(_point_numpy_files(path))
    files.extend(
        _files_under_named_dirs(
            path,
            directory_tokens=POINT_DIR_TOKENS,
            suffixes=POINT_FILE_SUFFIXES,
        )
    )
    return _unique_paths(files)


def _truth_file(path: Path) -> Path | None:
    files = _truth_files(path)
    return files[0] if files else None


def _truth_files(path: Path) -> list[Path]:
    exact = [
        path / "truth.csv",
        path / "ground_truth.csv",
        path / "gt.csv",
        path / "truth.tsv",
        path / "ground_truth.tsv",
        path / "gt.tsv",
        path / "truth.txt",
        path / "ground_truth.txt",
        path / "gt.txt",
        path / "truth.npy",
        path / "ground_truth.npy",
        path / "gt.npy",
        path / "truth.npz",
        path / "ground_truth.npz",
        path / "gt.npz",
        path / "truth.json",
        path / "ground_truth.json",
        path / "gt.json",
    ]
    globbed: list[Path] = []
    for suffix in TABLE_SUFFIXES + JSON_TABLE_SUFFIXES + TRAJECTORY_SUFFIXES:
        for pattern in (
            f"*truth*{suffix}",
            f"*ground_truth*{suffix}",
            f"*label*{suffix}",
            f"gt*{suffix}",
        ):
            globbed.extend(sorted(path.glob(pattern)))
    folder_files = _files_under_named_dirs(
        path,
        directory_tokens=TRUTH_DIR_TOKENS,
        suffixes=TABLE_SUFFIXES + JSON_TABLE_SUFFIXES + TRAJECTORY_SUFFIXES,
    )
    return _unique_paths(
        [
            item
            for item in [item for item in exact if item.exists()] + globbed + folder_files
            if not _relative_path_has_any(
                item,
                root=path,
                tokens=("topic_map", "calibration", "calib", "extrinsic", "class", "category"),
            )
        ]
    )


def _class_files(path: Path) -> list[Path]:
    exact = [
        path / f"{stem}{suffix}"
        for stem in ("class", "classes", "uav_type", "category")
        for suffix in CLASS_FILE_SUFFIXES
    ]
    folder_files = _files_under_named_dirs(
        path,
        directory_tokens=CLASS_DIR_TOKENS,
        suffixes=CLASS_FILE_SUFFIXES,
    )
    return _unique_paths([item for item in exact if item.exists()] + folder_files)


def _topic_map_files(path: Path) -> list[Path]:
    exact = [
        path / f"{stem}{suffix}"
        for stem in ("topic_map", "topic_map_exports", "mmuad_topic_map")
        for suffix in (".json", ".yaml", ".yml")
    ]
    globbed = []
    for suffix in (".json", ".yaml", ".yml"):
        globbed.extend(sorted(path.glob(f"*topic_map*{suffix}")))
    candidates = _unique_paths(exact + globbed)
    return [
        item
        for item in candidates
        if "template" not in item.stem.lower()
        and "native" not in item.stem.lower()
        and _is_export_topic_map(item)
    ]


def _is_export_topic_map(path: Path) -> bool:
    try:
        payload = load_topic_map_payload(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    exports = payload.get("exports", [])
    if not isinstance(exports, list):
        return False
    return any(isinstance(item, dict) and item.get("path") for item in exports)


def _topic_map_referenced_paths(topic_maps: list[Path]) -> set[Path]:
    referenced: set[Path] = set()
    for topic_map in topic_maps:
        try:
            payload = load_topic_map_payload(topic_map)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        for export in payload.get("exports", []):
            if not isinstance(export, dict) or not export.get("path"):
                continue
            referenced.add((topic_map.parent / str(export["path"])).resolve())
    return referenced


def _without_paths(paths: list[Path], excluded: set[Path]) -> list[Path]:
    if not excluded:
        return paths
    return [path for path in paths if path.resolve() not in excluded]


def _point_numpy_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in (".npy", ".npz"):
        for pattern in (
            f"*points*{suffix}",
            f"*point_cloud*{suffix}",
            f"*cloud*{suffix}",
            f"*lidar*{suffix}",
            f"*livox*{suffix}",
        ):
            files.extend(sorted(path.glob(pattern)))
    return [
        item
        for item in files
        if not _relative_path_has_any(
            item,
            root=path,
            tokens=("candidate", "detection", "track", "trajectory", "result"),
        )
        and not _relative_path_has_any(
            item,
            root=path,
            tokens=("truth", "ground_truth", "gt", "label"),
        )
    ]


def _official_point_cloud_frame_time_s(path: Path, *, sequence_root: Path) -> float | None:
    if data_file_suffix(path) not in {".npy", ".npz"}:
        return None
    if not _relative_path_has_any(
        path,
        root=sequence_root,
        tokens=("lidar_360", "livox_avia", "radar_enhance_pcl"),
    ):
        return None
    return infer_time_s_from_filename(path)


def _all_official_timestamp_dirs() -> tuple[str, ...]:
    dirs: list[str] = []
    for values in OFFICIAL_TRACK5_TIMESTAMP_SOURCE_DIRS.values():
        dirs.extend(values)
    return tuple(dirs)


def _timestamps_from_official_dirs(root: Path, directory_names: tuple[str, ...]) -> list[float]:
    wanted = {_normalize_official_dir_name(name) for name in directory_names}
    timestamps: set[float] = set()
    for directory in _official_timestamp_dirs(root, wanted):
        for item in sorted(directory.rglob("*")):
            if not item.is_file():
                continue
            timestamp = _timestamp_from_filename_or_none(item)
            if timestamp is not None:
                timestamps.add(timestamp)
    return sorted(timestamps)


def _official_timestamp_dirs(root: Path, wanted: set[str]) -> list[Path]:
    if not Path(root).is_dir():
        return []
    return [
        item
        for item in sorted(Path(root).iterdir())
        if item.is_dir() and _normalize_official_dir_name(item.name) in wanted
    ]


def _normalize_official_dir_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "_")


def _timestamp_from_filename_or_none(path: Path) -> float | None:
    if not re.search(r"[-+]?\d*\.?\d+", Path(path).stem):
        return None
    timestamp = infer_time_s_from_filename(path)
    if not np.isfinite(timestamp):
        return None
    return float(timestamp)


def _relative_path_has_any(path: Path, *, root: Path, tokens: tuple[str, ...]) -> bool:
    try:
        parts = Path(path).relative_to(root).parts
    except ValueError:
        parts = (Path(path).name,)
    text = " ".join(part.lower() for part in parts)
    return any(token in text for token in tokens)


def _files_under_named_dirs(
    path: Path,
    *,
    directory_tokens: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> list[Path]:
    files: list[Path] = []
    suffix_set = {suffix.lower() for suffix in suffixes}
    for item in sorted(path.rglob("*")):
        if not item.is_file() or not path_matches_suffix(item, suffix_set):
            continue
        try:
            parents = Path(item.relative_to(path)).parts[:-1]
        except ValueError:
            continue
        if parents and _directory_name_has_any(parents[0], directory_tokens):
            files.append(item)
    return files


def _files_under_sensor_dirs(
    path: Path,
    *,
    directory_tokens: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> list[Path]:
    files: list[Path] = []
    suffix_set = {suffix.lower() for suffix in suffixes}
    for item in sorted(path.rglob("*")):
        if not item.is_file() or not path_matches_suffix(item, suffix_set):
            continue
        try:
            parents = Path(item.relative_to(path)).parts[:-1]
        except ValueError:
            continue
        if any(_sensor_directory_name_has_any(parent, directory_tokens) for parent in parents):
            files.append(item)
    return files


def _is_modality_dir(path: Path) -> bool:
    normalized = str(path.name).lower().replace("-", "_").replace(" ", "_")
    return normalized in set(MODALITY_DIR_TOKENS) or _sensor_directory_name_has_any(
        normalized,
        CAMERA_DIR_TOKENS + RADAR_DIR_TOKENS,
    )


def _directory_name_has_any(name: str, tokens: tuple[str, ...]) -> bool:
    normalized = str(name).lower().replace("-", "_").replace(" ", "_")
    return any(
        normalized == token
        or normalized.startswith(f"{token}_")
        or normalized.endswith(f"_{token}")
        for token in tokens
    )


def _sensor_directory_name_has_any(name: str, tokens: tuple[str, ...]) -> bool:
    normalized = str(name).lower().replace("-", "_").replace(" ", "_")
    if _directory_name_has_any(normalized, tokens):
        return True
    for token in tokens:
        if normalized.startswith(token) and normalized[len(token) :].isdigit():
            return True
    return False


def _looks_like_radar_polar_file(path: Path) -> bool:
    name = path.stem.lower()
    if any(token in name for token in ("radar_polar", "polar_radar", "polar")):
        return True
    return _table_has_column_groups(path, RADAR_RANGE_ALIASES, RADAR_AZIMUTH_ALIASES)


def _looks_like_cartesian_candidate_file(path: Path) -> bool:
    if _looks_like_radar_polar_file(path):
        return False
    return _table_has_column_groups(
        path,
        CANDIDATE_TIME_ALIASES,
        CANDIDATE_X_ALIASES,
        CANDIDATE_Y_ALIASES,
        CANDIDATE_Z_ALIASES,
    )


def _looks_like_camera_detection_file(path: Path) -> bool:
    name = path.stem.lower()
    if any(
        token in name
        for token in ("camera_detection", "image_detection", "detection", "bbox", "boxes")
    ):
        return True
    return _table_has_camera_detection_columns(path)


def _table_has_camera_detection_columns(path: Path) -> bool:
    columns = _table_columns(path)
    lower = {str(column).strip().lower() for column in columns}
    if not lower:
        return False
    has_pixel_center = _has_any_column(lower, CAMERA_U_ALIASES) and _has_any_column(
        lower, CAMERA_V_ALIASES
    )
    has_bbox = (
        _has_any_column(lower, CAMERA_BBOX_X_ALIASES)
        and _has_any_column(lower, CAMERA_BBOX_Y_ALIASES)
        and _has_any_column(lower, CAMERA_BBOX_X2_ALIASES)
        and _has_any_column(lower, CAMERA_BBOX_Y2_ALIASES)
    )
    has_compact_bbox = _has_any_column(lower, CAMERA_COMPACT_BBOX_ALIASES)
    return has_pixel_center or has_bbox or has_compact_bbox


def _table_has_column_groups(path: Path, *alias_groups: tuple[str, ...]) -> bool:
    columns = _table_columns(path)
    lower = {str(column).strip().lower() for column in columns}
    return bool(lower) and all(_has_any_column(lower, group) for group in alias_groups)


def _has_any_column(columns: set[str], aliases: tuple[str, ...]) -> bool:
    return any(alias in columns for alias in aliases)


def _table_columns(path: Path) -> list[str]:
    suffix = data_file_suffix(path)
    try:
        if suffix in JSON_DATA_SUFFIXES:
            return _json_table_columns(path)
        if suffix == ".tsv":
            frame = pd.read_csv(path, sep="\t", nrows=0)
        elif suffix == ".txt":
            frame = pd.read_csv(path, sep=None, engine="python", nrows=0)
        else:
            frame = pd.read_csv(path, nrows=0)
    except Exception:
        return []
    return [str(column) for column in frame.columns]


def _json_table_columns(path: Path) -> list[str]:
    try:
        payload = read_json_export_payload(path)
    except (OSError, json.JSONDecodeError):
        return []
    return _json_payload_columns(payload)


def _json_payload_columns(payload: Any) -> list[str]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                return [str(key) for key in item]
        return []
    if not isinstance(payload, Mapping):
        return []
    for key in (
        "radar_polar",
        "radar_detections",
        "detections",
        "objects",
        "targets",
        "measurements",
        "returns",
        "predictions",
        "truth",
        "points",
        "point_cloud",
        "rows",
        "data",
    ):
        nested = _mapping_get_case_insensitive(payload, key)
        if nested is not None:
            columns = _json_payload_columns(nested)
            if columns:
                return columns
    if _looks_like_json_column_map(payload):
        return [str(key) for key in payload]
    if all(not isinstance(value, (list, tuple, Mapping)) for value in payload.values()):
        return [str(key) for key in payload]
    return []


def _looks_like_json_column_map(payload: Mapping[Any, Any]) -> bool:
    return any(isinstance(value, (list, tuple)) for value in payload.values())


def _source_from_path(path: Path, *, sequence_root: Path, default: str) -> str:
    try:
        relative = path.relative_to(sequence_root)
    except ValueError:
        return default
    if len(relative.parts) <= 1:
        return default
    return str(relative.parts[-2]).replace(" ", "_").replace("-", "_")


def _camera_source_hint_from_path(path: Path, *, sequence_root: Path) -> str | None:
    try:
        relative = Path(path).relative_to(sequence_root)
    except ValueError:
        return None
    if len(relative.parts) <= 1:
        return None
    parent = str(relative.parts[-2]).replace(" ", "_").replace("-", "_")
    if _sensor_directory_name_has_any(parent, CAMERA_DIR_TOKENS):
        return parent
    return None


def _is_camera_intrinsics_file(path: Path) -> bool:
    name = Path(path).name.lower()
    return name in CAMERA_INTRINSICS_NAMES or (
        "camera" in name and ("intrinsic" in name or "info" in name)
    )


def _sequence_class_label(paths: tuple[Path, ...], *, sequence_id: str) -> str | None:
    labels: list[str] = []
    for path in paths:
        labels.extend(_class_labels_from_file(path, sequence_id=sequence_id))
    labels = [label for label in labels if label]
    if not labels:
        return None
    counts = pd.Series(labels, dtype="object").value_counts(sort=True)
    return str(counts.index[0])


def _class_labels_from_file(path: Path, *, sequence_id: str | None = None) -> list[str]:
    suffix = data_file_suffix(path)
    if suffix in TRAJECTORY_SUFFIXES:
        payload = np.load(path, allow_pickle=False)
        if isinstance(payload, np.lib.npyio.NpzFile):
            key = _first_npz_key(payload, preferred=("class", "classes", "uav_type", "label", "category"))
            values = np.asarray(payload[key]).reshape(-1)
        else:
            values = np.asarray(payload).reshape(-1)
        return [_format_class_label(value) for value in values]
    if suffix in {".csv", ".tsv", ".txt"}:
        if suffix == ".txt":
            raw_text = read_text_export(path, errors="ignore").strip()
            tokens = [token for token in raw_text.replace(",", " ").split() if token]
            if len(tokens) == 1:
                return [_format_class_label(tokens[0])]
        frame = _read_table(path)
        lower = {str(column).lower(): column for column in frame.columns}
        for alias in ("uav_type", "class_name", "class", "label", "category"):
            if alias in lower:
                return [
                    _format_class_label(value)
                    for value in frame[lower[alias]].dropna().tolist()
                ]
        if not frame.empty:
            return [_format_class_label(value) for value in frame.iloc[:, 0].dropna().tolist()]
    if suffix in JSON_DATA_SUFFIXES:
        payload = read_json_export_payload(path)
        return _class_labels_from_json_payload(payload, sequence_id=sequence_id)
    if suffix in YAML_DATA_SUFFIXES:
        payload = _read_yaml_export_payload(path)
        return _class_labels_from_json_payload(payload, sequence_id=sequence_id)
    return []


def _read_yaml_export_payload(path: Path) -> Any:
    text = read_text_export(path)
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return json.loads(text)
    return yaml.safe_load(text)


def _class_labels_from_json_payload(
    payload: Any,
    *,
    sequence_id: str | None,
) -> list[str]:
    scalar = _class_json_scalar_label(payload)
    if scalar is not None:
        return [scalar]
    if isinstance(payload, list):
        labels: list[str] = []
        for item in payload:
            labels.extend(_class_labels_from_json_payload(item, sequence_id=sequence_id))
        return labels
    if not isinstance(payload, dict):
        return []

    row_label = _class_json_row_label(payload, sequence_id=sequence_id)
    if row_label:
        return [row_label]

    labels: list[str] = []
    for key in CLASS_JSON_CONTAINER_KEYS:
        nested = _mapping_get_case_insensitive(payload, key)
        if nested is not None:
            labels.extend(_class_labels_from_json_payload(nested, sequence_id=sequence_id))
    if labels:
        return labels

    if sequence_id is not None:
        sequence_value = _mapping_get_case_insensitive(payload, sequence_id)
        if sequence_value is not None:
            return _class_labels_from_json_payload(sequence_value, sequence_id=None)
        if _looks_like_sequence_class_mapping(payload):
            return []

    return _class_labels_from_json_mapping(payload)


def _class_json_row_label(
    entry: Mapping[Any, Any],
    *,
    sequence_id: str | None,
) -> str | None:
    entry_sequence = _mapping_text_value(entry, CLASS_JSON_SEQUENCE_ID_ALIASES)
    if entry_sequence is not None and sequence_id is not None:
        if entry_sequence.lower() != str(sequence_id).lower():
            return None
    return _mapping_text_value(entry, CLASS_JSON_LABEL_ALIASES)


def _class_labels_from_json_mapping(payload: Mapping[Any, Any]) -> list[str]:
    labels: list[str] = []
    for key, value in payload.items():
        if str(key).lower() in CLASS_JSON_METADATA_KEYS:
            continue
        if str(key).lower() in CLASS_JSON_LABEL_ALIASES:
            label = _class_json_scalar_label(value)
            if label is not None:
                labels.append(label)
            continue
        labels.extend(_class_labels_from_json_payload(value, sequence_id=None))
    return labels


def _looks_like_sequence_class_mapping(payload: Mapping[Any, Any]) -> bool:
    useful_keys = [
        str(key).lower()
        for key in payload
        if str(key).lower() not in CLASS_JSON_METADATA_KEYS
    ]
    return bool(useful_keys) and not any(key in CLASS_JSON_LABEL_ALIASES for key in useful_keys)


def _mapping_get_case_insensitive(mapping: Mapping[Any, Any], key: str) -> Any | None:
    for candidate, value in mapping.items():
        if str(candidate).lower() == key.lower():
            return value
    return None


def _mapping_text_value(mapping: Mapping[Any, Any], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        value = _mapping_get_case_insensitive(mapping, alias)
        label = _class_json_scalar_label(value)
        if label is not None:
            return label
    return None


def _class_json_scalar_label(value: Any) -> str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str | int | float):
        return None
    label = _format_class_label(value)
    return label or None


def _read_table(path: Path) -> pd.DataFrame:
    if data_file_suffix(path) == ".tsv":
        return pd.read_csv(path, sep="\t")
    if data_file_suffix(path) == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _first_npz_key(payload: np.lib.npyio.NpzFile, *, preferred: tuple[str, ...]) -> str:
    lower_to_key = {key.lower(): key for key in payload.files}
    for key in preferred:
        matched = lower_to_key.get(key.lower())
        if matched is not None:
            return matched
    return payload.files[0]


def _format_class_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _merge_truth_frames(frames: list[TruthFrame]) -> TruthFrame | None:
    rows = [frame.rows for frame in frames if not frame.rows.empty]
    if not rows:
        return None
    return TruthFrame(normalize_truth_columns(pd.concat(rows, ignore_index=True)))
