"""Sequence discovery/loading helpers for MMUAD-style directory exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.calibration import (
    CalibrationSet,
    load_calibration_auto,
    transform_candidate_frame,
)
from raft_uav.mmuad.camera import load_camera_detections_csv_as_candidates, load_camera_models
from raft_uav.mmuad.io import (
    load_candidate_file,
    load_point_cloud_file_as_candidates,
    load_truth_file,
    merge_candidate_frames,
)
from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates
from raft_uav.mmuad.rosbag_bridge import load_topic_map_exports
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame, normalize_truth_columns


TABLE_SUFFIXES = (".csv", ".tsv", ".txt")
TRAJECTORY_SUFFIXES = (".npy", ".npz")
POINT_FILE_SUFFIXES = (".csv", ".tsv", ".txt", ".npy", ".npz", ".pcd", ".ply")
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
    "lidar",
    "livox",
    "livox_avia",
    "mid360",
)
TRUTH_DIR_TOKENS = ("truth", "ground_truth", "gt", "label", "labels", "leica")
MODALITY_DIR_TOKENS = (
    CANDIDATE_DIR_TOKENS
    + POINT_DIR_TOKENS
    + TRUTH_DIR_TOKENS
    + ("camera", "cam", "image", "images", "class", "classes", "radar")
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
    calibration_file: Path | None


def discover_sequence_paths(root: Path, *, sequence_glob: str = "*") -> list[SequencePaths]:
    """Discover sequence folders in an exported MMUAD-style directory.

    The helper intentionally supports normalized/exported files rather than the
    official raw archive.  It looks for common names such as ``candidates.csv``,
    ``*_candidates.csv``, delimited variants such as ``candidates.tsv`` or
    ``detections.txt``, compact NumPy trajectory tables such as
    ``candidates.npy`` or ``trajectory.npz``, ``points.csv`` / ``points.tsv``,
    ``*_points.csv``, ASCII ``*.pcd``, ASCII ``*.ply``, exported ROS topic-map
    JSON files, ``truth.csv`` / ``truth.npy``, and ``calibration.json`` under
    each sequence folder.  If ``root`` itself holds such files, it is treated as
    a single sequence.
    """

    root = Path(root)
    sequence_dirs = _candidate_sequence_dirs(root, sequence_glob=sequence_glob)
    if sequence_dirs:
        return [_sequence_from_dir(path) for path in sequence_dirs]
    if _looks_like_sequence(root):
        return [_sequence_from_dir(root)]
    return []


def _candidate_sequence_dirs(root: Path, *, sequence_glob: str) -> list[Path]:
    children = [
        path
        for path in sorted(root.glob(sequence_glob))
        if path.is_dir() and not _is_modality_dir(path)
    ]
    candidates: list[Path] = []
    candidates.extend(path for path in children if _looks_like_sequence(path))
    for child in children:
        candidates.extend(
            path
            for path in sorted(child.glob(sequence_glob))
            if path.is_dir()
            and not _is_modality_dir(path)
            and _looks_like_sequence(path)
        )
    return _unique_paths(candidates)


def load_sequence_export(
    paths: SequencePaths,
    *,
    apply_calibration: bool = True,
    voxel_size_m: float = 0.75,
    min_cluster_points: int = 3,
    radar_azimuth_convention: str = "north-clockwise",
    radar_angle_unit: str = "deg",
    camera_fixed_depth_m: float | None = None,
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
            source=path.stem.replace("_radar_polar", "-radar"),
            sequence_id=paths.sequence_id,
            azimuth_convention=radar_azimuth_convention,
            angle_unit=radar_angle_unit,
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
    if paths.camera_detection_csvs:
        if paths.calibration_file is None:
            raise ValueError(
                f"camera detections discovered for {paths.root} but no "
                "calibration/intrinsics file exists"
            )
        camera_models = load_camera_models(paths.calibration_file)
        candidate_frames.extend(
            load_camera_detections_csv_as_candidates(
                path,
                camera_models=camera_models,
                fixed_depth_m=camera_fixed_depth_m,
            )
            for path in paths.camera_detection_csvs
        )
    if not candidate_frames:
        raise ValueError(f"no candidate or point-cloud files discovered for {paths.root}")
    candidates = merge_candidate_frames(candidate_frames)
    for path in paths.truth_files or (() if paths.truth_file is None else (paths.truth_file,)):
        truth_frames.append(
            load_truth_file(path, default_sequence_id=paths.sequence_id)
        )
    truth = _merge_truth_frames(truth_frames)
    calibration = None
    if paths.calibration_file is not None:
        calibration = load_calibration_auto(paths.calibration_file)
        if apply_calibration:
            candidates = transform_candidate_frame(candidates, calibration)
    return candidates, truth, calibration


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
    )


def _sequence_from_dir(path: Path) -> SequencePaths:
    topic_maps = _topic_map_files(path)
    topic_map_paths = _topic_map_referenced_paths(topic_maps)
    truth_files = _without_paths(_truth_files(path), topic_map_paths)
    truth_file = truth_files[0] if truth_files else None
    calibration = _first_existing(
        [
            path / "calibration.json",
            path / "calib.json",
            path / "extrinsics.json",
            path / "calibration.yaml",
            path / "calib.yaml",
            path / "extrinsics.yaml",
            path / "calibration.yml",
            path / "calib.yml",
            path / "extrinsics.yml",
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
        calibration_file=calibration,
    )


def _candidate_files(path: Path) -> list[Path]:
    names = [
        path / f"{stem}{suffix}"
        for stem in ("candidates", "detections")
        for suffix in TABLE_SUFFIXES
    ]
    files = [item for item in names if item.exists()]
    for suffix in TABLE_SUFFIXES:
        files.extend(sorted(path.glob(f"*_candidates{suffix}")))
        files.extend(sorted(path.glob(f"*_detections{suffix}")))
    files.extend(
        _files_under_named_dirs(
            path,
            directory_tokens=CANDIDATE_DIR_TOKENS,
            suffixes=TABLE_SUFFIXES,
        )
    )
    return _unique_paths(
        [
            item
            for item in files
            if not ("radar" in item.stem.lower() and "polar" in item.stem.lower())
            and "camera" not in item.stem.lower()
        ]
    )


def _radar_polar_files(path: Path) -> list[Path]:
    names = [
        path / f"{stem}{suffix}"
        for stem in ("radar_polar", "radar_detections_polar")
        for suffix in TABLE_SUFFIXES
    ]
    files = [item for item in names if item.exists()]
    for suffix in TABLE_SUFFIXES:
        files.extend(sorted(path.glob(f"*_radar_polar{suffix}")))
        files.extend(sorted(path.glob(f"*_polar_radar{suffix}")))
        files.extend(sorted(path.glob(f"*_radar_detections_polar{suffix}")))
    return _unique_paths(files)


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
            if not _name_has_any(item, ("truth", "ground_truth", "gt", "label"))
            and not _name_has_any(item, ("point", "points", "cloud", "lidar", "livox"))
        ]
    )


def _camera_detection_files(path: Path) -> list[Path]:
    names = [
        path / f"{stem}{suffix}"
        for stem in ("camera_detections", "image_detections")
        for suffix in TABLE_SUFFIXES
    ]
    files = [item for item in names if item.exists()]
    for suffix in TABLE_SUFFIXES:
        files.extend(sorted(path.glob(f"*_camera_detections{suffix}")))
        files.extend(sorted(path.glob(f"*_image_detections{suffix}")))
    return _unique_paths(files)


def _point_files(path: Path) -> list[Path]:
    names = [
        path / f"{stem}{suffix}"
        for stem in ("points", "point_cloud", "lidar_points")
        for suffix in TABLE_SUFFIXES
    ]
    files = [item for item in names if item.exists()]
    for suffix in TABLE_SUFFIXES:
        files.extend(sorted(path.glob(f"*_points{suffix}")))
        files.extend(sorted(path.glob(f"*_point_cloud{suffix}")))
    files.extend(sorted(path.glob("*.pcd")))
    files.extend(sorted(path.glob("*.ply")))
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
    ]
    globbed: list[Path] = []
    for suffix in TABLE_SUFFIXES + TRAJECTORY_SUFFIXES:
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
        suffixes=TABLE_SUFFIXES + TRAJECTORY_SUFFIXES,
    )
    return _unique_paths([item for item in exact if item.exists()] + globbed + folder_files)


def _topic_map_files(path: Path) -> list[Path]:
    exact = [
        path / "topic_map.json",
        path / "topic_map_exports.json",
        path / "mmuad_topic_map.json",
    ]
    globbed = sorted(path.glob("*topic_map*.json"))
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    exports = payload.get("exports", [])
    if not isinstance(exports, list):
        return False
    return any(isinstance(item, dict) and item.get("path") for item in exports)


def _topic_map_referenced_paths(topic_maps: list[Path]) -> set[Path]:
    referenced: set[Path] = set()
    for topic_map in topic_maps:
        try:
            payload = json.loads(topic_map.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
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
        if not _name_has_any(item, ("candidate", "detection", "track", "trajectory", "result"))
        and not _name_has_any(item, ("truth", "ground_truth", "gt", "label"))
    ]


def _name_has_any(path: Path, tokens: tuple[str, ...]) -> bool:
    text = " ".join(part.lower() for part in path.parts[-2:])
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
        if not item.is_file() or item.suffix.lower() not in suffix_set:
            continue
        try:
            parents = Path(item.relative_to(path)).parts[:-1]
        except ValueError:
            continue
        if parents and _directory_name_has_any(parents[0], directory_tokens):
            files.append(item)
    return files


def _is_modality_dir(path: Path) -> bool:
    normalized = str(path.name).lower().replace("-", "_").replace(" ", "_")
    return normalized in set(MODALITY_DIR_TOKENS)


def _directory_name_has_any(name: str, tokens: tuple[str, ...]) -> bool:
    normalized = str(name).lower().replace("-", "_").replace(" ", "_")
    return any(
        normalized == token
        or normalized.startswith(f"{token}_")
        or normalized.endswith(f"_{token}")
        for token in tokens
    )


def _source_from_path(path: Path, *, sequence_root: Path, default: str) -> str:
    try:
        relative = path.relative_to(sequence_root)
    except ValueError:
        return default
    if len(relative.parts) <= 1:
        return default
    return str(relative.parts[-2]).replace(" ", "_").replace("-", "_")


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
