"""Normalize inferred point-cloud source names for compressed exports."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from pathlib import Path

_PATCH_MARKER = "_raft_uav_point_cloud_source_name_patch_applied"


def _logical_data_stem(path: Path, *, data_file_suffix) -> str:
    """Return the filename without transparent compression or data suffixes."""

    path = Path(path)
    name = path.name
    if path.suffix.lower() == ".gz":
        name = name[: -len(path.suffix)]
    logical_suffix = str(data_file_suffix(path))
    if logical_suffix and name.lower().endswith(logical_suffix.lower()):
        name = name[: -len(logical_suffix)]
    return name


def _default_point_source(path: Path, *, data_file_suffix) -> str:
    """Match uncompressed source inference without retaining inner extensions."""

    return _logical_data_stem(
        path,
        data_file_suffix=data_file_suffix,
    ).replace("_points", "-cluster")


def install() -> None:
    """Patch point-cloud loaders so gzip compression does not alter source ids."""

    io_module = import_module("raft_uav.mmuad.io")
    if getattr(io_module, _PATCH_MARKER, False):
        return

    implementation = io_module._impl
    original_points = io_module.load_point_cloud_file_as_points
    original_candidates = io_module.load_point_cloud_file_as_candidates

    @wraps(original_points)
    def load_point_cloud_file_as_points(
        path: Path,
        *,
        source: str | None = None,
        sequence_id: str | None = None,
        time_s: float | None = None,
    ):
        effective_source = source or _default_point_source(
            path,
            data_file_suffix=io_module.data_file_suffix,
        )
        return original_points(
            path,
            source=effective_source,
            sequence_id=sequence_id,
            time_s=time_s,
        )

    @wraps(original_candidates)
    def load_point_cloud_file_as_candidates(
        path: Path,
        *,
        source: str | None = None,
        sequence_id: str | None = None,
        time_s: float | None = None,
        voxel_size_m: float = 0.75,
        min_points: int = 3,
        min_confidence: float = 0.0,
        point_extraction_mode: str = "static",
        dynamic_background_voxel_size_m: float | None = None,
        dynamic_background_min_frame_fraction: float = 0.6,
        dynamic_background_min_frames: int = 3,
        dynamic_background_neighbor_radius_voxels: int = 0,
    ):
        effective_source = source or _default_point_source(
            path,
            data_file_suffix=io_module.data_file_suffix,
        )
        return original_candidates(
            path,
            source=effective_source,
            sequence_id=sequence_id,
            time_s=time_s,
            voxel_size_m=voxel_size_m,
            min_points=min_points,
            min_confidence=min_confidence,
            point_extraction_mode=point_extraction_mode,
            dynamic_background_voxel_size_m=dynamic_background_voxel_size_m,
            dynamic_background_min_frame_fraction=dynamic_background_min_frame_fraction,
            dynamic_background_min_frames=dynamic_background_min_frames,
            dynamic_background_neighbor_radius_voxels=dynamic_background_neighbor_radius_voxels,
        )

    implementation.load_point_cloud_file_as_points = load_point_cloud_file_as_points
    implementation.load_point_cloud_file_as_candidates = (
        load_point_cloud_file_as_candidates
    )
    io_module.load_point_cloud_file_as_points = load_point_cloud_file_as_points
    io_module.load_point_cloud_file_as_candidates = load_point_cloud_file_as_candidates
    setattr(io_module, _PATCH_MARKER, True)
