"""Minimal ROS PointCloud2 decoding helpers for MMUAD exports.

The functions operate on duck-typed ROS message objects, so they can be used
with messages produced by optional ROS/rosbags packages without importing ROS at
module import time.  Only the numeric point-field types needed for x/y/z point
clouds are decoded.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.io import point_rows_to_candidates
from raft_uav.mmuad.schema import CandidateFrame


@dataclass(frozen=True)
class PointFieldSpec:
    """Normalized subset of ``sensor_msgs/PointField`` metadata."""

    name: str
    offset: int
    datatype: int
    count: int = 1


# ROS sensor_msgs/PointField constants.
_POINT_FIELD_FORMATS: dict[int, tuple[str, int]] = {
    1: ("b", 1),   # INT8
    2: ("B", 1),   # UINT8
    3: ("h", 2),   # INT16
    4: ("H", 2),   # UINT16
    5: ("i", 4),   # INT32
    6: ("I", 4),   # UINT32
    7: ("f", 4),   # FLOAT32
    8: ("d", 8),   # FLOAT64
}


def pointcloud2_to_dataframe(message: Any) -> pd.DataFrame:
    """Decode a ROS ``sensor_msgs/PointCloud2``-like message into xyz rows.

    ``message`` may be a real ROS message, a ``rosbags`` deserialized object, or
    a small test double with attributes ``fields``, ``data``, ``width``,
    ``height``, ``point_step``, and optional ``row_step`` / ``is_bigendian`` /
    ``is_dense``. Organized clouds are decoded with row padding respected.
    The returned frame contains finite ``x_m``, ``y_m`` and ``z_m`` columns plus
    any decodable extra scalar fields.
    """

    fields = _normalize_fields(getattr(message, "fields", []))
    by_name = {field.name: field for field in fields}
    missing = {"x", "y", "z"}.difference(by_name)
    if missing:
        raise ValueError(f"PointCloud2 message missing required fields: {sorted(missing)}")
    point_step = int(getattr(message, "point_step"))
    data = bytes(getattr(message, "data"))
    if point_step <= 0:
        raise ValueError("PointCloud2 point_step must be positive")
    endian = ">" if bool(getattr(message, "is_bigendian", False)) else "<"
    records: list[dict[str, float]] = []
    for base in _point_offsets(message, point_step=point_step, data_length=len(data)):
        record: dict[str, float] = {}
        for field in fields:
            if field.count != 1:
                continue
            fmt, width = _POINT_FIELD_FORMATS.get(int(field.datatype), ("", 0))
            if not fmt:
                continue
            relative_offset = int(field.offset)
            if relative_offset < 0 or relative_offset + width > point_step:
                continue
            offset = base + relative_offset
            if offset + width > len(data):
                continue
            try:
                value = struct.unpack_from(endian + fmt, data, offset)[0]
            except struct.error:
                continue
            record[field.name] = float(value)
        if {"x", "y", "z"}.issubset(record):
            records.append(record)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=["x_m", "y_m", "z_m"])
    frame = frame.rename(columns={"x": "x_m", "y": "y_m", "z": "z_m"})
    return frame.loc[np.isfinite(frame[["x_m", "y_m", "z_m"]]).all(axis=1)].reset_index(drop=True)


def _point_offsets(message: Any, *, point_step: int, data_length: int) -> list[int]:
    width = int(getattr(message, "width", 0))
    height = max(1, int(getattr(message, "height", 1)))
    if width <= 0:
        return [index * point_step for index in range(data_length // point_step)]

    row_step = getattr(message, "row_step", None)
    row_step = int(row_step) if row_step is not None else point_step * width
    if row_step <= 0:
        raise ValueError("PointCloud2 row_step must be positive")
    if row_step < point_step * width:
        raise ValueError("PointCloud2 row_step is smaller than width * point_step")

    offsets: list[int] = []
    for row in range(height):
        row_base = row * row_step
        if row_base >= data_length:
            break
        for column in range(width):
            base = row_base + column * point_step
            if base + point_step > data_length:
                break
            offsets.append(base)
    return offsets


def pointcloud2_to_candidates(
    message: Any,
    *,
    sequence_id: str,
    time_s: float,
    source: str = "pointcloud2-cluster",
    voxel_size_m: float = 0.75,
    min_points: int = 3,
    point_extraction_mode: str = "static",
    dynamic_background_voxel_size_m: float | None = None,
    dynamic_background_min_frame_fraction: float = 0.6,
    dynamic_background_min_frames: int = 3,
    dynamic_background_neighbor_radius_voxels: int = 0,
) -> CandidateFrame:
    """Decode and cluster a PointCloud2-like message into candidate centroids."""

    points = pointcloud2_to_dataframe(message)
    points["sequence_id"] = str(sequence_id)
    points["time_s"] = _finite_timestamp_seconds(time_s)
    return point_rows_to_candidates(
        points,
        source=source,
        voxel_size_m=voxel_size_m,
        min_points=min_points,
        min_confidence=0.0,
        point_extraction_mode=point_extraction_mode,
        dynamic_background_voxel_size_m=dynamic_background_voxel_size_m,
        dynamic_background_min_frame_fraction=dynamic_background_min_frame_fraction,
        dynamic_background_min_frames=dynamic_background_min_frames,
        dynamic_background_neighbor_radius_voxels=dynamic_background_neighbor_radius_voxels,
    )


def _finite_timestamp_seconds(value: Any) -> float:
    if isinstance(value, bool | np.bool_):
        raise ValueError("PointCloud2 time_s must be a finite numeric timestamp")
    try:
        timestamp_s = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("PointCloud2 time_s must be a finite numeric timestamp") from exc
    if not np.isfinite(timestamp_s):
        raise ValueError("PointCloud2 time_s must be a finite numeric timestamp")
    return timestamp_s


def _normalize_field_name(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).replace("\x00", "").strip()


def _normalize_fields(fields: Iterable[Any]) -> list[PointFieldSpec]:
    normalized: list[PointFieldSpec] = []
    for field in fields:
        normalized.append(
            PointFieldSpec(
                name=_normalize_field_name(getattr(field, "name")),
                offset=int(getattr(field, "offset")),
                datatype=int(getattr(field, "datatype")),
                count=int(getattr(field, "count", 1)),
            )
        )
    return normalized
