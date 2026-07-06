"""Radar export adapters for MMUAD-style tracking candidates.

The official MMUAD radar messages may use custom binary/ROS formats.  This
module handles a common exported intermediate: per-detection polar radar rows
with range, azimuth, optional elevation, and optional confidence/track columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.io import JSON_TABLE_SUFFIXES, data_file_suffix, read_json_export_payload
from raft_uav.mmuad.schema import (
    CandidateFrame,
    normalize_candidate_columns,
    normalize_time_column_aliases,
)


RADAR_AZIMUTH_CONVENTIONS = (
    "north-clockwise",
    "east-counterclockwise",
    "east-clockwise",
    "x-forward-left-positive",
)


def load_radar_polar_csv_as_candidates(
    path: Path,
    *,
    source: str = "radar-polar",
    sequence_id: str | None = None,
    azimuth_convention: str = "north-clockwise",
    angle_unit: str = "deg",
    range_std_m: float = 2.0,
    angle_std_deg: float = 2.0,
    z_std_m: float = 5.0,
) -> CandidateFrame:
    """Load exported polar radar detections and convert them to candidates.

    Accepted aliases include ``range_m``/``range``/``r`` and explicit-unit angle
    columns such as ``azimuth_rad``/``azimuth_deg``/``bearing_rad`` plus
    ``elevation_rad``/``elevation_deg``/``pitch_rad``. Generic angle columns
    such as ``azimuth``/``az`` and ``elevation``/``el`` use ``angle_unit``.
    Missing elevation defaults to zero.  The output coordinates are in the
    radar/export frame unless a later calibration transform is applied.
    CSV/TSV/TXT and JSON row/table exports are supported.
    """

    return radar_polar_frame_to_candidates(
        _read_radar_table(path),
        source=source,
        sequence_id=sequence_id,
        default_sequence_id=Path(path).parent.name,
        azimuth_convention=azimuth_convention,
        angle_unit=angle_unit,
        range_std_m=range_std_m,
        angle_std_deg=angle_std_deg,
        z_std_m=z_std_m,
    )


def radar_polar_frame_to_candidates(
    frame: pd.DataFrame,
    *,
    source: str = "radar-polar",
    sequence_id: str | None = None,
    default_sequence_id: str = "default",
    azimuth_convention: str = "north-clockwise",
    angle_unit: str = "deg",
    range_std_m: float = 2.0,
    angle_std_deg: float = 2.0,
    z_std_m: float = 5.0,
) -> CandidateFrame:
    """Convert an exported polar-radar table frame into candidates."""

    frame = normalize_time_column_aliases(frame, target="time_s")
    normalized = _normalize_radar_columns(frame)
    if sequence_id is not None:
        normalized["sequence_id"] = str(sequence_id)
    elif "sequence_id" not in normalized.columns:
        normalized["sequence_id"] = str(default_sequence_id)
    if "time_s" not in normalized.columns:
        raise ValueError("radar polar table requires time_s/timestamp_s/time column")
    azimuth = _radar_angle_column_to_rad(
        normalized,
        "azimuth",
        angle_unit=angle_unit,
        required=True,
    )
    elevation = _radar_angle_column_to_rad(
        normalized,
        "elevation",
        angle_unit=angle_unit,
        required=False,
    )
    range_m = pd.to_numeric(normalized["range_m"], errors="coerce").to_numpy(float)
    xyz = polar_to_cartesian(
        range_m,
        azimuth,
        elevation,
        azimuth_convention=azimuth_convention,
    )
    horizontal_std = _radar_horizontal_std(
        range_m,
        angle_std_deg=angle_std_deg,
        range_std_m=range_std_m,
    )
    records = pd.DataFrame(
        {
            "sequence_id": normalized["sequence_id"].astype(str),
            "time_s": pd.to_numeric(normalized["time_s"], errors="coerce"),
            "source": str(source),
            "track_id": normalized.get("track_id", np.nan),
            "x_m": xyz[:, 0],
            "y_m": xyz[:, 1],
            "z_m": xyz[:, 2],
            "std_xy_m": horizontal_std,
            "std_z_m": float(z_std_m),
            "confidence": pd.to_numeric(normalized.get("confidence", 1.0), errors="coerce"),
            "class_name": normalized.get("class_name", "uav"),
        }
    )
    return CandidateFrame(
        normalize_candidate_columns(records, default_sequence_id=str(default_sequence_id))
    )


def polar_to_cartesian(
    range_m: np.ndarray,
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray | float,
    *,
    azimuth_convention: str = "north-clockwise",
) -> np.ndarray:
    """Convert polar radar detections to Cartesian coordinates.

    Coordinate convention for output is ``x_m, y_m, z_m``.  For the most common
    geospatial convention, ``north-clockwise`` means azimuth zero points along
    +y and positive azimuth turns toward +x.
    """

    if azimuth_convention not in RADAR_AZIMUTH_CONVENTIONS:
        raise ValueError(
            f"unsupported azimuth convention {azimuth_convention!r}; "
            f"choices={RADAR_AZIMUTH_CONVENTIONS}"
        )
    r, az, el = np.broadcast_arrays(
        np.asarray(range_m, dtype=float),
        np.asarray(azimuth_rad, dtype=float),
        np.asarray(elevation_rad, dtype=float),
    )
    horizontal = r * np.cos(el)
    z = r * np.sin(el)
    if azimuth_convention == "north-clockwise":
        x = horizontal * np.sin(az)
        y = horizontal * np.cos(az)
    elif azimuth_convention == "east-counterclockwise":
        x = horizontal * np.cos(az)
        y = horizontal * np.sin(az)
    elif azimuth_convention == "east-clockwise":
        x = horizontal * np.cos(az)
        y = -horizontal * np.sin(az)
    else:  # x-forward-left-positive
        x = horizontal * np.cos(az)
        y = horizontal * np.sin(az)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def _normalize_radar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lower = {str(col).lower(): col for col in frame.columns}
    rename: dict[object, str] = {}
    aliases = {
        "time_s": ("time_s", "timestamp_s", "timestamp", "time", "t", "sec"),
        "range_m": ("range_m", "range", "r", "rho", "distance_m", "distance"),
        "azimuth_rad": ("azimuth_rad", "az_rad", "bearing_rad"),
        "azimuth_deg": ("azimuth_deg", "az_deg", "bearing_deg"),
        "azimuth": ("azimuth", "az", "bearing"),
        "elevation_rad": ("elevation_rad", "el_rad", "pitch_rad"),
        "elevation_deg": ("elevation_deg", "el_deg", "pitch_deg"),
        "elevation": ("elevation", "el", "pitch"),
        "track_id": ("track_id", "track", "id", "object_id"),
        "confidence": ("confidence", "score", "probability", "catprob", "cat_prob"),
        "sequence_id": ("sequence_id", "sequence", "seq", "scene_id"),
        "class_name": ("class_name", "uav_type", "class", "label", "category"),
    }
    for canonical, choices in aliases.items():
        if canonical in frame.columns:
            continue
        for alias in choices:
            original = lower.get(alias.lower())
            if original is not None:
                rename[original] = canonical
                break
    out = frame.rename(columns=rename).copy()
    missing = {"range_m"}.difference(out.columns)
    if not _radar_angle_column_exists(out, "azimuth"):
        missing.add("azimuth")
    if missing:
        raise ValueError(f"radar polar table missing columns: {sorted(missing)}")
    return out


def _radar_angle_column_exists(frame: pd.DataFrame, name: str) -> bool:
    return any(column in frame.columns for column in (f"{name}_rad", f"{name}_deg", name))


def _radar_angle_column_to_rad(
    frame: pd.DataFrame,
    name: str,
    *,
    angle_unit: str,
    required: bool,
) -> np.ndarray | float:
    for column, unit in (
        (f"{name}_rad", "rad"),
        (f"{name}_deg", "deg"),
        (name, angle_unit),
    ):
        if column in frame.columns:
            return _angle_to_rad(frame[column], angle_unit=unit)
    if required:
        raise ValueError(f"radar polar table missing columns: ['{name}']")
    return 0.0


def _angle_to_rad(values, *, angle_unit: str) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    if angle_unit == "deg":
        return np.deg2rad(arr)
    if angle_unit == "rad":
        return arr
    raise ValueError("angle_unit must be 'deg' or 'rad'")


def _radar_horizontal_std(
    range_m: np.ndarray,
    *,
    angle_std_deg: float,
    range_std_m: float,
) -> np.ndarray:
    angular = np.abs(np.asarray(range_m, dtype=float)) * np.deg2rad(float(angle_std_deg))
    return np.maximum(float(range_std_m), angular)


def _read_delimited_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if data_file_suffix(path) == ".tsv":
        return pd.read_csv(path, sep="\t")
    if data_file_suffix(path) == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _read_radar_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if data_file_suffix(path) in JSON_TABLE_SUFFIXES:
        return _read_json_radar_table(path)
    return _read_delimited_table(path)


def _read_json_radar_table(path: Path) -> pd.DataFrame:
    payload = read_json_export_payload(path)
    records = _json_radar_records(payload)
    return _json_radar_records_to_frame(records, path=path)


def _json_radar_records(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in _RADAR_NESTED_TABLE_KEYS:
        nested = _mapping_get_case_insensitive(payload, key)
        if nested is not None:
            return _json_radar_records_from_nested_container(payload, nested)
    if _looks_like_radar_column_map(payload) or _looks_like_radar_row(payload):
        return payload
    return []


def _json_radar_records_from_nested_container(parent: dict[Any, Any], nested: Any) -> Any:
    records = _json_radar_records(nested)
    defaults = _json_radar_parent_defaults(parent)
    if not defaults:
        return records
    if isinstance(records, list):
        merged: list[Any] = []
        for record in records:
            if not isinstance(record, dict):
                merged.append(record)
                continue
            merged.append(_merge_radar_parent_defaults(defaults, record))
        return merged
    if isinstance(records, dict) and (
        _looks_like_radar_column_map(records) or _looks_like_radar_row(records)
    ):
        return _merge_radar_parent_defaults(defaults, records)
    return records


def _json_radar_parent_defaults(parent: dict[Any, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key in _RADAR_PARENT_DEFAULT_KEYS:
        value = _mapping_get_case_insensitive(parent, key)
        if value is not None:
            defaults[key] = value
    return defaults


def _merge_radar_parent_defaults(defaults: dict[str, Any], record: dict[Any, Any]) -> dict[Any, Any]:
    row: dict[Any, Any] = {}
    record_has_time = _has_any_key(record, _RADAR_TIME_KEYS)
    record_has_sequence = _has_any_key(record, _RADAR_SEQUENCE_KEYS)
    for key, value in defaults.items():
        if key in _RADAR_TIME_KEYS and record_has_time:
            continue
        if key in _RADAR_SEQUENCE_KEYS and record_has_sequence:
            continue
        if _has_any_key(record, (key,)):
            continue
        row[key] = value
    row.update(record)
    return row


def _json_radar_records_to_frame(records: Any, *, path: Path | None = None) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, dict):
        if _looks_like_radar_column_map(records):
            return pd.DataFrame(records)
        if _looks_like_radar_row(records):
            return pd.DataFrame.from_records([records])
    if isinstance(records, list):
        if not records:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in records):
            return pd.DataFrame.from_records(records)
    label = str(path) if path is not None else "JSON payload"
    raise ValueError(f"radar polar JSON table {label} does not contain row objects")


def _mapping_get_case_insensitive(mapping: dict[Any, Any], key: str) -> Any | None:
    for candidate, value in mapping.items():
        if str(candidate).lower() == key.lower():
            return value
    return None


def _has_any_key(mapping: dict[Any, Any], keys: tuple[str, ...]) -> bool:
    present = {str(key).lower() for key in mapping}
    return any(key.lower() in present for key in keys)


_RADAR_NESTED_TABLE_KEYS = (
    "radar_polar",
    "radar_detections",
    "detections",
    "tracks",
    "targets",
    "objects",
    "measurements",
    "returns",
    "rows",
    "data",
)
_RADAR_SEQUENCE_KEYS = ("sequence_id", "sequence", "seq", "scene", "scene_id")
_RADAR_TIME_KEYS = (
    "time_s",
    "timestamp",
    "timestamp_s",
    "stamp_s",
    "time",
    "t",
    "sec",
    "secs",
    "seconds",
    "stamp",
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
    "nanosec",
    "nsec",
    "nsecs",
)
_RADAR_PARENT_DEFAULT_KEYS = _RADAR_SEQUENCE_KEYS + _RADAR_TIME_KEYS
_RADAR_HINT_KEYS = {
    *_RADAR_TIME_KEYS,
    "range_m",
    "range",
    "r",
    "rho",
    "distance_m",
    "distance",
    "azimuth_rad",
    "az_rad",
    "bearing_rad",
    "azimuth_deg",
    "az_deg",
    "azimuth",
    "az",
    "bearing",
    "bearing_deg",
    "elevation_rad",
    "el_rad",
    "pitch_rad",
    "elevation_deg",
    "el_deg",
    "elevation",
    "el",
    "pitch",
    "pitch_deg",
}


def _looks_like_radar_row(payload: dict[Any, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    return bool(keys.intersection(_RADAR_HINT_KEYS))


def _looks_like_radar_column_map(payload: dict[Any, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    if not keys.intersection(_RADAR_HINT_KEYS):
        return False
    return any(
        isinstance(value, (list, tuple))
        for key, value in payload.items()
        if str(key).lower() in _RADAR_HINT_KEYS
    )
