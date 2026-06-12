"""Radar export adapters for MMUAD-style tracking candidates.

The official MMUAD radar messages may use custom binary/ROS formats.  This
module handles a common exported intermediate: per-detection polar radar rows
with range, azimuth, optional elevation, and optional confidence/track columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

    Accepted aliases include ``range_m``/``range``/``r``,
    ``azimuth_deg``/``azimuth``/``az``, and
    ``elevation_deg``/``elevation``/``el``.  Missing elevation defaults to
    zero.  The output coordinates are in the radar/export frame unless a later
    calibration transform is applied. CSV/TSV/TXT and JSON row/table exports
    are supported.
    """

    frame = normalize_time_column_aliases(_read_radar_table(path), target="time_s")
    normalized = _normalize_radar_columns(frame)
    if sequence_id is not None:
        normalized["sequence_id"] = str(sequence_id)
    elif "sequence_id" not in normalized.columns:
        normalized["sequence_id"] = Path(path).parent.name
    if "time_s" not in normalized.columns:
        raise ValueError("radar polar table requires time_s/timestamp_s/time column")
    azimuth = _angle_to_rad(normalized["azimuth"].to_numpy(float), angle_unit=angle_unit)
    elevation = _angle_to_rad(normalized.get("elevation", 0.0), angle_unit=angle_unit)
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
    return CandidateFrame(normalize_candidate_columns(records))


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
    r = np.asarray(range_m, dtype=float)
    az = np.asarray(azimuth_rad, dtype=float)
    el = np.asarray(elevation_rad, dtype=float)
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
    return np.column_stack([x, y, z])


def _normalize_radar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lower = {str(col).lower(): col for col in frame.columns}
    rename: dict[object, str] = {}
    aliases = {
        "time_s": ("time_s", "timestamp_s", "timestamp", "time", "t", "sec"),
        "range_m": ("range_m", "range", "r", "rho", "distance_m"),
        "azimuth": ("azimuth_deg", "azimuth", "az", "bearing", "bearing_deg"),
        "elevation": ("elevation_deg", "elevation", "el", "pitch", "pitch_deg"),
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
    missing = {"range_m", "azimuth"}.difference(out.columns)
    if missing:
        raise ValueError(f"radar polar table missing columns: {sorted(missing)}")
    if "elevation" not in out.columns:
        out["elevation"] = 0.0
    return out


def _angle_to_rad(values, *, angle_unit: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
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
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _read_radar_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _read_json_radar_table(path)
    return _read_delimited_table(path)


def _read_json_radar_table(path: Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = _json_radar_records(payload)
    return _json_radar_records_to_frame(records, path=path)


def _json_radar_records(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in (
        "radar_polar",
        "radar_detections",
        "detections",
        "tracks",
        "rows",
        "data",
    ):
        nested = _mapping_get_case_insensitive(payload, key)
        if nested is not None:
            return _json_radar_records(nested)
    if _looks_like_radar_column_map(payload) or _looks_like_radar_row(payload):
        return payload
    return []


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


_RADAR_HINT_KEYS = {
    "time_s",
    "timestamp",
    "timestamp_s",
    "timestamp_ns",
    "timestamp_ms",
    "sec",
    "nanosec",
    "range_m",
    "range",
    "r",
    "rho",
    "distance_m",
    "azimuth_deg",
    "azimuth",
    "az",
    "bearing",
    "bearing_deg",
    "elevation_deg",
    "elevation",
    "el",
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
