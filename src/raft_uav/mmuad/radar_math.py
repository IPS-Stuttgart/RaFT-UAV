from __future__ import annotations

import numpy as np
import pandas as pd

RADAR_AZIMUTH_CONVENTIONS = (
    "north-clockwise",
    "east-counterclockwise",
    "east-clockwise",
    "x-forward-left-positive",
)


def polar_to_cartesian(
    range_m: np.ndarray,
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray | float,
    *,
    azimuth_convention: str = "north-clockwise",
) -> np.ndarray:
    """Convert polar radar detections to Cartesian coordinates."""

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
    else:
        x = horizontal * np.cos(az)
        y = horizontal * np.sin(az)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def normalize_radar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lower = {str(col).lower(): col for col in frame.columns}
    rename: dict[object, str] = {}
    for canonical, choices in _RADAR_ALIASES.items():
        if canonical in frame.columns:
            continue
        for alias in choices:
            if (original := lower.get(alias)) is not None:
                rename[original] = canonical
                break
    out = frame.rename(columns=rename).copy()
    missing = {"range_m"}.difference(out.columns)
    if not any(column in out.columns for column in _angle_columns("azimuth")):
        missing.add("azimuth")
    if missing:
        raise ValueError(f"radar polar table missing columns: {sorted(missing)}")
    return out


def angle_column_to_rad(
    frame: pd.DataFrame,
    base_name: str,
    *,
    default_angle_unit: str,
    missing_default: float | None = None,
) -> np.ndarray:
    if default_angle_unit not in {"deg", "rad"}:
        raise ValueError("angle_unit must be 'deg' or 'rad'")
    for column, unit in zip(_angle_columns(base_name), ("rad", "deg", default_angle_unit)):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            return angle_to_rad(values, angle_unit=unit)
    if missing_default is not None:
        return angle_to_rad(missing_default, angle_unit=default_angle_unit)
    raise ValueError(f"radar polar table missing {base_name!r} angle column")


def angle_to_rad(values, *, angle_unit: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if angle_unit == "deg":
        return np.deg2rad(arr)
    if angle_unit == "rad":
        return arr
    raise ValueError("angle_unit must be 'deg' or 'rad'")


def radar_horizontal_std(
    range_m: np.ndarray,
    *,
    angle_std_deg: float,
    range_std_m: float,
) -> np.ndarray:
    angular = np.abs(np.asarray(range_m, dtype=float)) * np.deg2rad(float(angle_std_deg))
    return np.maximum(float(range_std_m), angular)


def _angle_columns(base_name: str) -> tuple[str, str, str]:
    return f"{base_name}_rad", f"{base_name}_deg", base_name


_RADAR_ALIASES = {
    "time_s": "time_s timestamp_s timestamp time t sec".split(),
    "range_m": "range_m range r rho distance_m".split(),
    "azimuth_rad": "azimuth_rad az_rad bearing_rad".split(),
    "azimuth_deg": "azimuth_deg az_deg bearing_deg".split(),
    "azimuth": "azimuth az bearing".split(),
    "elevation_rad": "elevation_rad el_rad pitch_rad".split(),
    "elevation_deg": "elevation_deg el_deg pitch_deg".split(),
    "elevation": "elevation el pitch".split(),
    "track_id": "track_id track id object_id".split(),
    "confidence": "confidence score probability catprob cat_prob".split(),
    "sequence_id": "sequence_id sequence seq scene_id".split(),
    "class_name": "class_name uav_type class label category".split(),
}
