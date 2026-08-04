"""Dataset IO."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.io import aerpaw as _aerpaw

_CANONICAL_RF_EXPORT_NAMES = ("AADM.csv", "AADM_rerun.csv")
_RF_SENSOR_AND_RADAR_DIR_NAMES = (
    "RF Sensor and Radar",
    "RF_Sensor_and_Radar",
)
_ORIGINAL_DISCOVER_ATTR = "_discover_flights_before_rf_preference"
_ORIGINAL_NORMALIZE_RF_ATTR = "_normalize_rf_before_clock_offset_validation"
_ORIGINAL_NORMALIZE_RADAR_ATTR = "_normalize_radar_before_clock_offset_validation"
_original_discover_flights = getattr(
    _aerpaw,
    _ORIGINAL_DISCOVER_ATTR,
    _aerpaw.discover_flights,
)
_original_normalize_rf = getattr(
    _aerpaw,
    _ORIGINAL_NORMALIZE_RF_ATTR,
    _aerpaw.normalize_rf,
)
_original_normalize_radar = getattr(
    _aerpaw,
    _ORIGINAL_NORMALIZE_RADAR_ATTR,
    _aerpaw.normalize_radar,
)
setattr(_aerpaw, _ORIGINAL_DISCOVER_ATTR, _original_discover_flights)
setattr(_aerpaw, _ORIGINAL_NORMALIZE_RF_ATTR, _original_normalize_rf)
setattr(_aerpaw, _ORIGINAL_NORMALIZE_RADAR_ATTR, _original_normalize_radar)


def _finite_real_scalar(value: object, *, field: str) -> float:
    """Return a finite real scalar without accepting pseudo-numbers."""

    error = f"{field} must be a finite real scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        item = scalar.item()
        if np.ma.is_masked(item) or isinstance(
            item,
            (bool, np.bool_, complex, np.complexfloating),
        ):
            raise ValueError(error)
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number):
        raise ValueError(error)
    return number


def _find_rf_sensor_and_radar_root(dataset_root: Path) -> Path:
    """Find either supported RF-root spelling at any extraction depth."""

    root = Path(dataset_root)
    if root.is_dir() and root.name in _RF_SENSOR_AND_RADAR_DIR_NAMES:
        return root
    for name in _RF_SENSOR_AND_RADAR_DIR_NAMES:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name in _RF_SENSOR_AND_RADAR_DIR_NAMES:
            return candidate
    raise FileNotFoundError(f"Could not find RF Sensor and Radar folder under {root}")


def _prefer_canonical_rf_export(
    flight: _aerpaw.FlightPaths,
) -> _aerpaw.FlightPaths:
    """Prefer exact canonical AADM exports without crossing file variants."""

    canonical = [
        candidate
        for candidate in (
            flight.root / name for name in _CANONICAL_RF_EXPORT_NAMES
        )
        if candidate.is_file()
    ]
    if not canonical or flight.rf_variant not in {"original", "rerun"}:
        return flight
    rf_csv = _aerpaw._preferred_variant(canonical, variant=flight.rf_variant)
    if rf_csv is None:
        return flight
    return replace(
        flight,
        rf_csv=rf_csv,
        rf_variant=_aerpaw._path_variant(rf_csv),
    )


def _discover_flights(
    dataset_root: Path,
    *,
    variant: str = "auto",
) -> list[_aerpaw.FlightPaths]:
    """Discover flights with deterministic canonical RF-file preference."""

    flights = _original_discover_flights(dataset_root, variant=variant)
    return [_prefer_canonical_rf_export(flight) for flight in flights]


def _normalize_rf(
    rf: pd.DataFrame,
    projector: Any,
    truth_origin_time: pd.Timestamp,
    default_std_m: float = 75.0,
    clock_offset_s: float = _aerpaw.DEFAULT_RF_CLOCK_OFFSET_S,
) -> pd.DataFrame:
    """Normalize RF rows after validating the independent clock offset."""

    return _original_normalize_rf(
        rf,
        projector,
        truth_origin_time,
        default_std_m=default_std_m,
        clock_offset_s=_finite_real_scalar(
            clock_offset_s,
            field="clock_offset_s",
        ),
    )


def _normalize_radar(
    radar: pd.DataFrame,
    projector: Any,
    truth_origin_time: pd.Timestamp,
    clock_offset_s: float = _aerpaw.DEFAULT_RADAR_CLOCK_OFFSET_S,
) -> pd.DataFrame:
    """Normalize radar rows after validating the independent clock offset."""

    return _original_normalize_radar(
        radar,
        projector,
        truth_origin_time,
        clock_offset_s=_finite_real_scalar(
            clock_offset_s,
            field="clock_offset_s",
        ),
    )


_aerpaw.find_rf_sensor_and_radar_root = _find_rf_sensor_and_radar_root
_aerpaw._IMPL.find_rf_sensor_and_radar_root = _find_rf_sensor_and_radar_root
_aerpaw.discover_flights = _discover_flights
_aerpaw._IMPL.discover_flights = _discover_flights
_aerpaw.normalize_rf = _normalize_rf
_aerpaw._IMPL.normalize_rf = _normalize_rf
_aerpaw.normalize_radar = _normalize_radar
_aerpaw._IMPL.normalize_radar = _normalize_radar
