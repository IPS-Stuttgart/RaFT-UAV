"""Dataset IO."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from raft_uav.io import aerpaw as _aerpaw

_CANONICAL_RF_EXPORT_NAMES = ("AADM.csv", "AADM_rerun.csv")
_RF_SENSOR_AND_RADAR_DIR_NAMES = (
    "RF Sensor and Radar",
    "RF_Sensor_and_Radar",
)
_ORIGINAL_DISCOVER_ATTR = "_discover_flights_before_rf_preference"
_original_discover_flights = getattr(
    _aerpaw,
    _ORIGINAL_DISCOVER_ATTR,
    _aerpaw.discover_flights,
)
setattr(_aerpaw, _ORIGINAL_DISCOVER_ATTR, _original_discover_flights)


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
    *,
    variant: str,
) -> _aerpaw.FlightPaths:
    """Prefer exact canonical AADM exports over unrelated CSV artifacts."""

    canonical = [
        candidate
        for candidate in (
            flight.root / name for name in _CANONICAL_RF_EXPORT_NAMES
        )
        if candidate.is_file()
    ]
    if not canonical:
        return flight
    rf_csv = _aerpaw._preferred_variant(canonical, variant=variant)
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
    return [
        _prefer_canonical_rf_export(flight, variant=variant)
        for flight in flights
    ]


_aerpaw.find_rf_sensor_and_radar_root = _find_rf_sensor_and_radar_root
_aerpaw._IMPL.find_rf_sensor_and_radar_root = _find_rf_sensor_and_radar_root
_aerpaw.discover_flights = _discover_flights
_aerpaw._IMPL.discover_flights = _discover_flights
