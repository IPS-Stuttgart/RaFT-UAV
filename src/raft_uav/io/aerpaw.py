"""AERPAW Dataset-28 / Dryad RF Sensor and Radar loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.coordinates import LocalENUProjector


@dataclass(frozen=True)
class FlightPaths:
    """Paths for one RF/radar tracking flight."""

    name: str
    root: Path
    rf_csv: Path | None
    radar_json: Path | None
    truth_txt: Path | None


def find_rf_sensor_and_radar_root(dataset_root: Path) -> Path:
    """Find the RF Sensor and Radar directory under an extracted Dryad dataset."""

    root = Path(dataset_root)
    candidates = [root / "RF Sensor and Radar", root / "RF_Sensor_and_Radar"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name == "RF Sensor and Radar":
            return candidate
    raise FileNotFoundError(f"Could not find RF Sensor and Radar folder under {root}")


def discover_flights(dataset_root: Path) -> list[FlightPaths]:
    """Discover candidate flights with RF, radar, and ground-truth files."""

    rf_radar_root = find_rf_sensor_and_radar_root(dataset_root)
    flights: list[FlightPaths] = []
    for folder in sorted(path for path in rf_radar_root.iterdir() if path.is_dir()):
        rf_files = sorted(folder.glob("AADM*.csv"))
        radar_files = sorted(folder.glob("radar_data*.json"))
        truth_files = sorted(folder.glob("*vehicleOut*.txt"))
        if not (rf_files or radar_files or truth_files):
            continue
        flights.append(
            FlightPaths(
                name=folder.name,
                root=folder,
                rf_csv=rf_files[0] if rf_files else None,
                radar_json=radar_files[0] if radar_files else None,
                truth_txt=truth_files[0] if truth_files else None,
            )
        )
    return flights


def select_flight(dataset_root: Path, name: str) -> FlightPaths:
    """Select a discovered flight by exact name or case-insensitive substring."""

    flights = discover_flights(dataset_root)
    exact = [flight for flight in flights if flight.name == name]
    if len(exact) == 1:
        return exact[0]
    partial = [flight for flight in flights if name.lower() in flight.name.lower()]
    if len(partial) == 1:
        return partial[0]
    available = ", ".join(flight.name for flight in flights[:20])
    raise ValueError(f"Could not uniquely select flight {name!r}. Available: {available}")


def read_rf_csv(path: Path) -> pd.DataFrame:
    """Read Keysight RF sensor localization outputs."""

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def read_truth(path: Path) -> pd.DataFrame:
    """Read UAV telemetry text/csv data with delimiter inference."""

    return pd.read_csv(path, sep=None, engine="python")


def read_radar_json(path: Path) -> pd.DataFrame:
    """Read Fortem radar JSON logs into a flat table."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = list(_iter_radar_records(payload))
    return pd.json_normalize(records)


def _iter_radar_records(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield dictionaries that look like radar target records."""

    if isinstance(payload, dict):
        if {"range", "azimuth", "elevation"}.intersection(payload):
            yield payload
        for value in payload.values():
            yield from _iter_radar_records(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_radar_records(item)


def projector_from_truth(truth: pd.DataFrame) -> LocalENUProjector:
    """Create a local ENU projector from the first valid truth latitude/longitude row."""

    lat_col = _first_present(truth, ["Latitude", "latitude", "lat"])
    lon_col = _first_present(truth, ["Longitude", "longitude", "lon", "lng"])
    alt_col = _first_present(truth, ["Altitude", "altitude", "alt"])
    first = truth[[lat_col, lon_col, alt_col]].dropna().iloc[0]
    return LocalENUProjector(
        origin_latitude_deg=float(first[lat_col]),
        origin_longitude_deg=float(first[lon_col]),
        origin_altitude_m=float(first[alt_col]),
    )


def truth_to_enu(truth: pd.DataFrame, projector: LocalENUProjector) -> pd.DataFrame:
    """Append ENU columns to a truth dataframe."""

    lat_col = _first_present(truth, ["Latitude", "latitude", "lat"])
    lon_col = _first_present(truth, ["Longitude", "longitude", "lon", "lng"])
    alt_col = _first_present(truth, ["Altitude", "altitude", "alt"])
    out = truth.copy()
    enu = projector.transform_many(out[lat_col].to_numpy(), out[lon_col].to_numpy(), out[alt_col].to_numpy())
    out[["east_m", "north_m", "up_m"]] = enu
    return out


def rf_measurements_to_enu(
    rf: pd.DataFrame,
    projector: LocalENUProjector,
    default_std_m: float = 75.0,
) -> list[TrackingMeasurement]:
    """Convert RF localization rows to 2D ENU measurements."""

    lat_col = _first_present(rf, ["Latitude", "latitude", "lat"])
    lon_col = _first_present(rf, ["Longitude", "longitude", "lon", "lng"])
    time_col = _first_present(rf, ["Time", "time", "timestamp"])
    cep_col = _first_present(rf, ["CEP", "cep"], required=False)
    measurements: list[TrackingMeasurement] = []
    for _, row in rf.dropna(subset=[lat_col, lon_col, time_col]).iterrows():
        enu = projector.transform(float(row[lat_col]), float(row[lon_col]), projector.origin_altitude_m)
        std_m = float(row[cep_col]) if cep_col and np.isfinite(row[cep_col]) else float(default_std_m)
        covariance = np.diag([std_m**2, std_m**2])
        measurements.append(
            TrackingMeasurement(
                time_s=_coerce_time_s(row[time_col]),
                vector=enu[:2],
                covariance=covariance,
                source="rf",
            )
        )
    return measurements


def radar_measurements_to_enu(
    radar: pd.DataFrame,
    projector: LocalENUProjector,
    default_xy_std_m: float = 25.0,
    default_z_std_m: float = 35.0,
) -> list[TrackingMeasurement]:
    """Convert radar rows with LLA fields to 3D ENU measurements."""

    lat_col = _first_present(radar, ["lla.latitude", "lla.lat", "latitude", "lat"], required=False)
    lon_col = _first_present(radar, ["lla.longitude", "lla.lon", "longitude", "lon"], required=False)
    alt_col = _first_present(radar, ["lla.altitude", "lla.alt", "altitude", "alt"], required=False)
    time_col = _first_present(radar, ["gpsX", "time", "timestamp"], required=False)
    if not all([lat_col, lon_col, alt_col, time_col]):
        return []

    measurements: list[TrackingMeasurement] = []
    for _, row in radar.dropna(subset=[lat_col, lon_col, alt_col, time_col]).iterrows():
        enu = projector.transform(float(row[lat_col]), float(row[lon_col]), float(row[alt_col]))
        measurements.append(
            TrackingMeasurement(
                time_s=_coerce_time_s(row[time_col]),
                vector=enu,
                covariance=np.diag([default_xy_std_m**2, default_xy_std_m**2, default_z_std_m**2]),
                source="radar",
            )
        )
    return measurements


def _first_present(frame: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    columns = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    if required:
        raise KeyError(f"None of the expected columns are present: {candidates}")
    return None


def _coerce_time_s(value: Any) -> float:
    """Coerce numeric or datetime-like timestamps to seconds."""

    if isinstance(value, (int, float, np.number)):
        return float(value)
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return float(value)
    return float(parsed.timestamp())
