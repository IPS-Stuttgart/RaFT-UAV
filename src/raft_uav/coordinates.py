"""Coordinate transforms for AERPAW-style geodetic tracking data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class LocalENUProjector:
    """Project WGS84 latitude/longitude/altitude into a local ENU frame."""

    origin_latitude_deg: float
    origin_longitude_deg: float
    origin_altitude_m: float

    def __post_init__(self) -> None:
        lon = self.origin_longitude_deg
        lat = self.origin_latitude_deg
        alt = self.origin_altitude_m
        x0, y0, z0 = _geodetic_to_ecef(lat, lon, alt)
        object.__setattr__(self, "_origin_ecef", np.array([x0, y0, z0], dtype=float))

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
        rotation = np.array(
            [
                [-sin_lon, cos_lon, 0.0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ],
            dtype=float,
        )
        object.__setattr__(self, "_ecef_to_enu_rotation", rotation)

    def transform(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
    ) -> np.ndarray:
        """Transform one WGS84 coordinate to local ENU meters."""

        x, y, z = _geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)
        delta = np.array([x, y, z], dtype=float) - self._origin_ecef
        return self._ecef_to_enu_rotation @ delta

    def transform_many(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
        altitude_m: np.ndarray,
    ) -> np.ndarray:
        """Transform many WGS84 coordinates to an ``(n, 3)`` ENU array."""

        lon = np.asarray(longitude_deg, dtype=float)
        lat = np.asarray(latitude_deg, dtype=float)
        alt = np.asarray(altitude_m, dtype=float)
        x, y, z = _geodetic_to_ecef(lat, lon, alt)
        ecef = np.column_stack([x, y, z])
        delta = ecef - self._origin_ecef.reshape(1, 3)
        return delta @ self._ecef_to_enu_rotation.T


def _geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m):
    """Convert WGS84 geodetic coordinates to ECEF meters."""

    lat = np.deg2rad(latitude_deg)
    lon = np.deg2rad(longitude_deg)
    alt = np.asarray(altitude_m, dtype=float)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    radius = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (radius + alt) * cos_lat * np.cos(lon)
    y = (radius + alt) * cos_lat * np.sin(lon)
    z = (radius * (1.0 - WGS84_E2) + alt) * sin_lat
    return x, y, z
