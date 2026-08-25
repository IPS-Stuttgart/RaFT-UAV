"""Keep radar-geometry distances and summaries finite for finite inputs."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Iterable

import numpy as np
import pandas as pd


_radar_geometry = import_module("raft_uav.diagnostics.radar_geometry")
_PATCH_MARKER = "_raft_uav_radar_geometry_stability_patch_applied"
_ORIGINAL_BUILD_AUDIT_FRAME = _radar_geometry.build_radar_geometry_audit_frame


def _stable_row_norm(values: np.ndarray) -> np.ndarray:
    """Return overflow-stable Euclidean norms along the final axis."""

    array = np.asarray(values, dtype=float)
    return np.hypot.reduce(np.abs(array), axis=1)


def _stable_series_summary(series: pd.Series) -> dict[str, float | int | None]:
    """Summarize finite values after scaling them into a safe numeric range."""

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p95": None,
            "max": None,
        }

    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return {
            "count": int(values.size),
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    normalized = values / scale
    return {
        "count": int(values.size),
        "mean": float(scale * np.mean(normalized)),
        "std": float(scale * np.std(normalized, ddof=1)) if values.size > 1 else 0.0,
        "p50": float(scale * np.percentile(normalized, 50.0)),
        "p95": float(scale * np.percentile(normalized, 95.0)),
        "max": float(scale * np.max(normalized)),
    }


@wraps(_ORIGINAL_BUILD_AUDIT_FRAME)
def build_radar_geometry_audit_frame(
    radar: pd.DataFrame,
    *,
    radar_origin_enu_m: Iterable[float] | np.ndarray = (0.0, 0.0, 0.0),
    azimuth_convention: str = "north-clockwise",
) -> pd.DataFrame:
    """Recompute derived radar-geometry norms without squaring large values."""

    with np.errstate(over="ignore", invalid="ignore"):
        out = _ORIGINAL_BUILD_AUDIT_FRAME(
            radar,
            radar_origin_enu_m=radar_origin_enu_m,
            azimuth_convention=azimuth_convention,
        )

    delta = out[
        [
            "geometry_delta_east_m",
            "geometry_delta_north_m",
            "geometry_delta_up_m",
        ]
    ].to_numpy(dtype=float)
    lla_from_radar_origin = (
        out[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
        - out[
            [
                "radar_origin_east_m",
                "radar_origin_north_m",
                "radar_origin_up_m",
            ]
        ].to_numpy(dtype=float)
    )
    lla_slant_range = _stable_row_norm(lla_from_radar_origin)

    out["geometry_delta_horizontal_m"] = _stable_row_norm(delta[:, :2])
    out["geometry_delta_3d_m"] = _stable_row_norm(delta)
    out["lla_slant_range_from_radar_origin_m"] = lla_slant_range
    out["range_minus_lla_slant_range_m"] = (
        pd.to_numeric(out["range_m"], errors="coerce").to_numpy(dtype=float)
        - lla_slant_range
    )
    return out


def install() -> None:
    """Install stable radar-geometry computations once per interpreter."""

    if getattr(_radar_geometry, _PATCH_MARKER, False):
        return
    _radar_geometry.build_radar_geometry_audit_frame = build_radar_geometry_audit_frame
    _radar_geometry._series_summary = _stable_series_summary
    setattr(_radar_geometry, _PATCH_MARKER, True)


install()
