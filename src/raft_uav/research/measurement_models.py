"""Measurement-model calibration helpers for radar/RF experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearRadarBiasModel:
    """Linear residual-bias model in radar geometry features."""

    feature_names: tuple[str, ...]
    coefficients: np.ndarray

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = radar_geometry_feature_matrix(frame, self.feature_names)
        return x @ self.coefficients


def enu_covariance_from_range_az_el(
    range_m: float,
    azimuth_rad: float,
    elevation_rad: float,
    *,
    range_std_m: float,
    azimuth_std_rad: float,
    elevation_std_rad: float,
    min_std_m: float = 1.0,
) -> np.ndarray:
    """Transform native polar radar uncertainty into ENU covariance.

    The coordinate convention is east = r cos(el) sin(az), north = r cos(el)
    cos(az), up = r sin(el), with azimuth measured clockwise from north.
    """

    r = _validate_finite_nonnegative(range_m, "range_m")
    az = float(azimuth_rad)
    el = float(elevation_rad)
    if not np.isfinite(az):
        raise ValueError("azimuth_rad must be finite")
    if not np.isfinite(el):
        raise ValueError("elevation_rad must be finite")
    range_std = _validate_finite_nonnegative(range_std_m, "range_std_m")
    azimuth_std = _validate_finite_nonnegative(azimuth_std_rad, "azimuth_std_rad")
    elevation_std = _validate_finite_nonnegative(elevation_std_rad, "elevation_std_rad")
    min_std = _validate_finite_nonnegative(min_std_m, "min_std_m")

    ce = float(np.cos(el))
    se = float(np.sin(el))
    ca = float(np.cos(az))
    sa = float(np.sin(az))
    jacobian = np.array(
        [
            [ce * sa, r * ce * ca, -r * se * sa],
            [ce * ca, -r * ce * sa, -r * se * ca],
            [se, 0.0, r * ce],
        ],
        dtype=float,
    )
    native = np.diag([range_std**2, azimuth_std**2, elevation_std**2])
    covariance = jacobian @ native @ jacobian.T
    if min_std > 0.0:
        covariance = covariance + np.eye(3) * min_std**2
    return 0.5 * (covariance + covariance.T)


def covariance_columns_from_native_radar(
    frame: pd.DataFrame,
    *,
    range_std_m: float,
    azimuth_std_deg: float,
    elevation_std_deg: float,
    min_std_m: float = 1.0,
) -> pd.DataFrame:
    """Append ``association_cov_*`` columns from native radar coordinates."""

    required = {"range_m", "azimuth_rad", "elevation_rad"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"radar frame is missing native-coordinate columns: {sorted(missing)}")
    out = frame.copy()
    az_std = np.deg2rad(float(azimuth_std_deg))
    el_std = np.deg2rad(float(elevation_std_deg))
    covariances = [
        enu_covariance_from_range_az_el(
            row.range_m,
            row.azimuth_rad,
            row.elevation_rad,
            range_std_m=range_std_m,
            azimuth_std_rad=az_std,
            elevation_std_rad=el_std,
            min_std_m=min_std_m,
        )
        for row in out.itertuples(index=False)
    ]
    out["association_cov_ee"] = [float(c[0, 0]) for c in covariances]
    out["association_cov_nn"] = [float(c[1, 1]) for c in covariances]
    out["association_cov_uu"] = [float(c[2, 2]) for c in covariances]
    out["association_cov_en"] = [float(c[0, 1]) for c in covariances]
    out["association_cov_eu"] = [float(c[0, 2]) for c in covariances]
    out["association_cov_nu"] = [float(c[1, 2]) for c in covariances]
    out["association_covariance_mode"] = "native-range-az-el"
    return out


def fit_linear_radar_bias_model(
    examples: pd.DataFrame,
    *,
    residual_columns: Sequence[str] = ("residual_east_m", "residual_north_m", "residual_up_m"),
    feature_names: Sequence[str] = (
        "intercept",
        "range_m",
        "sin_azimuth",
        "cos_azimuth",
        "elevation_rad",
        "radial_velocity_mps",
    ),
    ridge_lambda: float = 1.0,
) -> LinearRadarBiasModel:
    """Fit a LOFO-safe linear radar spatial-bias model."""

    resolved_feature_names = tuple(feature_names)
    x = radar_geometry_feature_matrix(examples, resolved_feature_names)
    y = examples.loc[:, list(residual_columns)].to_numpy(dtype=float)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if not np.any(keep):
        raise ValueError("no finite bias-training examples")
    x = x[keep]
    y = y[keep]
    penalty = float(ridge_lambda) * np.eye(x.shape[1])
    for index, name in enumerate(resolved_feature_names):
        if name == "intercept":
            penalty[index, index] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return LinearRadarBiasModel(resolved_feature_names, coefficients)


def apply_linear_radar_bias_model(frame: pd.DataFrame, model: LinearRadarBiasModel) -> pd.DataFrame:
    """Subtract predicted radar bias from ENU position columns."""

    out = frame.copy()
    bias = model.predict(out)
    for idx, column in enumerate(("east_m", "north_m", "up_m")):
        out[f"bias_{column}"] = bias[:, idx]
        out[column] = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float) - bias[:, idx]
    out["bias_model"] = "linear-geometry"
    return out


def radar_geometry_feature_matrix(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name in feature_names:
        if name == "intercept":
            columns.append(np.ones(len(frame)))
        elif name == "sin_azimuth":
            columns.append(np.sin(_numeric(frame, "azimuth_rad")))
        elif name == "cos_azimuth":
            columns.append(np.cos(_numeric(frame, "azimuth_rad")))
        elif name == "log1p_range_m":
            columns.append(np.log1p(np.maximum(_numeric(frame, "range_m"), 0.0)))
        elif name in frame.columns:
            columns.append(_numeric(frame, name))
        else:
            columns.append(np.full(len(frame), np.nan))
    return np.column_stack(columns) if columns else np.empty((len(frame), 0))


def rf_quality_covariance_scale(
    rf: pd.DataFrame,
    *,
    base_scale: float = 1.0,
    missing_penalty: float = 2.0,
) -> np.ndarray:
    """Return heuristic RF covariance scales from available quality columns."""

    if rf.empty:
        return np.empty(0)
    scale = np.full(len(rf), float(base_scale), dtype=float)
    for column in ("rssi", "snr", "quality", "num_receivers", "num_anchors"):
        if column not in rf.columns:
            continue
        values = pd.to_numeric(rf[column], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        scale[missing] *= float(missing_penalty)
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = np.percentile(finite, [10, 90])
            if hi > lo:
                normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
                scale *= np.where(np.isfinite(normalized), 1.5 - normalized, 1.0)
    return np.clip(scale, 0.25, 10.0)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _validate_finite_nonnegative(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number
