"""LOFO radar calibration diagnostics for selected Fortem rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from raft_uav.evaluation.metrics import nearest_time_indices, summarize_errors

POSITION_COLUMNS = ("east_m", "north_m", "up_m")


@dataclass(frozen=True)
class MeasurementTruthPairs:
    """Matched measurement/truth arrays for one or more flights."""

    measurement_times_s: np.ndarray
    measurement_positions_m: np.ndarray
    truth_times_s: np.ndarray
    truth_positions_m: np.ndarray


@dataclass(frozen=True)
class SpatialCalibration:
    """Rigid horizontal correction plus vertical bias."""

    yaw_rad: float
    offset_east_m: float
    offset_north_m: float
    offset_up_m: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


IDENTITY_CALIBRATION = SpatialCalibration(
    yaw_rad=0.0,
    offset_east_m=0.0,
    offset_north_m=0.0,
    offset_up_m=0.0,
)


def pair_measurements_to_truth(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    time_offset_s: float = 0.0,
    max_time_delta_s: float = 2.0,
) -> MeasurementTruthPairs:
    """Pair ENU measurements to nearest truth rows after applying a time offset."""

    _require_columns(measurements, ("time_s", *POSITION_COLUMNS), context="measurements")
    _require_columns(truth, ("time_s", *POSITION_COLUMNS), context="truth")
    measurement_times = measurements["time_s"].to_numpy(dtype=float) + float(time_offset_s)
    measurement_positions = measurements.loc[:, POSITION_COLUMNS].to_numpy(dtype=float)
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth.loc[:, POSITION_COLUMNS].to_numpy(dtype=float)
    if truth_times.size == 0:
        raise ValueError("truth must not be empty")
    if measurement_times.size == 0:
        return MeasurementTruthPairs(
            measurement_times_s=np.array([], dtype=float),
            measurement_positions_m=np.empty((0, 3), dtype=float),
            truth_times_s=np.array([], dtype=float),
            truth_positions_m=np.empty((0, 3), dtype=float),
        )

    indices = nearest_time_indices(truth_times, measurement_times)
    delta_t = np.abs(truth_times[indices] - measurement_times)
    finite = (
        np.isfinite(measurement_times)
        & np.isfinite(delta_t)
        & np.isfinite(measurement_positions).all(axis=1)
        & np.isfinite(truth_positions[indices]).all(axis=1)
        & (delta_t <= float(max_time_delta_s))
    )
    return MeasurementTruthPairs(
        measurement_times_s=measurement_times[finite],
        measurement_positions_m=measurement_positions[finite],
        truth_times_s=truth_times[indices][finite],
        truth_positions_m=truth_positions[indices][finite],
    )


def concatenate_pairs(pairs: Iterable[MeasurementTruthPairs]) -> MeasurementTruthPairs:
    """Concatenate matched measurement/truth pairs from multiple flights."""

    items = list(pairs)
    if not items:
        return MeasurementTruthPairs(
            measurement_times_s=np.array([], dtype=float),
            measurement_positions_m=np.empty((0, 3), dtype=float),
            truth_times_s=np.array([], dtype=float),
            truth_positions_m=np.empty((0, 3), dtype=float),
        )
    return MeasurementTruthPairs(
        measurement_times_s=np.concatenate([item.measurement_times_s for item in items]),
        measurement_positions_m=np.vstack([item.measurement_positions_m for item in items]),
        truth_times_s=np.concatenate([item.truth_times_s for item in items]),
        truth_positions_m=np.vstack([item.truth_positions_m for item in items]),
    )


def fit_constant_offset(pairs: MeasurementTruthPairs) -> SpatialCalibration:
    """Fit a robust constant ENU translation from measurement to truth."""

    _require_pair_count(pairs, minimum=1)
    residual = pairs.truth_positions_m - pairs.measurement_positions_m
    offset = np.nanmedian(residual, axis=0)
    return SpatialCalibration(
        yaw_rad=0.0,
        offset_east_m=float(offset[0]),
        offset_north_m=float(offset[1]),
        offset_up_m=float(offset[2]),
    )


def fit_yaw_offset_altitude(pairs: MeasurementTruthPairs) -> SpatialCalibration:
    """Fit horizontal yaw+translation and a robust vertical offset."""

    _require_pair_count(pairs, minimum=2)
    measured_xy = pairs.measurement_positions_m[:, :2]
    truth_xy = pairs.truth_positions_m[:, :2]
    measured_center = np.mean(measured_xy, axis=0)
    truth_center = np.mean(truth_xy, axis=0)
    measured_centered = measured_xy - measured_center
    truth_centered = truth_xy - truth_center
    covariance = measured_centered.T @ truth_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    translation = truth_center - measured_center @ rotation.T
    yaw_rad = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    up_offset = float(np.nanmedian(pairs.truth_positions_m[:, 2] - pairs.measurement_positions_m[:, 2]))
    return SpatialCalibration(
        yaw_rad=yaw_rad,
        offset_east_m=float(translation[0]),
        offset_north_m=float(translation[1]),
        offset_up_m=up_offset,
    )


def apply_spatial_calibration(
    measurements: pd.DataFrame,
    calibration: SpatialCalibration,
) -> pd.DataFrame:
    """Return a copy with corrected ENU columns and diagnostic bias columns."""

    _require_columns(measurements, POSITION_COLUMNS, context="measurements")
    out = measurements.copy()
    c = float(np.cos(calibration.yaw_rad))
    s = float(np.sin(calibration.yaw_rad))
    xy = out.loc[:, ("east_m", "north_m")].to_numpy(dtype=float)
    corrected_xy = xy @ np.array([[c, s], [-s, c]]) + np.array(
        [calibration.offset_east_m, calibration.offset_north_m]
    )
    out["east_m"] = corrected_xy[:, 0]
    out["north_m"] = corrected_xy[:, 1]
    out["up_m"] = out["up_m"].astype(float) + calibration.offset_up_m
    out["radar_calibration_yaw_rad"] = float(calibration.yaw_rad)
    out["radar_calibration_offset_east_m"] = float(calibration.offset_east_m)
    out["radar_calibration_offset_north_m"] = float(calibration.offset_north_m)
    out["radar_calibration_offset_up_m"] = float(calibration.offset_up_m)
    return out


def evaluate_calibrated_measurements(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    calibration: SpatialCalibration = IDENTITY_CALIBRATION,
    time_offset_s: float = 0.0,
    max_time_delta_s: float = 2.0,
) -> dict[str, float]:
    """Evaluate corrected measurements against nearest truth rows."""

    corrected = apply_spatial_calibration(measurements, calibration)
    pairs = pair_measurements_to_truth(
        corrected,
        truth,
        time_offset_s=time_offset_s,
        max_time_delta_s=max_time_delta_s,
    )
    if pairs.measurement_positions_m.size == 0:
        summary = summarize_errors(np.array([], dtype=float))
        summary.update({"matched_rows": 0.0, "truth_coverage_rows": 0.0})
        return summary
    errors = np.linalg.norm(pairs.measurement_positions_m - pairs.truth_positions_m, axis=1)
    summary = summarize_errors(errors)
    summary["matched_rows"] = float(errors.size)
    summary["truth_coverage_rows"] = float(np.unique(pairs.truth_times_s).size)
    summary["max_m"] = float(np.max(errors))
    return summary


def fit_time_offset(
    measurements_by_flight: dict[str, pd.DataFrame],
    truth_by_flight: dict[str, pd.DataFrame],
    offsets_s: Iterable[float],
    *,
    max_time_delta_s: float = 2.0,
    calibration: SpatialCalibration = IDENTITY_CALIBRATION,
) -> tuple[float, pd.DataFrame]:
    """Choose the offset minimizing aggregate 3D RMSE over training flights."""

    rows: list[dict[str, float]] = []
    for offset_s in offsets_s:
        errors: list[np.ndarray] = []
        matched_rows = 0
        for flight, measurements in measurements_by_flight.items():
            truth = truth_by_flight[flight]
            corrected = apply_spatial_calibration(measurements, calibration)
            pairs = pair_measurements_to_truth(
                corrected,
                truth,
                time_offset_s=float(offset_s),
                max_time_delta_s=max_time_delta_s,
            )
            if pairs.measurement_positions_m.size == 0:
                continue
            errors.append(np.linalg.norm(pairs.measurement_positions_m - pairs.truth_positions_m, axis=1))
            matched_rows += int(pairs.measurement_positions_m.shape[0])
        all_errors = np.concatenate(errors) if errors else np.array([], dtype=float)
        summary = summarize_errors(all_errors)
        rows.append(
            {
                "time_offset_s": float(offset_s),
                "matched_rows": float(matched_rows),
                "rmse_m": _finite_or_nan(summary["rmse_m"]),
                "mae_m": _finite_or_nan(summary["mae_m"]),
                "p95_m": _finite_or_nan(summary["p95_m"]),
            }
        )
    sweep = pd.DataFrame(rows)
    valid = sweep[np.isfinite(sweep["rmse_m"].to_numpy(dtype=float))]
    if valid.empty:
        raise RuntimeError("no finite time-offset candidates")
    best_index = valid["rmse_m"].astype(float).idxmin()
    return float(valid.loc[best_index, "time_offset_s"]), sweep


def _finite_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _require_pair_count(pairs: MeasurementTruthPairs, *, minimum: int) -> None:
    count = int(pairs.measurement_positions_m.shape[0])
    if count < minimum:
        raise ValueError(f"need at least {minimum} matched pairs, got {count}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {', '.join(missing)}")
