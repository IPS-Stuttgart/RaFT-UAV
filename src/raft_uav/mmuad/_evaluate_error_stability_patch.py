"""Keep MMUAD submission error calculations stable for large finite values."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd


_evaluate = import_module("raft_uav.mmuad.evaluate")
_PATCH_MARKER = "_error_stability_patch_applied"


def _scaled_euclidean_norm(values: Any) -> float:
    """Return a Euclidean norm without squaring large unscaled coordinates."""

    vector = np.asarray(values, dtype=float)
    if vector.size == 0:
        return 0.0
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.linalg.norm(vector))
    normalized = vector / scale
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * np.sqrt(np.sum(np.square(normalized))))


def _scaled_mean(values: Any) -> float | None:
    """Return a mean that stays finite whenever the true finite mean is representable."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return None
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.mean(array))
    with np.errstate(over="ignore", invalid="ignore"):
        return float(scale * np.mean(array / scale))


def _scaled_error_statistics(values: Any) -> dict[str, float | None]:
    """Summarize errors after scaling finite values into a safe numeric range."""

    array = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(array)
    if not bool(finite.any()):
        return {"mean": None, "rmse": None, "p95": None, "max": None}

    non_nan = array[~np.isnan(array)]
    if not bool(np.isfinite(non_nan).all()):
        with np.errstate(over="ignore", invalid="ignore"):
            return {
                "mean": float(np.nanmean(array)),
                "rmse": float(np.sqrt(np.nanmean(np.square(array)))),
                "p95": float(np.nanpercentile(array, 95.0)),
                "max": float(np.nanmax(array)),
            }

    scale = float(np.max(np.abs(non_nan)))
    if scale == 0.0:
        return {"mean": 0.0, "rmse": 0.0, "p95": 0.0, "max": 0.0}

    normalized = non_nan / scale
    with np.errstate(over="ignore", invalid="ignore"):
        return {
            "mean": float(scale * np.mean(normalized)),
            "rmse": float(scale * np.sqrt(np.mean(np.square(normalized)))),
            "p95": float(scale * np.percentile(normalized, 95.0)),
            "max": float(scale * np.max(normalized)),
        }


def _matched_prediction_row(
    *,
    sequence_id: str,
    prediction: pd.Series,
    truth: pd.Series,
) -> dict[str, Any]:
    """Build one matched row using overflow-safe Euclidean distances."""

    prediction_position = np.array(
        [float(prediction[axis]) for axis in ("x_m", "y_m", "z_m")],
        dtype=float,
    )
    truth_position = np.array(
        [float(truth[axis]) for axis in ("x_m", "y_m", "z_m")],
        dtype=float,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        error = prediction_position - truth_position
    return {
        "sequence_id": sequence_id,
        "time_s": float(prediction["time_s"]),
        "track_id": _evaluate._valid_track_id_text(
            prediction.get("track_id", "uav0")
        )
        or "uav0",
        "truth_time_s": float(truth["time_s"]),
        "truth_track_id": _evaluate._truth_track_id(truth),
        "time_delta_s": abs(float(truth["time_s"]) - float(prediction["time_s"])),
        "matched": True,
        "unmatched_reason": "",
        "error_2d_m": _scaled_euclidean_norm(error[:2]),
        "error_3d_m": _scaled_euclidean_norm(error),
        "vertical_error_m": float(error[2]),
    }


def _error_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute pooled error metrics without overflowing finite magnitudes."""

    if frame.empty or "error_3d_m" not in frame.columns:
        return {"count": 0}
    err3 = frame["error_3d_m"].to_numpy(float)
    err2 = frame["error_2d_m"].to_numpy(float)
    stats3 = _scaled_error_statistics(err3)
    stats2 = _scaled_error_statistics(err2)
    return {
        "count": int(np.isfinite(err3).sum()),
        "mean_3d_m": stats3["mean"],
        "rmse_3d_m": stats3["rmse"],
        "p95_3d_m": stats3["p95"],
        "max_3d_m": stats3["max"],
        "ade_3d_m": stats3["mean"],
        "fde_3d_m": _evaluate._mean_final_error(frame, "error_3d_m"),
        "mean_2d_m": stats2["mean"],
        "p95_2d_m": stats2["p95"],
        "max_2d_m": stats2["max"],
        "fde_2d_m": _evaluate._mean_final_error(frame, "error_2d_m"),
    }


def install() -> None:
    """Install stable matched-row and pooled-error calculations once."""

    if getattr(_evaluate, _PATCH_MARKER, False):
        return
    _evaluate._matched_prediction_row = _matched_prediction_row
    _evaluate._error_metrics = _error_metrics
    implementation = getattr(_evaluate, "_IMPL", None)
    if implementation is not None:
        implementation._matched_prediction_row = _matched_prediction_row
        implementation._error_metrics = _error_metrics
    setattr(_evaluate, _PATCH_MARKER, True)


install()
