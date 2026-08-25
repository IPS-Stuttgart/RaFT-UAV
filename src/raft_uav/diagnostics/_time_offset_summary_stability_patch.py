"""Keep time-offset error summaries finite for finite inputs."""

from __future__ import annotations

from functools import wraps
from importlib import import_module

import numpy as np


_time_offset = import_module("raft_uav.diagnostics.time_offset")
_PATCH_MARKER = "_raft_uav_time_offset_summary_stability_patch_applied"
_ORIGINAL_SUMMARIZE_ERRORS = _time_offset.summarize_errors


def _scaled_error_statistics(errors: np.ndarray) -> dict[str, float]:
    """Compute summary statistics after scaling values into [-1, 1]."""

    scale = float(np.max(np.abs(errors)))
    if scale == 0.0:
        return {
            "mean_error_m": 0.0,
            "std_error_m": 0.0,
            "rmse_error_m": 0.0,
            "p50_error_m": 0.0,
            "p95_error_m": 0.0,
            "max_error_m": 0.0,
        }

    normalized = errors / scale
    return {
        "mean_error_m": float(scale * np.mean(normalized)),
        "std_error_m": float(scale * np.std(normalized)),
        "rmse_error_m": float(
            scale * np.sqrt(np.mean(np.square(normalized)))
        ),
        "p50_error_m": float(scale * np.percentile(normalized, 50.0)),
        "p95_error_m": float(scale * np.percentile(normalized, 95.0)),
        "max_error_m": float(scale * np.max(normalized)),
    }


@wraps(_ORIGINAL_SUMMARIZE_ERRORS)
def summarize_errors(
    *,
    tau_s: float,
    candidate_count: int,
    selected_count: int,
    matched_count: int,
    errors_m: np.ndarray,
) -> dict[str, float | int]:
    """Preserve count metrics while avoiding overflow in finite error statistics."""

    with np.errstate(over="ignore", invalid="ignore"):
        summary = _ORIGINAL_SUMMARIZE_ERRORS(
            tau_s=tau_s,
            candidate_count=candidate_count,
            selected_count=selected_count,
            matched_count=matched_count,
            errors_m=errors_m,
        )

    errors_masked = np.ma.asarray(errors_m, dtype=float).reshape(-1)
    errors = np.asarray(errors_masked.filled(np.nan), dtype=float)
    finite_errors = errors[np.isfinite(errors)]
    if finite_errors.size:
        summary.update(_scaled_error_statistics(finite_errors))
    return summary


def install() -> None:
    """Install stable time-offset error summaries once per interpreter."""

    if getattr(_time_offset, _PATCH_MARKER, False):
        return
    _time_offset.summarize_errors = summarize_errors
    legacy = getattr(_time_offset, "_legacy", None)
    if legacy is not None:
        legacy.summarize_errors = summarize_errors
    setattr(_time_offset, _PATCH_MARKER, True)


install()
