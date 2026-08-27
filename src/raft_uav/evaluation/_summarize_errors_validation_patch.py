"""Validate scalar position-error summaries and avoid numeric overflow."""

from __future__ import annotations

import numpy as np

from . import metrics as _metrics

_ORIGINAL_SUMMARIZE_ERRORS = _metrics.summarize_errors
_SUMMARY_KEYS = (
    "mean_m",
    "std_m",
    "rmse_m",
    "mae_m",
    "p50_m",
    "p95_m",
    "max_m",
)


def _scaled_error_statistics(errors: np.ndarray) -> dict[str, float]:
    """Compute non-negative error statistics on a normalized scale."""

    scale = float(np.max(errors))
    if scale == 0.0:
        return {
            "mean_m": 0.0,
            "std_m": 0.0,
            "rmse_m": 0.0,
            "mae_m": 0.0,
            "p50_m": 0.0,
            "p95_m": 0.0,
            "max_m": 0.0,
        }

    normalized = errors / scale
    mean_m = float(scale * np.mean(normalized))
    return {
        "mean_m": mean_m,
        "std_m": float(scale * np.std(normalized)),
        "rmse_m": float(scale * np.sqrt(np.mean(np.square(normalized)))),
        "mae_m": mean_m,
        "p50_m": float(scale * np.percentile(normalized, 50.0)),
        "p95_m": float(scale * np.percentile(normalized, 95.0)),
        "max_m": scale,
    }


def _summarize_errors(errors_m: np.ndarray) -> dict[str, float | None]:
    """Reject negative magnitudes and scale only when direct arithmetic overflows."""

    # Run the established metric-input validation first. In particular, the
    # active metrics wrapper rejects cyclic object containers before NumPy is
    # asked to coerce them to floating point. Performing the dtype conversion
    # first can recurse inside NumPy and segfault for self-referential object
    # arrays instead of raising the repository's documented ValueError.
    with np.errstate(over="ignore", invalid="ignore"):
        summary = _ORIGINAL_SUMMARIZE_ERRORS(errors_m)

    errors_masked = np.ma.asarray(errors_m, dtype=float).reshape(-1)
    errors = np.asarray(errors_masked.filled(np.nan), dtype=float)
    finite_errors = errors[np.isfinite(errors)]
    if bool(np.any(finite_errors < 0.0)):
        raise ValueError("errors_m must contain only non-negative values")

    # Preserve the established bit-for-bit output for ordinary values. Scaling
    # changes harmless last-bit rounding, so use it only when a direct statistic
    # is non-finite despite all retained inputs being finite.
    if finite_errors.size and not all(
        summary[key] is not None and np.isfinite(float(summary[key]))
        for key in _SUMMARY_KEYS
    ):
        summary.update(_scaled_error_statistics(finite_errors))
    return summary


_metrics.summarize_errors = _summarize_errors
