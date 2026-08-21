"""Reject negative values passed to scalar position-error summaries."""

from __future__ import annotations

import numpy as np

from . import metrics as _metrics

_ORIGINAL_SUMMARIZE_ERRORS = _metrics.summarize_errors


def _summarize_errors(errors_m: np.ndarray) -> dict[str, float | None]:
    """Preserve metric semantics by rejecting impossible negative magnitudes."""

    # Run the established metric-input validation first.  In particular, the
    # active metrics wrapper rejects cyclic object containers before NumPy is
    # asked to coerce them to floating point.  Performing the dtype conversion
    # first can recurse inside NumPy and segfault for self-referential object
    # arrays instead of raising the repository's documented ValueError.
    summary = _ORIGINAL_SUMMARIZE_ERRORS(errors_m)

    errors_masked = np.ma.asarray(errors_m, dtype=float).reshape(-1)
    errors = np.asarray(errors_masked.filled(np.nan), dtype=float)
    finite_errors = errors[np.isfinite(errors)]
    if bool(np.any(finite_errors < 0.0)):
        raise ValueError("errors_m must contain only non-negative values")
    return summary


_metrics.summarize_errors = _summarize_errors
