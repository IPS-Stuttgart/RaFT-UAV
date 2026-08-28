"""Preserve binary rounding when fifth-wave means need overflow fallback."""

from __future__ import annotations

import numpy as np

from . import _fifth_wave_error_stability_patch as _stability


_PATCH_MARKER = "_raft_uav_fifth_wave_mean_rounding_patch_applied"


def _stable_mean(values: np.ndarray) -> float:
    """Return a finite mean without adding avoidable non-binary scaling error."""

    array = np.asarray(values, dtype=float).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        direct = float(np.mean(array))
    if array.size == 0 or np.isfinite(direct) or not np.isfinite(array).all():
        return direct

    max_abs = float(np.max(np.abs(array)))
    if max_abs == 0.0:
        return direct

    # Scale by an exactly representable power of two.  Dividing by an arbitrary
    # data maximum (the previous fallback) can introduce a rounding error before
    # the mean is rescaled; for [8e307, 1e308] that produced
    # 8.999999999999999e307 instead of the representable 9e307.  A binary
    # exponent shift keeps the fallback overflow-safe without that extra source
    # of rounding error.
    _, exponent = np.frexp(max_abs)
    shift = int(exponent) - 1
    scale = float(np.ldexp(1.0, shift))
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_mean = float(np.mean(array / scale))
        return float(np.ldexp(normalized_mean, shift))


def install() -> None:
    """Install the exact-scaling mean fallback used by fifth-wave metrics."""

    if getattr(_stability, _PATCH_MARKER, False):
        return
    _stability._stable_mean = _stable_mean
    setattr(_stability, _PATCH_MARKER, True)


install()
