"""Preserve fifth-wave block-bootstrap semantics after stability hardening."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module

import numpy as np

from . import _fifth_wave_bootstrap_capture as _capture


_fifth_wave = import_module("raft_uav.evaluation.fifth_wave_diagnostics")
_IMPL = _fifth_wave._IMPL
_PATCH_MARKER = "_raft_uav_fifth_wave_bootstrap_compatibility_patch_applied"
_HOMOGENEOUS_METRICS = frozenset({"mean", "median", "rmse", "mae", "p95"})


def _finite_interval(interval: object) -> bool:
    """Return whether every numeric bootstrap endpoint is finite."""

    return bool(
        np.isfinite(float(interval.estimate))
        and np.isfinite(float(interval.lower))
        and np.isfinite(float(interval.upper))
    )


def block_bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    *,
    metric="mean",
    block_size: int = 50,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
):
    """Run the original block bootstrap and scale only after finite overflow."""

    with np.errstate(over="ignore", invalid="ignore"):
        direct = _capture.ORIGINAL_BLOCK_BOOTSTRAP_INTERVAL(
            values,
            metric=metric,
            block_size=block_size,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )
    if _finite_interval(direct):
        return direct

    array = _IMPL._finite_vector(values)
    if (
        array.size == 0
        or callable(metric)
        or metric not in _HOMOGENEOUS_METRICS
        or not np.isfinite(array).all()
    ):
        return direct

    scale = float(np.max(np.abs(array)))
    if scale == 0.0 or not np.isfinite(scale):
        return direct

    with np.errstate(over="ignore", invalid="ignore"):
        normalized = _capture.ORIGINAL_BLOCK_BOOTSTRAP_INTERVAL(
            array / scale,
            metric=metric,
            block_size=block_size,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )
        estimate = float(scale * normalized.estimate)
        lower = float(scale * normalized.lower)
        upper = float(scale * normalized.upper)

    return _IMPL.BootstrapInterval(
        metric=normalized.metric,
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence=float(normalized.confidence),
        samples=int(normalized.samples),
        block_size=int(normalized.block_size),
        resamples=int(normalized.resamples),
    )


def install() -> None:
    """Restore the original sampling semantics behind the public validator."""

    if getattr(_fifth_wave, _PATCH_MARKER, False):
        return
    _fifth_wave._ORIGINAL_BLOCK_BOOTSTRAP_INTERVAL = block_bootstrap_interval
    setattr(_fifth_wave, _PATCH_MARKER, True)
    setattr(_IMPL, _PATCH_MARKER, True)


install()
