"""Preserve fifth-wave block-bootstrap semantics after stability hardening."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module

import numpy as np

from . import _fifth_wave_error_stability_patch as _stability


_fifth_wave = import_module("raft_uav.evaluation.fifth_wave_diagnostics")
_IMPL = _fifth_wave._IMPL
_PATCH_MARKER = "_raft_uav_fifth_wave_bootstrap_core_patch_applied"


def _percentile_bounds(
    draws: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    """Return the original percentile bounds with a finite overflow fallback."""

    alpha = 1.0 - float(confidence)
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    with np.errstate(over="ignore", invalid="ignore"):
        bounds = np.percentile(draws, [lower_q, upper_q])
    if np.isfinite(bounds).all():
        return float(bounds[0]), float(bounds[1])
    return (
        _stability._stable_percentile(draws, lower_q),
        _stability._stable_percentile(draws, upper_q),
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
    """Run the established non-overlapping block bootstrap with stable metrics."""

    x = _IMPL._finite_vector(values)
    if x.size == 0:
        return _IMPL.BootstrapInterval(
            _IMPL._metric_name(metric),
            np.nan,
            np.nan,
            np.nan,
            confidence,
            0,
            block_size,
            resamples,
        )
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    fn = _stability._metric_function(metric)
    estimate = fn(x)
    blocks = _IMPL._contiguous_blocks(x, min(int(block_size), x.size))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(resamples), dtype=float)
    for draw_index in range(int(resamples)):
        sampled = [
            blocks[int(block_index)]
            for block_index in rng.integers(0, len(blocks), size=len(blocks))
        ]
        draws[draw_index] = fn(np.concatenate(sampled)[: x.size])

    lower, upper = _percentile_bounds(draws, confidence=float(confidence))
    return _IMPL.BootstrapInterval(
        metric=_IMPL._metric_name(metric),
        estimate=float(estimate),
        lower=lower,
        upper=upper,
        confidence=float(confidence),
        samples=int(x.size),
        block_size=int(block_size),
        resamples=int(resamples),
    )


def install() -> None:
    """Install the stable core behind the existing public validation wrapper."""

    if getattr(_fifth_wave, _PATCH_MARKER, False):
        return
    _fifth_wave._ORIGINAL_BLOCK_BOOTSTRAP_INTERVAL = block_bootstrap_interval
    setattr(_fifth_wave, _PATCH_MARKER, True)
    setattr(_IMPL, _PATCH_MARKER, True)


install()
