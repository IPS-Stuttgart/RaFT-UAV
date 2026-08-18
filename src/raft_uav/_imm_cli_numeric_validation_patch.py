"""Reject non-finite IMM experiment controls before dataset I/O."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
import math
from typing import Any, Callable


_imm_cli = import_module("raft_uav.imm_cli")
_PATCH_MARKER = "_raft_uav_imm_cli_numeric_validation_patch_applied"
_ORIGINAL_RUN_EXPERIMENT: Callable[..., int] = _imm_cli.run_experiment


def _require_finite(value: object, *, message: str) -> None:
    """Reject scalar values that coerce to NaN or infinity.

    Non-numeric payloads are intentionally left to the maintained implementation
    so this compatibility guard only closes the non-finite comparison loophole.
    """

    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return
    if not math.isfinite(number):
        raise ValueError(message)


@wraps(_ORIGINAL_RUN_EXPERIMENT)
def _run_experiment(*args: Any, **kwargs: Any) -> int:
    """Validate finite controls whose legacy inequality checks admit NaN/Inf."""

    _require_finite(
        kwargs.get("imm_mode_switch_time_constant", 20.0),
        message="imm_mode_switch_time_constant must be positive and finite",
    )
    if kwargs.get("smoother", "none") in {"fixed-lag", "fixed-lag-map"}:
        _require_finite(
            kwargs.get("smoother_lag_s", 20.0),
            message="smoother_lag_s must be nonnegative and finite for fixed-lag smoothing",
        )
    _require_finite(
        kwargs.get("rf_inflation_alpha", 1.0),
        message="inflation alphas must be positive and finite",
    )
    _require_finite(
        kwargs.get("radar_inflation_alpha", 1.0),
        message="inflation alphas must be positive and finite",
    )
    return _ORIGINAL_RUN_EXPERIMENT(*args, **kwargs)


def install() -> None:
    """Install the IMM experiment input guard once per interpreter."""

    if getattr(_imm_cli, _PATCH_MARKER, False):
        return
    _imm_cli.run_experiment = _run_experiment
    setattr(_imm_cli, _PATCH_MARKER, True)
