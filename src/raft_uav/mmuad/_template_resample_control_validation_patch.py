"""Reject recursively boxed Track 5 template-resampling controls."""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCH_ATTRIBUTE = "_raft_uav_recursive_template_control_guard"


def _normalize_optional_nonnegative_float(value: Any, *, field: str) -> float | None:
    """Return a finite non-negative scalar without lossy container coercion."""

    if value is None:
        return None
    message = f"{field} must be a finite non-negative number"
    seen: set[int] = set()
    while isinstance(value, np.ndarray):
        if np.ma.is_masked(value) or value.ndim != 0:
            raise ValueError(message)
        identity = id(value)
        if identity in seen:
            raise ValueError(message)
        seen.add(identity)
        value = value.item()
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(message)
    return numeric


def install() -> None:
    """Install the scalar validator at the active compatibility boundary."""

    from raft_uav.mmuad import track5_template_resample as module

    current = module._normalize_optional_nonnegative_float
    if getattr(current, _PATCH_ATTRIBUTE, False):
        return
    setattr(_normalize_optional_nonnegative_float, _PATCH_ATTRIBUTE, True)
    module._normalize_optional_nonnegative_float = _normalize_optional_nonnegative_float
