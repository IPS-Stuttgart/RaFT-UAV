"""Reject complex MOT match-distance scalars hidden by object containers."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_raft_uav_rejects_complex_mot_match_distance"


def _is_complex_scalar(value: object) -> bool:
    """Return whether one scalar-like value has a complex numeric dtype."""

    if np.ma.is_masked(value):
        return False
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.iscomplexobj(array):
        return True
    try:
        scalar = array.item()
    except (TypeError, ValueError):
        return False
    return bool(np.iscomplexobj(scalar))


def install() -> None:
    """Install strict complex-value rejection at the MOT metric boundary."""

    from raft_uav.mmuad import mot

    original: Callable[[Any], float] = mot._validated_match_distance_m
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def validated(value: Any) -> float:
        if _is_complex_scalar(value):
            raise ValueError("match_distance_m must be finite and nonnegative")
        return original(value)

    setattr(validated, _PATCH_MARKER, True)
    setattr(mot, "_validated_match_distance_m", validated)
