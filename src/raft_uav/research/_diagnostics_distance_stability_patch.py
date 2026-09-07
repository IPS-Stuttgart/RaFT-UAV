"""Keep research diagnostic distances stable for finite extreme coordinates."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_raft_uav_research_diagnostic_distance_stability"
_GUARD_MARKER = "_raft_uav_research_diagnostic_distance_errstate_guard"


class _StableLinalgProxy:
    """Delegate linalg calls while repairing overflowed Euclidean norms."""

    def __init__(self, original: Any) -> None:
        self._original = original

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def norm(
        self,
        values: Any,
        ord: Any = None,
        axis: Any = None,
        keepdims: bool = False,
    ) -> Any:
        """Preserve ordinary norms and repair only finite-input overflow."""

        with np.errstate(over="ignore", invalid="ignore"):
            direct = self._original.norm(
                values,
                ord=ord,
                axis=axis,
                keepdims=keepdims,
            )
        if ord is not None:
            return direct

        array = np.asarray(values, dtype=float)
        direct_array = np.asarray(direct, dtype=float)
        if bool(np.isfinite(direct_array).all()):
            return direct

        if axis is None:
            if not bool(np.isfinite(array).all()):
                return direct
            with np.errstate(over="ignore", invalid="ignore"):
                stable = np.hypot.reduce(np.abs(array).reshape(-1))
            if keepdims:
                return np.asarray(stable).reshape((1,) * array.ndim)
            return stable

        if isinstance(axis, tuple):
            if len(axis) != 1:
                return direct
            axis = axis[0]
        if not isinstance(axis, (int, np.integer)):
            return direct

        normalized_axis = int(axis)
        reduced_direct = (
            np.squeeze(direct_array, axis=normalized_axis)
            if keepdims
            else direct_array
        )
        finite_inputs = np.isfinite(array).all(axis=normalized_axis)
        repair = finite_inputs & ~np.isfinite(reduced_direct)
        if not bool(np.any(repair)):
            return direct

        with np.errstate(over="ignore", invalid="ignore"):
            stable = np.hypot.reduce(np.abs(array), axis=normalized_axis)
        repaired = np.asarray(reduced_direct).copy()
        repaired[repair] = np.asarray(stable)[repair]
        if keepdims:
            repaired = np.expand_dims(repaired, axis=normalized_axis)
        return repaired


class _StableNumpyProxy:
    """Proxy NumPy for the maintained diagnostics implementation only."""

    def __init__(self, original: Any) -> None:
        self._original = original
        self.linalg = _StableLinalgProxy(original.linalg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _guard_distance_arithmetic(function: Callable[..., Any]) -> Callable[..., Any]:
    """Suppress benign residual overflow while preserving function semantics."""

    if getattr(function, _GUARD_MARKER, False):
        return function

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        with np.errstate(over="ignore", invalid="ignore"):
            return function(*args, **kwargs)

    setattr(guarded, _GUARD_MARKER, True)
    return guarded


def apply_diagnostics_distance_stability_patch(module: ModuleType) -> None:
    """Patch candidate-recall and association-regret distance arithmetic."""

    implementation = getattr(module, "_LEGACY", module)
    if getattr(implementation, _PATCH_MARKER, False):
        return

    implementation.np = _StableNumpyProxy(implementation.np)

    for name in ("candidate_set_recall", "association_regret"):
        guarded = _guard_distance_arithmetic(getattr(module, name))
        setattr(module, name, guarded)
        setattr(implementation, name, guarded)

    setattr(implementation, _PATCH_MARKER, True)
    setattr(module, _PATCH_MARKER, True)
