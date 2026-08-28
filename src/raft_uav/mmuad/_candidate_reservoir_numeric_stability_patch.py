"""Keep candidate-reservoir oracle diagnostics stable for finite extremes."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Any, Callable

import numpy as np


_reservoir = import_module("raft_uav.mmuad.candidate_reservoir")
_IMPL = _reservoir._IMPL
_PATCH_MARKER = "_raft_uav_candidate_reservoir_numeric_stability"
_ORIGINAL_NUMPY = _IMPL.np
_ORIGINAL_LINALG = _ORIGINAL_NUMPY.linalg
_ORIGINAL_BUILD_ORACLE_RECALL_TABLES = (
    _reservoir._ORIGINAL_BUILD_ORACLE_RECALL_TABLES
)


class _StableLinalgProxy:
    """Delegate linalg calls while repairing overflowed Euclidean norms."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_ORIGINAL_LINALG, name)

    def norm(
        self,
        values: Any,
        ord: Any = None,
        axis: Any = None,
        keepdims: bool = False,
    ) -> Any:
        """Preserve ordinary results and repair finite-input norm overflow."""

        with np.errstate(over="ignore", invalid="ignore"):
            direct = _ORIGINAL_LINALG.norm(
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
        if normalized_axis < 0:
            normalized_axis += array.ndim
        if normalized_axis < 0 or normalized_axis >= array.ndim:
            return direct

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
    """Proxy NumPy only inside the legacy candidate-reservoir module."""

    linalg = _StableLinalgProxy()

    def __getattr__(self, name: str) -> Any:
        return getattr(_ORIGINAL_NUMPY, name)


def _guard_oracle_arithmetic(function: Callable[..., Any]) -> Callable[..., Any]:
    """Suppress benign overflow while retaining the original oracle logic."""

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        with np.errstate(over="ignore", invalid="ignore"):
            return function(*args, **kwargs)

    return guarded


def install() -> None:
    """Install finite-extreme guards on candidate-reservoir oracle diagnostics."""

    if getattr(_reservoir, _PATCH_MARKER, False):
        return

    guarded = _guard_oracle_arithmetic(_ORIGINAL_BUILD_ORACLE_RECALL_TABLES)
    _IMPL.np = _StableNumpyProxy()
    _IMPL.build_oracle_recall_tables = guarded
    _reservoir._ORIGINAL_BUILD_ORACLE_RECALL_TABLES = guarded

    setattr(_IMPL, _PATCH_MARKER, True)
    setattr(_reservoir, _PATCH_MARKER, True)


install()
