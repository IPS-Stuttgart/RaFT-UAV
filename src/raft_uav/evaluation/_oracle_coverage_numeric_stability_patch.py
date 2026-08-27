"""Keep oracle-coverage distance diagnostics stable for finite extreme inputs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd


_oracle_coverage = import_module("raft_uav.evaluation.oracle_coverage")
_IMPL = getattr(_oracle_coverage, "_IMPL", _oracle_coverage)
_PATCH_MARKER = "_oracle_coverage_numeric_stability_patch_applied"
_ORIGINAL_NUMPY = _IMPL.np
_ORIGINAL_LINALG = _ORIGINAL_NUMPY.linalg


class _StableLinalgProxy:
    """Delegate NumPy linalg calls while hardening the default Euclidean norm."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_ORIGINAL_LINALG, name)

    def norm(
        self,
        values: Any,
        ord: Any = None,
        axis: Any = None,
        keepdims: bool = False,
    ) -> Any:
        """Preserve ordinary results and fall back to scaled Euclidean arithmetic."""

        with np.errstate(over="ignore", invalid="ignore"):
            direct = _ORIGINAL_LINALG.norm(
                values,
                ord=ord,
                axis=axis,
                keepdims=keepdims,
            )
        if bool(np.isfinite(np.asarray(direct, dtype=float)).all()):
            return direct
        if ord is not None:
            return direct

        array = np.asarray(values, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            absolute = np.abs(array)
            if axis is None:
                stable = np.hypot.reduce(absolute.reshape(-1))
                if keepdims:
                    stable = np.asarray(stable).reshape((1,) * array.ndim)
                return stable
            if isinstance(axis, tuple):
                if len(axis) != 1:
                    return direct
                axis = axis[0]
            if isinstance(axis, (int, np.integer)):
                stable = np.hypot.reduce(absolute, axis=int(axis))
                if keepdims:
                    stable = np.expand_dims(stable, axis=int(axis))
                return stable
        return direct


class _StableNumpyProxy:
    """Proxy NumPy for the legacy module without mutating process-global NumPy."""

    linalg = _StableLinalgProxy()

    def __getattr__(self, name: str) -> Any:
        return getattr(_ORIGINAL_NUMPY, name)


def _nearest_truth_time_delta_s(truth: pd.DataFrame, time_s: float) -> float:
    """Return the nearest truth-time distance without leaking subtraction overflow."""

    if truth.empty or "time_s" not in truth.columns:
        return float("nan")
    truth_times = (
        pd.to_numeric(truth["time_s"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    truth_times = np.sort(truth_times[np.isfinite(truth_times)])
    if truth_times.size == 0:
        return float("nan")
    insertion = int(np.searchsorted(truth_times, float(time_s)))
    right = int(np.clip(insertion, 0, truth_times.size - 1))
    left = int(np.clip(insertion - 1, 0, truth_times.size - 1))
    with np.errstate(over="ignore", invalid="ignore"):
        left_delta = abs(truth_times[left] - float(time_s))
        right_delta = abs(truth_times[right] - float(time_s))
    return float(min(left_delta, right_delta))


def install() -> None:
    """Install numeric-stability guards on compact oracle-coverage internals."""

    if getattr(_oracle_coverage, _PATCH_MARKER, False):
        return

    # Functions defined in the maintained sibling implementation resolve their
    # global ``np`` through ``_IMPL.__dict__``. Replacing only that module-local
    # binding avoids any process-global NumPy monkey patch while making every
    # existing Euclidean-norm call preserve large representable magnitudes.
    _IMPL.np = _StableNumpyProxy()
    _IMPL._nearest_truth_time_delta_s = _nearest_truth_time_delta_s
    _oracle_coverage._nearest_truth_time_delta_s = _nearest_truth_time_delta_s

    setattr(_IMPL, _PATCH_MARKER, True)
    setattr(_oracle_coverage, _PATCH_MARKER, True)


install()
