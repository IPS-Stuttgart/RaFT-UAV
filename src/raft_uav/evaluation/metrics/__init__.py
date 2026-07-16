"""Compatibility wrapper with symmetric truth-grid timestamp tolerance.

The maintained implementation lives in the sibling ``metrics.py`` module.  This
package preserves the public import path while ensuring that timestamps within
the existing 1 ns equality tolerance are accepted at either interpolation
bracket endpoint, regardless of whether a maximum time-delta gate is configured.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "metrics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._metrics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load metrics implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _truth_grid_with_symmetric_tolerance(
    estimate_times: np.ndarray,
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    *,
    max_time_delta_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep tolerance-equivalent truth samples at either bracket endpoint."""

    endpoint_atol_s = 1.0e-9
    supported = (truth_times >= estimate_times[0]) & (
        truth_times <= estimate_times[-1]
    )
    supported |= np.isclose(
        truth_times,
        estimate_times[0],
        rtol=0.0,
        atol=endpoint_atol_s,
    )
    supported |= np.isclose(
        truth_times,
        estimate_times[-1],
        rtol=0.0,
        atol=endpoint_atol_s,
    )
    query_times = truth_times[supported]
    query_truth_positions = truth_positions[supported]
    if query_times.size == 0 or max_time_delta_s is None:
        return query_times, query_truth_positions

    max_delta = float(max_time_delta_s)
    right = np.searchsorted(estimate_times, query_times, side="left")
    right = np.clip(right, 0, estimate_times.size - 1)
    left = np.clip(right - 1, 0, estimate_times.size - 1)

    left_exact = np.isclose(
        estimate_times[left],
        query_times,
        rtol=0.0,
        atol=endpoint_atol_s,
    )
    right_exact = np.isclose(
        estimate_times[right],
        query_times,
        rtol=0.0,
        atol=endpoint_atol_s,
    )
    left_delta = np.abs(query_times - estimate_times[left])
    right_delta = np.abs(estimate_times[right] - query_times)
    close_to_bracket = (left_delta <= max_delta) & (right_delta <= max_delta)
    keep = left_exact | right_exact | close_to_bracket
    return query_times[keep], query_truth_positions[keep]


_IMPL._truth_grid_with_estimate_support = _truth_grid_with_symmetric_tolerance

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
