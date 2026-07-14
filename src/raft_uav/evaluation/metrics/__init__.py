"""Compatibility wrapper validating nearest-time query timestamps.

The maintained implementation lives in the sibling ``metrics.py`` module. This
package preserves the public import path while preventing non-finite query times
from being silently matched to an arbitrary endpoint.
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
_ORIGINAL_NEAREST_TIME_INDICES = _IMPL.nearest_time_indices


def nearest_time_indices(
    reference_times_s: np.ndarray,
    query_times_s: np.ndarray,
) -> np.ndarray:
    """Return nearest finite-reference indices for finite query timestamps.

    Non-finite query timestamps have no well-defined nearest finite sample.
    Reject them explicitly instead of allowing ``searchsorted`` to map them to
    the first or last reference row.
    """

    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    if not bool(np.isfinite(query).all()):
        raise ValueError("query_times_s must contain only finite timestamps")
    return _ORIGINAL_NEAREST_TIME_INDICES(reference_times_s, query)


# Functions defined in the loaded implementation resolve globals from that
# module, so patch its binding as well as the public wrapper export.
_IMPL.nearest_time_indices = nearest_time_indices

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["nearest_time_indices"] = nearest_time_indices

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
