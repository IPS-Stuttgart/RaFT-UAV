"""Compatibility fixes for candidate-mixture MAP.

The maintained implementation lives in the sibling ``candidate_mixture_map.py``
module. This package keeps the public import path while preserving opaque IDs in
CSV inputs and retaining complete candidate frames when target-template times
fall outside the configured matching tolerance.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_mixture_map.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_map_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate-mixture implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


class _PandasCsvProxy:
    """Delegate pandas operations while preserving identifiers in plain CSV reads."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if args or kwargs:
            rows = self._module.read_csv(path, *args, **kwargs)
            out = rows.copy()
            out.columns = [str(column).strip() for column in out.columns]
            return out
        return read_estimate_csv(Path(path))


_IMPL.pd = _PandasCsvProxy(pd)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _target_time_candidate_groups(
    sequence_rows: pd.DataFrame,
    *,
    candidate_times: np.ndarray,
    target_times: np.ndarray,
    tolerance_s: float,
) -> list[tuple[float, pd.DataFrame]]:
    """Match target times without collapsing a nearest timestamp to one row.

    Candidate tables commonly contain several hypotheses at each timestamp. If
    no timestamp lies inside the tolerance window, the legacy fallback selected
    one row by positional index. Keep every hypothesis from the nearest timestamp
    instead, matching the grouped behavior used when a timestamp is in tolerance.
    """

    groups: list[tuple[float, pd.DataFrame]] = []
    if len(sequence_rows) == 0 or len(target_times) == 0:
        return groups
    tolerance = max(float(tolerance_s), 0.0)
    for target_time in target_times:
        left = int(np.searchsorted(candidate_times, target_time - tolerance, side="left"))
        right = int(np.searchsorted(candidate_times, target_time + tolerance, side="right"))
        if right <= left:
            nearest = int(np.argmin(np.abs(candidate_times - target_time)))
            nearest_time = float(candidate_times[nearest])
            left = int(np.searchsorted(candidate_times, nearest_time, side="left"))
            right = int(np.searchsorted(candidate_times, nearest_time, side="right"))
        groups.append((float(target_time), sequence_rows.iloc[left:right]))
    return groups


# Make the legacy frame builder resolve the corrected nearest-time grouping.
_IMPL._target_time_candidate_groups = _target_time_candidate_groups
globals()["_target_time_candidate_groups"] = _target_time_candidate_groups

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
