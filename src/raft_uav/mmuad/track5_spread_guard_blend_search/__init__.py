"""Compatibility wrapper for reusable spread-guard blend search controls.

The implementation lives in the sibling ``track5_spread_guard_blend_search.py``
file. This wrapper preserves the public import path while materializing policy
and label iterables once, before the implementation reuses them for each spread
threshold.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_spread_guard_blend_search.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_spread_guard_blend_search_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load spread-guard blend search implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SEARCH = _IMPL.search_track5_spread_guard_blend_settings


def search_track5_spread_guard_blend_settings(
    estimate_inputs: Iterable[Any],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    spread_thresholds_m: Iterable[float],
    fallback_blends: Iterable[float] = (0.0, 0.25, 0.5),
    fallback_policies: Iterable[str] = ("max-weight",),
    fallback_labels: Iterable[str] = (),
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate every requested threshold, policy, label, and blend combination."""

    policies = tuple(fallback_policies)
    labels = tuple(fallback_labels)
    return _ORIGINAL_SEARCH(
        estimate_inputs,
        template=template,
        truth=truth,
        spread_thresholds_m=spread_thresholds_m,
        fallback_blends=fallback_blends,
        fallback_policies=policies,
        fallback_labels=labels,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )


_IMPL.search_track5_spread_guard_blend_settings = search_track5_spread_guard_blend_settings

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
