"""Compatibility fixes for reliability-quantile MMUAD anchor selection.

The maintained implementation lives in the sibling
``candidate_mixture_group_weighted_anchor_quantile.py`` module. This package
preserves the public import path while making ``missing_anchor_policy`` ignore
zero-reliability anchors when deciding whether support exists.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_weighted_multi_anchor_mass_topk import (
    _raise_for_missing_positive_weight_anchor_support,
)

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_group_weighted_anchor_quantile.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_weighted_anchor_quantile_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load weighted anchor quantile implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ADD_QUANTILE_UTILITY = (
    _IMPL.add_weighted_quantile_multi_anchor_conditioned_selection_utility
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def add_weighted_quantile_multi_anchor_conditioned_selection_utility(
    candidates: pd.DataFrame,
    anchor_estimates: Mapping[str, pd.DataFrame],
    *,
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: Any = None,
    anchor_config: Any = None,
    quantile_config: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach quantile anchor utility with reliability-aware missing support."""

    effective_anchor_config = anchor_config or _IMPL.AnchorConditioningConfig()
    scored, normalized_anchors, summary = _ORIGINAL_ADD_QUANTILE_UTILITY(
        candidates,
        anchor_estimates,
        anchor_reliability=anchor_reliability,
        mixture_config=mixture_config,
        anchor_config=effective_anchor_config,
        quantile_config=quantile_config,
    )
    if effective_anchor_config.missing_anchor_policy == "error":
        _raise_for_missing_positive_weight_anchor_support(
            scored,
            scored["mixture_weighted_quantile_multi_anchor_matched_weight"],
        )
    return scored, normalized_anchors, summary


_IMPL.add_weighted_quantile_multi_anchor_conditioned_selection_utility = (
    add_weighted_quantile_multi_anchor_conditioned_selection_utility
)
globals()["add_weighted_quantile_multi_anchor_conditioned_selection_utility"] = (
    add_weighted_quantile_multi_anchor_conditioned_selection_utility
)

__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
