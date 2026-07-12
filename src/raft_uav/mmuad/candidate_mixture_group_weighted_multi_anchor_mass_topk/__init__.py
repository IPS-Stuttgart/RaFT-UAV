"""Compatibility fixes for reliability-weighted MMUAD anchor selection.

The maintained implementation lives in the sibling
``candidate_mixture_group_weighted_multi_anchor_mass_topk.py`` module. This
package preserves the public import path while making ``missing_anchor_policy``
respect reliability weights: an anchor with zero reliability cannot satisfy the
``error`` policy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_group_weighted_multi_anchor_mass_topk.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_weighted_multi_anchor_mass_topk_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load weighted multi-anchor implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ADD_WEIGHTED_UTILITY = (
    _IMPL.add_weighted_multi_anchor_conditioned_selection_utility
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _raise_for_missing_positive_weight_anchor_support(
    rows: pd.DataFrame,
    matched_weight: Any,
) -> None:
    """Raise when candidate frames lack support from positive-weight anchors."""

    weight = np.asarray(matched_weight, dtype=float)
    unsupported = ~np.isfinite(weight) | (weight <= 0.0)
    if not unsupported.any():
        return

    missing_frames = (
        rows.loc[unsupported, ["sequence_id", "time_s"]]
        .drop_duplicates()
        .head(5)
        .itertuples(index=False, name=None)
    )
    examples = ", ".join(
        f"{sequence}@{float(time_s):g}" for sequence, time_s in missing_frames
    )
    raise ValueError(
        "missing support from every positive-reliability anchor trajectory "
        f"for candidate frames: {examples}"
    )


def add_weighted_multi_anchor_conditioned_selection_utility(
    candidates: pd.DataFrame,
    anchor_estimates: Mapping[str, pd.DataFrame],
    *,
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: Any = None,
    anchor_config: Any = None,
    aggregation_config: Any = None,
    reliability_config: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach weighted anchor utility with reliability-aware missing support."""

    effective_anchor_config = anchor_config or _IMPL.AnchorConditioningConfig()
    scored, normalized_anchors, summary = _ORIGINAL_ADD_WEIGHTED_UTILITY(
        candidates,
        anchor_estimates,
        anchor_reliability=anchor_reliability,
        mixture_config=mixture_config,
        anchor_config=effective_anchor_config,
        aggregation_config=aggregation_config,
        reliability_config=reliability_config,
    )
    if effective_anchor_config.missing_anchor_policy == "error":
        _raise_for_missing_positive_weight_anchor_support(
            scored,
            scored["mixture_weighted_multi_anchor_matched_weight"],
        )
    return scored, normalized_anchors, summary


_IMPL.add_weighted_multi_anchor_conditioned_selection_utility = (
    add_weighted_multi_anchor_conditioned_selection_utility
)
globals()["add_weighted_multi_anchor_conditioned_selection_utility"] = (
    add_weighted_multi_anchor_conditioned_selection_utility
)

__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
