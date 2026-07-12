"""Compatibility handling for empty MMUAD multi-anchor candidate tables.

The maintained implementation lives in the sibling
``candidate_mixture_group_multi_anchor_mass_topk.py`` module. This package
preserves its public import path while keeping the single-anchor empty-input
contract: an empty, schema-valid candidate table returns empty diagnostics
instead of failing while reading columns that are only populated for non-empty
inputs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_group_multi_anchor_mass_topk.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_multi_anchor_mass_topk_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load multi-anchor group selector from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ADD_MULTI_ANCHOR_UTILITY = (
    _IMPL.add_multi_anchor_conditioned_selection_utility
)


def _empty_series(index: pd.Index, dtype: str) -> pd.Series:
    return pd.Series(index=index, dtype=dtype)


def add_multi_anchor_conditioned_selection_utility(
    candidates: pd.DataFrame,
    anchor_estimates: Mapping[str, pd.DataFrame],
    *,
    mixture_config: Any | None = None,
    anchor_config: Any | None = None,
    aggregation_config: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score multiple anchors while preserving valid empty candidate inputs."""

    rows = _IMPL.normalize_candidate_columns(
        pd.DataFrame(candidates).copy()
    ).reset_index(drop=True)
    if not rows.empty:
        return _ORIGINAL_ADD_MULTI_ANCHOR_UTILITY(
            candidates,
            anchor_estimates,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
            aggregation_config=aggregation_config,
        )

    mixture_config = mixture_config or _IMPL.CandidateMixtureMapConfig()
    anchor_config = anchor_config or _IMPL.AnchorConditioningConfig()
    aggregation_config = (
        aggregation_config or _IMPL.MultiAnchorAggregationConfig()
    )
    _IMPL._validate_aggregation_config(aggregation_config)

    anchor_items = [
        (str(label).strip(), pd.DataFrame(estimates).copy())
        for label, estimates in anchor_estimates.items()
    ]
    if not anchor_items:
        raise ValueError("at least one anchor trajectory is required")
    labels = [label for label, _ in anchor_items]
    if any(not label for label in labels):
        raise ValueError("anchor labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError("anchor labels must be unique after trimming")
    slugs = _IMPL._unique_anchor_slugs(labels)

    neutral_anchor_config = _IMPL.replace(
        anchor_config,
        anchor_selection_weight=0.0,
        missing_anchor_policy="neutral",
    )
    normalized_anchor_parts: list[pd.DataFrame] = []
    anchor_summaries: dict[str, Any] = {}

    for label, anchor_rows in anchor_items:
        _, normalized, anchor_summary = (
            _IMPL.add_anchor_conditioned_selection_utility(
                rows,
                anchor_rows,
                mixture_config=mixture_config,
                anchor_config=neutral_anchor_config,
            )
        )
        normalized_part = normalized.copy()
        normalized_part.insert(0, "anchor_name", label)
        normalized_anchor_parts.append(normalized_part)
        anchor_summaries[label] = anchor_summary

    scored = rows.copy()
    for column in (
        "mixture_multi_anchor_base_raw_score",
        "mixture_multi_anchor_sigma_m",
        "mixture_multi_anchor_base_utility",
    ):
        scored[column] = _empty_series(scored.index, "float64")

    for slug in slugs:
        scored[f"mixture_multi_anchor_{slug}_matched"] = _empty_series(
            scored.index, "bool"
        )
        for suffix in ("time_delta_s", "distance_m", "cost"):
            scored[f"mixture_multi_anchor_{slug}_{suffix}"] = _empty_series(
                scored.index, "float64"
            )

    scored["mixture_multi_anchor_matched_count"] = _empty_series(
        scored.index, "int64"
    )
    scored["mixture_multi_anchor_matched_fraction"] = _empty_series(
        scored.index, "float64"
    )
    scored["mixture_multi_anchor_best_anchor"] = _empty_series(
        scored.index, "object"
    )
    scored["mixture_multi_anchor_best_distance_m"] = _empty_series(
        scored.index, "float64"
    )
    scored["mixture_multi_anchor_aggregate_cost"] = _empty_series(
        scored.index, "float64"
    )
    scored[_IMPL.MULTI_ANCHOR_UTILITY_COLUMN] = _empty_series(
        scored.index, "float64"
    )

    normalized_anchors = pd.concat(
        normalized_anchor_parts,
        ignore_index=True,
    )
    summary = _IMPL._multi_anchor_summary(
        scored,
        normalized_anchors,
        labels=labels,
        anchor_summaries=anchor_summaries,
        anchor_config=anchor_config,
        aggregation_config=aggregation_config,
    )
    return scored, normalized_anchors, summary


_IMPL.add_multi_anchor_conditioned_selection_utility = (
    add_multi_anchor_conditioned_selection_utility
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["add_multi_anchor_conditioned_selection_utility"] = (
    add_multi_anchor_conditioned_selection_utility
)
__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
