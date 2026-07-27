"""Compatibility fixes for multi-anchor coverage diagnostics and controls.

The maintained implementation lives in the sibling
``candidate_mixture_group_multi_anchor_coverage.py`` module. This package
preserves the public import path while normalizing serialized anchor-match and
rescue flags before they affect group rescue or summary counts. It also rejects
malformed coverage controls before lossy ``float``/``int`` coercion can change
an experiment's effective configuration.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import _boolean_series
from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_group_multi_anchor_coverage.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_multi_anchor_coverage_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load multi-anchor coverage implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_MULTI_ANCHOR_SELECTION = (
    _IMPL.select_multi_anchor_posterior_mass_hypothesis_group_topk
)
_ORIGINAL_COVERAGE_SELECTION = (
    _IMPL.select_multi_anchor_coverage_hypothesis_group_topk
)
_ORIGINAL_COVERAGE_SUMMARY = _IMPL._coverage_summary
_MATCH_PREFIX = "mixture_multi_anchor_"
_MATCH_SUFFIX = "_matched"


def _normalize_multi_anchor_match_flags(rows: Any) -> pd.DataFrame:
    """Return rows with every serialized per-anchor match flag normalized."""

    normalized = pd.DataFrame(rows).copy()
    for column in normalized.columns:
        name = str(column)
        if name.startswith(_MATCH_PREFIX) and name.endswith(_MATCH_SUFFIX):
            normalized[column] = _boolean_series(normalized[column], normalized.index)
    return normalized


def select_multi_anchor_posterior_mass_hypothesis_group_topk(
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any, Any, Any]:
    """Normalize upstream match flags before coverage selection consumes them."""

    scored, selected, anchors, summary = _ORIGINAL_MULTI_ANCHOR_SELECTION(
        *args,
        **kwargs,
    )
    return _normalize_multi_anchor_match_flags(scored), selected, anchors, summary


def _validate_coverage_config(config: Any) -> None:
    """Reject controls that would be changed by Boolean or integer coercion."""

    if not isinstance(config, _IMPL.AnchorGroupCoverageConfig):
        raise TypeError("coverage_config must be an AnchorGroupCoverageConfig or None")
    if not isinstance(config.enabled, (bool, np.bool_)):
        raise ValueError("enabled must be a Boolean scalar")

    distance = optional_float(config.max_anchor_distance_m)
    if distance is None or distance < 0.0:
        raise ValueError(
            "max_anchor_distance_m must be a finite non-negative real scalar"
        )

    max_extra = optional_int(config.max_extra_groups_per_frame)
    if max_extra is None or max_extra < 0:
        raise ValueError(
            "max_extra_groups_per_frame must be a non-negative integer scalar"
        )

    max_siblings = optional_int(config.max_siblings_per_rescued_group)
    if max_siblings is None or max_siblings <= 0:
        raise ValueError(
            "max_siblings_per_rescued_group must be a positive integer scalar"
        )


def select_multi_anchor_coverage_hypothesis_group_topk(
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Validate coverage controls before dispatching to the maintained selector."""

    call_kwargs = dict(kwargs)
    config = call_kwargs.get("coverage_config")
    if config is None:
        config = _IMPL.AnchorGroupCoverageConfig()
    _validate_coverage_config(config)
    call_kwargs["coverage_config"] = config
    return _ORIGINAL_COVERAGE_SELECTION(*args, **call_kwargs)


def _coverage_summary(
    base_summary: dict[str, Any],
    *,
    coverage_config: Any,
    distance_columns: list[tuple[str, str]],
    selected_before: pd.DataFrame,
    selected_after: pd.DataFrame,
    coverage_frames: pd.DataFrame,
) -> dict[str, Any]:
    """Build coverage counts after normalizing persisted rescue flags."""

    normalized_after = pd.DataFrame(selected_after).copy()
    rescued_column = _IMPL.COVERAGE_RESCUED
    if rescued_column in normalized_after.columns:
        normalized_after[rescued_column] = _boolean_series(
            normalized_after[rescued_column],
            normalized_after.index,
        )
    return _ORIGINAL_COVERAGE_SUMMARY(
        base_summary,
        coverage_config=coverage_config,
        distance_columns=distance_columns,
        selected_before=selected_before,
        selected_after=normalized_after,
        coverage_frames=coverage_frames,
    )


_IMPL.select_multi_anchor_posterior_mass_hypothesis_group_topk = (
    select_multi_anchor_posterior_mass_hypothesis_group_topk
)
_IMPL._validate_coverage_config = _validate_coverage_config
_IMPL.select_multi_anchor_coverage_hypothesis_group_topk = (
    select_multi_anchor_coverage_hypothesis_group_topk
)
_IMPL._coverage_summary = _coverage_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_multi_anchor_match_flags"] = _normalize_multi_anchor_match_flags
globals()["select_multi_anchor_posterior_mass_hypothesis_group_topk"] = (
    select_multi_anchor_posterior_mass_hypothesis_group_topk
)
globals()["_validate_coverage_config"] = _validate_coverage_config
globals()["select_multi_anchor_coverage_hypothesis_group_topk"] = (
    select_multi_anchor_coverage_hypothesis_group_topk
)
globals()["_coverage_summary"] = _coverage_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
