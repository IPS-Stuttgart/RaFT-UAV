"""Strict configuration boundary for weighted anchor-quantile group selection.

The maintained implementation lives in the sibling
``candidate_mixture_group_weighted_anchor_quantile.py`` module. This package
preserves the public import path while ensuring that explicitly supplied
configuration objects cannot be silently replaced by defaults merely because
they are falsy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_group_weighted_anchor_quantile.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_weighted_anchor_quantile_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        "cannot load weighted anchor-quantile implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

CandidateMixtureMapConfig = _IMPL.CandidateMixtureMapConfig
HypothesisGroupConfig = _IMPL.HypothesisGroupConfig
PosteriorMassGroupTopKConfig = _IMPL.PosteriorMassGroupTopKConfig
AnchorConditioningConfig = _IMPL.AnchorConditioningConfig
WeightedAnchorQuantileConfig = _IMPL.WeightedAnchorQuantileConfig
MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult = (
    _IMPL.MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult
)

_LEGACY_ADD_WEIGHTED_QUANTILE = (
    _IMPL.add_weighted_quantile_multi_anchor_conditioned_selection_utility
)
_LEGACY_SELECT_WEIGHTED_QUANTILE = (
    _IMPL.select_weighted_quantile_posterior_mass_hypothesis_group_topk
)
_LEGACY_RUN_WEIGHTED_QUANTILE = (
    _IMPL.run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map
)


def _resolve_optional_config(
    value: Any,
    config_type: type[Any],
    field: str,
) -> Any:
    """Return a default only for ``None`` and reject every wrong explicit type."""

    if value is None:
        return config_type()
    if not isinstance(value, config_type):
        raise TypeError(f"{field} must be {config_type.__name__} or None")
    return value


def add_weighted_quantile_multi_anchor_conditioned_selection_utility(
    candidates: pd.DataFrame,
    anchor_estimates: Mapping[str, pd.DataFrame],
    *,
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    quantile_config: WeightedAnchorQuantileConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach the utility after validating every explicit configuration."""

    mixture_config = _resolve_optional_config(
        mixture_config,
        CandidateMixtureMapConfig,
        "mixture_config",
    )
    anchor_config = _resolve_optional_config(
        anchor_config,
        AnchorConditioningConfig,
        "anchor_config",
    )
    quantile_config = _resolve_optional_config(
        quantile_config,
        WeightedAnchorQuantileConfig,
        "quantile_config",
    )
    return _LEGACY_ADD_WEIGHTED_QUANTILE(
        candidates,
        anchor_estimates,
        anchor_reliability=anchor_reliability,
        mixture_config=mixture_config,
        anchor_config=anchor_config,
        quantile_config=quantile_config,
    )


def select_weighted_quantile_posterior_mass_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    quantile_config: WeightedAnchorQuantileConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select physical groups after validating every explicit configuration."""

    mixture_config = _resolve_optional_config(
        mixture_config,
        CandidateMixtureMapConfig,
        "mixture_config",
    )
    group_config = _resolve_optional_config(
        group_config,
        HypothesisGroupConfig,
        "group_config",
    )
    selection_config = _resolve_optional_config(
        selection_config,
        PosteriorMassGroupTopKConfig,
        "selection_config",
    )
    anchor_config = _resolve_optional_config(
        anchor_config,
        AnchorConditioningConfig,
        "anchor_config",
    )
    quantile_config = _resolve_optional_config(
        quantile_config,
        WeightedAnchorQuantileConfig,
        "quantile_config",
    )
    return _LEGACY_SELECT_WEIGHTED_QUANTILE(
        candidates,
        anchor_estimates=anchor_estimates,
        anchor_reliability=anchor_reliability,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        anchor_config=anchor_config,
        quantile_config=quantile_config,
    )


def run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    quantile_config: WeightedAnchorQuantileConfig | None = None,
    final_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Run the weighted quantile pipeline after strict config validation."""

    mixture_config = _resolve_optional_config(
        mixture_config,
        CandidateMixtureMapConfig,
        "mixture_config",
    )
    group_config = _resolve_optional_config(
        group_config,
        HypothesisGroupConfig,
        "group_config",
    )
    selection_config = _resolve_optional_config(
        selection_config,
        PosteriorMassGroupTopKConfig,
        "selection_config",
    )
    anchor_config = _resolve_optional_config(
        anchor_config,
        AnchorConditioningConfig,
        "anchor_config",
    )
    quantile_config = _resolve_optional_config(
        quantile_config,
        WeightedAnchorQuantileConfig,
        "quantile_config",
    )
    return _LEGACY_RUN_WEIGHTED_QUANTILE(
        candidates,
        anchor_estimates=anchor_estimates,
        anchor_reliability=anchor_reliability,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        anchor_config=anchor_config,
        quantile_config=quantile_config,
        final_initial_estimates=final_initial_estimates,
        truth=truth,
    )


_IMPL.add_weighted_quantile_multi_anchor_conditioned_selection_utility = (
    add_weighted_quantile_multi_anchor_conditioned_selection_utility
)
_IMPL.select_weighted_quantile_posterior_mass_hypothesis_group_topk = (
    select_weighted_quantile_posterior_mass_hypothesis_group_topk
)
_IMPL.run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map = (
    run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["add_weighted_quantile_multi_anchor_conditioned_selection_utility"] = (
    add_weighted_quantile_multi_anchor_conditioned_selection_utility
)
globals()["select_weighted_quantile_posterior_mass_hypothesis_group_topk"] = (
    select_weighted_quantile_posterior_mass_hypothesis_group_topk
)
globals()["run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map"] = (
    run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map
)
globals()["_resolve_optional_config"] = _resolve_optional_config

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
