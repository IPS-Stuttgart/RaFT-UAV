"""Compatibility fixes for anchor-conditioned MMUAD group selection.

The maintained implementation lives in the sibling
``candidate_mixture_group_anchor_mass_topk.py`` module. This package preserves
its public import path while ensuring that the result's ``scored_candidates``
table contains every anchor-scored input candidate, not only the rows retained
by group selection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_group_anchor_mass_topk.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_anchor_mass_topk_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load anchor group selection implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

# Export the maintained implementation before replacing the two public functions
# that need access to the complete scored table.
globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _select_scored_anchor_candidates(
    scored: pd.DataFrame,
    *,
    anchors: pd.DataFrame,
    anchor_summary: dict[str, Any],
    mixture_config: Any,
    group_config: Any,
    selection_config: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection_mixture_config = _IMPL.replace(
        mixture_config,
        score_column=_IMPL.ANCHOR_UTILITY_COLUMN,
        fallback_score_columns=(),
        score_normalization="none",
        score_weight=1.0,
        temperature=1.0,
        sigma_log_weight=0.0,
    )
    selected, summary = _IMPL.select_posterior_mass_hypothesis_group_topk(
        scored,
        mixture_config=selection_mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )
    summary = dict(summary)
    summary["schema"] = "raft-uav-mmuad-anchor-posterior-mass-group-topk-v1"
    summary["anchor_conditioning"] = anchor_summary
    summary["selection_mixture_config"] = _IMPL.asdict(selection_mixture_config)
    summary["truth_used_for_selection"] = False
    summary["anchor_rows_used"] = int(len(anchors))
    return selected, _IMPL._jsonable(summary)


def select_anchor_posterior_mass_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    initial_estimates: pd.DataFrame,
    mixture_config: Any | None = None,
    group_config: Any | None = None,
    selection_config: Any | None = None,
    anchor_config: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select anchor-conditioned groups while scoring the input only once."""

    mixture_config = mixture_config or _IMPL.CandidateMixtureMapConfig()
    group_config = group_config or _IMPL.HypothesisGroupConfig()
    selection_config = selection_config or _IMPL.PosteriorMassGroupTopKConfig()
    anchor_config = anchor_config or _IMPL.AnchorConditioningConfig()
    scored, anchors, anchor_summary = _IMPL.add_anchor_conditioned_selection_utility(
        candidates,
        initial_estimates,
        mixture_config=mixture_config,
        anchor_config=anchor_config,
    )
    selected, summary = _select_scored_anchor_candidates(
        scored,
        anchors=anchors,
        anchor_summary=anchor_summary,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )
    return selected, anchors, summary


def run_anchor_posterior_mass_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    initial_estimates: pd.DataFrame,
    mixture_config: Any | None = None,
    group_config: Any | None = None,
    selection_config: Any | None = None,
    anchor_config: Any | None = None,
    truth: pd.DataFrame | None = None,
) -> Any:
    """Run selection while retaining diagnostics for rejected candidates."""

    mixture_config = mixture_config or _IMPL.CandidateMixtureMapConfig()
    group_config = group_config or _IMPL.HypothesisGroupConfig()
    selection_config = selection_config or _IMPL.PosteriorMassGroupTopKConfig()
    anchor_config = anchor_config or _IMPL.AnchorConditioningConfig()

    scored, anchors, anchor_summary = _IMPL.add_anchor_conditioned_selection_utility(
        candidates,
        initial_estimates,
        mixture_config=mixture_config,
        anchor_config=anchor_config,
    )
    selected, summary = _select_scored_anchor_candidates(
        scored,
        anchors=anchors,
        anchor_summary=anchor_summary,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )

    effective_mixture_config = mixture_config
    if int(selection_config.max_group_top_k) > 0:
        effective_mixture_config = _IMPL.replace(mixture_config, top_k=0)
    grouped = _IMPL.run_grouped_candidate_mixture_map(
        selected,
        mixture_config=effective_mixture_config,
        group_config=group_config,
        initial_estimates=anchors,
        truth=truth,
    )
    summary["final_mixture_config"] = _IMPL.asdict(effective_mixture_config)
    return _IMPL.AnchorPosteriorMassGroupTopKCandidateMixtureResult(
        scored_candidates=scored,
        selected_candidates=selected,
        grouped_result=grouped,
        selection_summary=_IMPL._jsonable(summary),
    )


_IMPL.select_anchor_posterior_mass_hypothesis_group_topk = (
    select_anchor_posterior_mass_hypothesis_group_topk
)
_IMPL.run_anchor_posterior_mass_group_topk_candidate_mixture_map = (
    run_anchor_posterior_mass_group_topk_candidate_mixture_map
)
globals()["select_anchor_posterior_mass_hypothesis_group_topk"] = (
    select_anchor_posterior_mass_hypothesis_group_topk
)
globals()["run_anchor_posterior_mass_group_topk_candidate_mixture_map"] = (
    run_anchor_posterior_mass_group_topk_candidate_mixture_map
)

__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
