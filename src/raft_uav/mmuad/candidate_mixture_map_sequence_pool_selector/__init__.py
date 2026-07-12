"""Sequence-aware wrapper for candidate-mixture pool selection.

The maintained implementation lives in the sibling
``candidate_mixture_map_sequence_pool_selector.py`` module. This package keeps
its public import path while making external initial estimates use the same
sequence alias and sequence-less expansion rules as the per-sequence
multi-start workflow.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.mmuad import candidate_mixture_map_sequence_multistart as sequence_multistart

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_map_sequence_pool_selector.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_map_sequence_pool_selector_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load candidate-mixture sequence pool selector implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RUN_SEQUENCE_POOL_SELECTOR = _IMPL.run_sequence_pool_selector

# Export the maintained implementation first; corrected functions below replace
# the affected callable while preserving all existing public/private helpers.
globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _normalize_sequence_pool_initialization(
    candidates: pd.DataFrame,
    initial_estimates: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Canonicalize sequence aliases or expand one shared trajectory per sequence."""

    return sequence_multistart._expand_sequence_less_external_initialization(
        candidates,
        initial_estimates,
    )


def run_sequence_pool_selector(
    candidates: pd.DataFrame,
    *,
    mixture_config: Any | None = None,
    selector_config: Any | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> Any:
    """Run pool selection with external initialization routed to real sequences."""

    normalized_initial = _normalize_sequence_pool_initialization(
        candidates,
        initial_estimates,
    )
    return _ORIGINAL_RUN_SEQUENCE_POOL_SELECTOR(
        candidates,
        mixture_config=mixture_config,
        selector_config=selector_config,
        initial_estimates=normalized_initial,
        truth=truth,
    )


# Make the legacy CLI and function globals resolve the corrected behavior.
_IMPL.run_sequence_pool_selector = run_sequence_pool_selector
globals()["run_sequence_pool_selector"] = run_sequence_pool_selector
globals()["_normalize_sequence_pool_initialization"] = (
    _normalize_sequence_pool_initialization
)

__doc__ = _IMPL.__doc__
__all__ = sorted(
    {
        *[
            name
            for name in dir(_IMPL)
            if not (name.startswith("__") and name.endswith("__"))
        ],
        "run_sequence_pool_selector",
    }
)
