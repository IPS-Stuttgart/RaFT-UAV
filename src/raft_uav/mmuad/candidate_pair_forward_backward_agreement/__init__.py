"""Compatibility fixes for tie-invariant adaptive pair-posterior ranking.

The maintained implementation lives in the sibling module named
``candidate_pair_forward_backward_agreement.py``.
This package preserves the public import path while assigning
equal local-score and output ranks to exactly tied candidates instead of
breaking ties by row order.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from raft_uav.mmuad._adaptive_pair_ranks import (
    descending_average_ranks,
    normalize_scores_with_average_ties,
)

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent / "candidate_pair_forward_backward_agreement.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_pair_forward_backward_agreement_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load adaptive pair implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_IMPL._normalize_scores = normalize_scores_with_average_ties
_IMPL._descending_ranks = descending_average_ranks

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
