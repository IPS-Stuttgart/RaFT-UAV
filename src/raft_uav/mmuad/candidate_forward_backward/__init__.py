"""Compatibility wrapper for the first-order forward-backward implementation.

The maintained implementation lives in the sibling ``candidate_forward_backward.py``
module.  This package keeps the public import path while making track-continuation
bonuses robust to CSV scalar-type drift such as ``491`` versus ``491.0``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.mmuad.candidate_identity import canonical_track_ids

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_forward_backward.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_forward_backward_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load forward-backward implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRANSITION_LOG_LIKELIHOOD = _IMPL._transition_log_likelihood


def _transition_log_likelihood_with_canonical_track_ids(
    previous: dict[str, Any],
    current: dict[str, Any],
    config: Any,
) -> Any:
    """Apply the existing transition model after canonicalizing track identity."""

    previous_rows = dict(previous)
    current_rows = dict(current)
    previous_rows["track_ids"] = canonical_track_ids(previous.get("track_ids", ()))
    current_rows["track_ids"] = canonical_track_ids(current.get("track_ids", ()))
    return _ORIGINAL_TRANSITION_LOG_LIKELIHOOD(previous_rows, current_rows, config)


_IMPL._transition_log_likelihood = _transition_log_likelihood_with_canonical_track_ids

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
