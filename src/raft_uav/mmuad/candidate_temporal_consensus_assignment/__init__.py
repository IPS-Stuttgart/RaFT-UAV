"""Strict configuration boundary for assignment temporal consensus.

The maintained implementation lives in the sibling
``candidate_temporal_consensus_assignment.py`` module. This package preserves
the public import path while routing explicit configurations through the shared
validated temporal-consensus boundary before the legacy implementation resolves
defaults.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.mmuad.candidate_temporal_consensus import _validated_config

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_temporal_consensus_assignment.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_temporal_consensus_assignment_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load assignment temporal-consensus implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ADD_ASSIGNMENT_TEMPORAL_CANDIDATE_CONSENSUS = (
    _IMPL.add_assignment_temporal_candidate_consensus
)


def add_assignment_temporal_candidate_consensus(
    candidates: Any,
    *,
    config: _IMPL.TemporalConsensusConfig | None = None,
    assignment_mode: str = "one-to-one",
) -> Any:
    """Attach assignment consensus after validating the explicit configuration."""

    return _ORIGINAL_ADD_ASSIGNMENT_TEMPORAL_CANDIDATE_CONSENSUS(
        candidates,
        config=_validated_config(config),
        assignment_mode=assignment_mode,
    )


_IMPL.add_assignment_temporal_candidate_consensus = (
    add_assignment_temporal_candidate_consensus
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["add_assignment_temporal_candidate_consensus"] = (
    add_assignment_temporal_candidate_consensus
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
