"""Compatibility package for timestamp-offset diagnostics.

This package intentionally shadows the historical ``time_offset.py`` module so
imports can keep using ``raft_uav.diagnostics.time_offset`` while applying small
compatibility fixes without changing the legacy module layout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


_LEGACY_PATH = Path(__file__).resolve().parent.parent / "time_offset.py"
_LEGACY_NAME = "_raft_uav_diagnostics_time_offset_legacy"


def _load_legacy_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise ImportError(f"cannot load legacy time_offset module from {_LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()
_original_catprob_candidate_pool = _legacy.catprob_candidate_pool


def catprob_candidate_pool(candidates, threshold):
    """Return the cat-probability candidate pool, accepting disabled thresholds."""

    if threshold is None:
        return candidates
    return _original_catprob_candidate_pool(candidates, threshold)


_legacy.catprob_candidate_pool = catprob_candidate_pool

for _name, _value in vars(_legacy).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

globals()["catprob_candidate_pool"] = catprob_candidate_pool
__all__ = sorted(_name for _name in globals() if not _name.startswith("_"))

del _load_legacy_module, _name, _value
