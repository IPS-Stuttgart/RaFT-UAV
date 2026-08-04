"""Keep radar candidate scoring safe for duplicate pandas indices."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import pandas as pd

_PATCH_MARKER = "_raft_uav_duplicate_candidate_index_safe"


def _unique_scored_index(scored: pd.DataFrame) -> pd.DataFrame:
    """Return scored candidates with a unique positional index when needed."""

    if scored.index.is_unique:
        return scored
    normalized = scored.reset_index(drop=True)
    normalized.attrs.update(scored.attrs)
    return normalized


def apply_radar_candidate_index_patch(module: ModuleType) -> None:
    """Patch the shared radar scoring boundary once."""

    original = module._nis_scored_candidates
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> pd.DataFrame:
        scored = original(*args, **kwargs)
        return _unique_scored_index(scored)

    setattr(wrapped, _PATCH_MARKER, True)
    module._nis_scored_candidates = wrapped
