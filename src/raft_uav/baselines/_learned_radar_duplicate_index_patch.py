"""Keep learned radar candidate selection single-row with duplicate indices."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import pandas as pd

_PATCH_MARKER = "_raft_uav_normalizes_learned_radar_candidate_indices"


def apply_learned_radar_duplicate_index_patch(module: ModuleType) -> None:
    """Normalize duplicate scorer indices before label-based candidate selection."""

    original: Callable[..., pd.DataFrame] = (
        module.score_radar_candidates_with_learned_likelihood
    )
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def unique_index_scorer(*args: Any, **kwargs: Any) -> pd.DataFrame:
        scored = original(*args, **kwargs)
        if scored.index.is_unique:
            return scored
        return scored.reset_index(drop=True)

    setattr(unique_index_scorer, _PATCH_MARKER, True)
    module.score_radar_candidates_with_learned_likelihood = unique_index_scorer
