"""Keep IMM radar selection positional when candidate indices are duplicated."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_duplicate_index_patch"


def apply_imm_radar_duplicate_index_patch(module: Any) -> None:
    """Patch IMM radar selection to honor its single-row return contract."""

    original = module._select_imm_radar_candidate
    if getattr(original, _PATCH_MARKER, False):
        return

    def patched(*args: Any, **kwargs: Any) -> pd.Series | None:
        selected = original(*args, **kwargs)
        if selected is None or isinstance(selected, pd.Series):
            return selected
        if not isinstance(selected, pd.DataFrame):
            raise TypeError("IMM radar candidate selection returned an unsupported object")
        if selected.empty:
            return None

        scores = pd.to_numeric(
            selected["association_score"],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = ~np.isnan(scores)
        if not valid.any():
            return None
        valid_positions = np.flatnonzero(valid)
        best_position = int(valid_positions[np.argmin(scores[valid])])
        return selected.iloc[best_position].copy()

    setattr(patched, _PATCH_MARKER, True)
    module._select_imm_radar_candidate = patched
