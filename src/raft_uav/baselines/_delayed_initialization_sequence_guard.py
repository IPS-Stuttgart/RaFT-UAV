"""Reject pooled radar sequences in delayed initialization."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


_PATCH_MARKER = "_raft_uav_delayed_initialization_sequence_guard"


def apply_delayed_initialization_sequence_guard(module: Any) -> None:
    """Require delayed initialization radar input to describe at most one sequence."""

    original = module.build_delayed_initial_hypotheses
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def guarded_build_delayed_initial_hypotheses(*args: Any, **kwargs: Any):
        sequence_ids = _explicit_sequence_ids(kwargs.get("radar"))
        if len(sequence_ids) > 1:
            formatted = ", ".join(sorted(repr(value) for value in sequence_ids))
            raise ValueError(
                "delayed initialization requires radar from one sequence_id; "
                f"got {formatted}"
            )
        return original(*args, **kwargs)

    setattr(guarded_build_delayed_initial_hypotheses, _PATCH_MARKER, True)
    module.build_delayed_initial_hypotheses = guarded_build_delayed_initial_hypotheses


def _explicit_sequence_ids(radar: object) -> set[str]:
    """Return normalized non-missing sequence identifiers from a radar frame."""

    if not isinstance(radar, pd.DataFrame) or radar.empty or "sequence_id" not in radar.columns:
        return set()

    sequence_ids: set[str] = set()
    for value in radar["sequence_id"].tolist():
        if _is_missing_sequence_id(value):
            continue
        normalized = str(value).strip()
        if normalized:
            sequence_ids.add(normalized)
    return sequence_ids


def _is_missing_sequence_id(value: object) -> bool:
    if value is None or np.ma.is_masked(value):
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)
