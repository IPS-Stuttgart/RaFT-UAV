"""Canonicalize normalized sequence IDs before Track 5 acceleration repair."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import pandas as pd

from raft_uav.mmuad.submission import parse_official_sequence_cell

_PATCH_MARKER = "_raft_uav_canonicalizes_acceleration_sequence_ids"


def _canonicalized_submission(submission: object) -> object:
    """Return normalized rows with official sequence-ID canonicalization applied."""

    rows = pd.DataFrame(submission).copy()
    if "sequence_id" not in rows.columns:
        return submission

    values = rows["sequence_id"]
    if isinstance(values, pd.DataFrame):
        return submission

    canonical: list[str] = []
    for value in values:
        try:
            canonical.append(parse_official_sequence_cell(value))
        except (TypeError, ValueError, OverflowError):
            # Preserve the maintained wrapper's field-specific validation errors.
            return submission
    rows["sequence_id"] = canonical
    return rows


def install() -> None:
    """Install sequence-ID canonicalization at the acceleration-repair boundary."""

    from raft_uav.mmuad import track5_acceleration_limit

    original: Callable[..., Any] = track5_acceleration_limit.repair_track5_acceleration_kinks
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def canonicalized(submission: object, *args: Any, **kwargs: Any) -> Any:
        return original(_canonicalized_submission(submission), *args, **kwargs)

    setattr(canonicalized, _PATCH_MARKER, True)
    setattr(track5_acceleration_limit, "repair_track5_acceleration_kinks", canonicalized)
    implementation = getattr(track5_acceleration_limit, "_IMPL", None)
    if implementation is not None:
        setattr(implementation, "repair_track5_acceleration_kinks", canonicalized)
