"""Reject malformed candidate-pull result timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd

_INSTALLED = False
_ERROR = "official result Timestamp must be a finite real scalar"


def _valid_timestamp(original: object, normalized: object) -> bool:
    """Return whether a normalized timestamp came from a valid real scalar."""

    if np.ma.is_masked(original) or isinstance(original, (bool, np.bool_)):
        return False
    try:
        scalar = np.asarray(normalized)
    except (TypeError, ValueError):
        return False
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        return False
    try:
        value = float(scalar)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(value))


def install() -> None:
    """Install candidate-pull result timestamp validation exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from raft_uav.mmuad import candidate_pull

    original_normalize = candidate_pull._normalize_official_results

    def _normalize_official_results(
        results: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Normalize official rows only when every timestamp is finite and real."""

        source = pd.DataFrame(results).copy()
        rows, xyz = original_normalize(results)
        normalized = rows["Timestamp"].to_numpy(copy=False)
        for position, value in enumerate(normalized):
            original_value = source["Timestamp"].iloc[position]
            if _valid_timestamp(original_value, value):
                continue
            row_label = source.index[position]
            raise ValueError(
                f"{_ERROR} at row {row_label!r}: {original_value!r}"
            )
        return rows, xyz

    candidate_pull._normalize_official_results = _normalize_official_results
    implementation = getattr(candidate_pull, "_IMPL", None)
    if implementation is not None:
        implementation._normalize_official_results = _normalize_official_results
    _INSTALLED = True
