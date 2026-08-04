"""Compatibility package for lossless candidate-pool top-K validation.

The maintained implementation lives in the sibling ``candidate_pool_compare.py``
module. This package preserves the public import path while rejecting Boolean,
non-integral, non-scalar, masked, complex, and cyclic top-K pseudo-integers before
candidate diagnostics are computed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import importlib.util
from pathlib import Path
import sys

import pandas as pd

from raft_uav.numeric import optional_int as _optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_pool_compare.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_pool_compare_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate pool comparison from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_BUILD_CANDIDATE_POOL_COMPARE_TABLES = _IMPL.build_candidate_pool_compare_tables


def _normalize_top_k_values(top_k_values: Iterable[object]) -> tuple[int, ...]:
    """Return sorted unique positive top-K values without lossy coercion."""

    if isinstance(top_k_values, str | bytes):
        raise ValueError("top_k_values must be an iterable of integer scalars")
    try:
        values = iter(top_k_values)
    except TypeError as exc:
        raise ValueError("top_k_values must be an iterable of integer scalars") from exc

    normalized: set[int] = set()
    for index, value in enumerate(values):
        parsed = _optional_int(value)
        if parsed is None:
            raise ValueError(
                f"top_k_values[{index}] must be an integer scalar, got {value!r}"
            )
        if parsed > 0:
            normalized.add(parsed)
    return tuple(sorted(normalized))


def build_candidate_pool_compare_tables(
    reference_candidates: pd.DataFrame,
    candidate_pools: Mapping[str, pd.DataFrame],
    truth: pd.DataFrame,
    *,
    top_k_values: Iterable[object] = _IMPL._DEFAULT_TOP_K,
    score_column: str = _IMPL._DEFAULT_SCORE_COLUMN,
    fallback_score_column: str = _IMPL._DEFAULT_FALLBACK_SCORE_COLUMN,
    max_truth_time_delta_s: float = 0.5,
    good_candidate_threshold_m: float = 5.0,
    loss_tolerance_m: float = 1.0e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build comparison tables after validating top-K controls losslessly."""

    return _ORIGINAL_BUILD_CANDIDATE_POOL_COMPARE_TABLES(
        reference_candidates,
        candidate_pools,
        truth,
        top_k_values=_normalize_top_k_values(top_k_values),
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        max_truth_time_delta_s=max_truth_time_delta_s,
        good_candidate_threshold_m=good_candidate_threshold_m,
        loss_tolerance_m=loss_tolerance_m,
    )


_IMPL.build_candidate_pool_compare_tables = build_candidate_pool_compare_tables

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_top_k_values"] = _normalize_top_k_values
globals()["build_candidate_pool_compare_tables"] = build_candidate_pool_compare_tables

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
