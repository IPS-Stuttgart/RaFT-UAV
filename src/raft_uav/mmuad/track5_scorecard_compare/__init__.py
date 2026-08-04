"""Compatibility guards for Track 5 scorecard per-sequence comparisons.

The maintained implementation lives in the sibling
``track5_scorecard_compare.py`` module. This package preserves the public import
path while rejecting missing or duplicate per-sequence identifiers instead of
silently treating malformed rows as real sequences, while preserving textual
sequence identifiers when the CLI reads pose-by-sequence CSV files, and while
requiring regression tolerances to be finite non-negative real scalars.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_scorecard_compare.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_scorecard_compare_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 scorecard comparison implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE_POSE_TABLE = _IMPL._normalize_pose_by_sequence_table
_ORIGINAL_COMPARE_POSE_TABLES = _IMPL.compare_pose_by_sequence_tables
_ORIGINAL_MAIN = _IMPL.main
_SEQUENCE_ID_DTYPES = {
    "sequence_id": "string",
    "sequence": "string",
}
_REGRESSION_TOLERANCE_ERROR = (
    "regression_tolerance_mse must be a finite non-negative real scalar"
)


class _SequencePreservingPandasProxy:
    """Delegate pandas operations while loading sequence identifiers as text."""

    def __init__(self, pandas_module: Any) -> None:
        self._pandas_module = pandas_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pandas_module, name)

    def read_csv(self, *args: Any, **kwargs: Any) -> Any:
        dtype = kwargs.get("dtype")
        if dtype is None:
            kwargs["dtype"] = dict(_SEQUENCE_ID_DTYPES)
        elif isinstance(dtype, dict):
            normalized_dtype = dict(dtype)
            for column, column_dtype in _SEQUENCE_ID_DTYPES.items():
                normalized_dtype.setdefault(column, column_dtype)
            kwargs["dtype"] = normalized_dtype
        return self._pandas_module.read_csv(*args, **kwargs)


def _normalize_pose_by_sequence_table(rows: Any, *, label: str):
    """Reject missing or duplicate sequence rows before legacy normalization."""

    frame = _IMPL.pd.DataFrame(rows).copy()
    sequence_column = None
    if "sequence_id" in frame.columns:
        sequence_column = "sequence_id"
    elif "sequence" in frame.columns:
        sequence_column = "sequence"

    if sequence_column is not None:
        raw_sequence_ids = frame[sequence_column]
        sequence_text = raw_sequence_ids.astype("string").str.strip()
        missing = raw_sequence_ids.isna() | sequence_text.eq("").fillna(False)
        if bool(missing.any()):
            examples = missing[missing].index.tolist()[:5]
            raise ValueError(
                f"{label} pose table contains missing sequence_id rows: {examples}"
            )

        sequence_ids = raw_sequence_ids.astype(str)
        duplicate = sequence_ids.duplicated(keep=False)
        if duplicate.any():
            examples = sequence_ids.loc[duplicate].drop_duplicates().head(5).tolist()
            raise ValueError(
                f"{label} pose table contains duplicate sequence_id rows: {examples}"
            )

    return _ORIGINAL_NORMALIZE_POSE_TABLE(frame, label=label)


def _normalize_regression_tolerance(value: Any) -> float:
    """Return a lossless finite non-negative regression tolerance."""

    current = value
    seen: set[int] = set()
    while isinstance(current, _IMPL.np.ndarray):
        identity = id(current)
        if identity in seen or current.ndim != 0:
            raise ValueError(_REGRESSION_TOLERANCE_ERROR)
        seen.add(identity)
        if _IMPL.np.ma.isMaskedArray(current) and _IMPL.np.ma.is_masked(current):
            raise ValueError(_REGRESSION_TOLERANCE_ERROR)
        current = current.item()

    if isinstance(current, (bool, _IMPL.np.bool_)) or _IMPL.np.iscomplexobj(current):
        raise ValueError(_REGRESSION_TOLERANCE_ERROR)
    try:
        normalized = float(current)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_REGRESSION_TOLERANCE_ERROR) from exc
    if not _IMPL.np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(_REGRESSION_TOLERANCE_ERROR)
    return normalized


def compare_pose_by_sequence_tables(
    baseline: Any,
    candidate: Any,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    regression_tolerance_mse: Any = 0.0,
):
    """Compare pose tables with a valid non-negative regression tolerance."""

    tolerance = _normalize_regression_tolerance(regression_tolerance_mse)
    return _ORIGINAL_COMPARE_POSE_TABLES(
        baseline,
        candidate,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        regression_tolerance_mse=tolerance,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the scorecard CLI without numeric inference on sequence identifiers."""

    pandas_module = _IMPL.pd
    _IMPL.pd = _SequencePreservingPandasProxy(pandas_module)
    try:
        return _ORIGINAL_MAIN(argv)
    finally:
        _IMPL.pd = pandas_module


_IMPL._normalize_pose_by_sequence_table = _normalize_pose_by_sequence_table
_IMPL.compare_pose_by_sequence_tables = compare_pose_by_sequence_tables
_IMPL.main = main

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_pose_by_sequence_table"] = _normalize_pose_by_sequence_table
globals()["_normalize_regression_tolerance"] = _normalize_regression_tolerance
globals()["compare_pose_by_sequence_tables"] = compare_pose_by_sequence_tables
globals()["main"] = main

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
