"""Compatibility package for validated MMUAD candidate oracle-gap diagnostics.

The maintained implementation lives in the sibling ``candidate_oracle_gap.py``
module. This package validates the nearest-time gate, prevents genuinely complex
timestamps, positions, or confidence values from being silently cast to their
real components, applies the shared final-sample convention to duplicate truth
timestamps, and prevents pooled physical flights from sharing candidate context.
"""

from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_gap.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_gap_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy candidate oracle-gap module from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP = _IMPL.build_candidate_oracle_gap
_ORIGINAL_FINITE_CANDIDATE_ROWS = _IMPL._finite_candidate_rows
_ORIGINAL_FINITE_TRUTH_ROWS = _IMPL._finite_truth_rows
_MISSING_SCOPE_TEXT = frozenset({"", "nan", "none", "null", "<na>", "nat"})


def _coerce_real_numeric_columns(
    rows: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Convert numeric columns without discarding nonzero imaginary components."""

    normalized = rows.copy()
    for column in columns:
        if column not in normalized.columns:
            continue
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        values = numeric.to_numpy()
        if np.iscomplexobj(values):
            real = np.real(values)
            imaginary = np.imag(values)
            numeric = pd.Series(
                np.where(
                    np.isfinite(imaginary) & (imaginary == 0.0),
                    real,
                    np.nan,
                ),
                index=normalized.index,
                dtype=float,
            )
        normalized[column] = numeric
    return normalized


def _finite_candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """Reject candidates whose required numeric fields are genuinely complex."""

    normalized = _coerce_real_numeric_columns(
        candidates,
        ("time_s", "x_m", "y_m", "z_m", "confidence"),
    )
    return _ORIGINAL_FINITE_CANDIDATE_ROWS(normalized)


def _finite_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    """Keep the final finite truth row at each normalized sequence timestamp."""

    order_column = "_candidate_oracle_gap_truth_row_order"
    while order_column in truth.columns:
        order_column = f"_{order_column}"
    positioned = truth.copy()
    positioned[order_column] = np.arange(len(positioned), dtype=int)

    normalized = _coerce_real_numeric_columns(
        positioned,
        ("time_s", "x_m", "y_m", "z_m"),
    )
    rows = _ORIGINAL_FINITE_TRUTH_ROWS(normalized)
    if rows.empty:
        return rows.drop(columns=[order_column], errors="ignore")

    key_columns = ["sequence_id", "time_s"]
    return (
        rows.sort_values([*key_columns, order_column], kind="mergesort")
        .drop_duplicates(key_columns, keep="last")
        .sort_values(key_columns, kind="mergesort")
        .drop(columns=[order_column], errors="ignore")
        .reset_index(drop=True)
    )


def _normalize_max_time_delta_s(value: Any) -> float | None:
    if value is None:
        return None
    normalized = optional_float(value)
    if normalized is None or normalized < 0.0:
        raise ValueError(
            "max_time_delta_s must be a nonnegative finite scalar or None"
        )
    return normalized


def _canonical_flight_id(value: object, *, field: str) -> str | None:
    """Return one normalized physical-flight identifier or ``None`` when missing."""

    if not is_scalar(value):
        raise ValueError(f"{field} values must be scalar")
    if value is None:
        return None
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return None
    text = str(value).strip()
    return None if text.casefold() in _MISSING_SCOPE_TEXT else text


def _complete_flight_ids(
    rows: pd.DataFrame,
    *,
    name: str,
) -> pd.Series | None:
    """Return complete normalized flight IDs, or ``None`` when metadata is absent."""

    if rows.empty or "flight_id" not in rows.columns:
        return None
    values = rows["flight_id"]
    if isinstance(values, pd.DataFrame):
        raise ValueError(f"{name} has duplicate 'flight_id' columns")
    normalized = pd.Series(
        [
            _canonical_flight_id(value, field=f"{name}.flight_id")
            for value in values.tolist()
        ],
        index=rows.index,
        dtype=object,
    )
    present = normalized.notna()
    if not bool(present.any()):
        return None
    if not bool(present.all()):
        raise ValueError(
            f"{name} flight_id metadata is partially missing; provide one "
            "flight_id for every row or omit flight_id metadata entirely"
        )
    return normalized


def _normalized_oracle_gap_inputs(
    candidates: Any,
    selected: Any,
    truth: Any,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.Series | None],
]:
    """Normalize inputs once and validate optional physical-flight metadata."""

    frames = {
        "candidates": _IMPL._as_candidate_rows(candidates),
        "selected": _IMPL._as_candidate_rows(
            selected,
            default_source="selected",
        ),
        "truth": _IMPL._as_truth_rows(truth),
    }
    flights = {
        name: _complete_flight_ids(rows, name=name)
        for name, rows in frames.items()
    }
    return frames, flights


def _ambiguous_sequence_ids(
    frames: dict[str, pd.DataFrame],
    flights: dict[str, pd.Series | None],
) -> list[str]:
    """Return sequence IDs known to refer to multiple physical flights."""

    flights_by_sequence: dict[str, set[str]] = {}
    for name, rows in frames.items():
        values = flights[name]
        if rows.empty or values is None:
            continue
        for sequence_id, flight_id in zip(
            rows["sequence_id"].astype(str).tolist(),
            values.astype(str).tolist(),
        ):
            flights_by_sequence.setdefault(sequence_id, set()).add(flight_id)
    return sorted(
        sequence_id
        for sequence_id, flight_values in flights_by_sequence.items()
        if len(flight_values) > 1
    )


def _scope_keys(
    frames: dict[str, pd.DataFrame],
    flights: dict[str, pd.Series | None],
) -> list[tuple[str, str]]:
    """Return all physical `(sequence_id, flight_id)` scopes in stable order."""

    keys: set[tuple[str, str]] = set()
    for name, rows in frames.items():
        values = flights[name]
        if rows.empty:
            continue
        if values is None:  # pragma: no cover - guarded by caller
            raise RuntimeError(f"{name} is missing required flight scope metadata")
        keys.update(
            zip(
                rows["sequence_id"].astype(str).tolist(),
                values.astype(str).tolist(),
            )
        )
    return sorted(keys)


def _rows_for_scope(
    rows: pd.DataFrame,
    flight_ids: pd.Series | None,
    *,
    sequence_id: str,
    flight_id: str,
) -> pd.DataFrame:
    """Select one physical scope and remove its private grouping column."""

    if rows.empty:
        return rows.copy()
    if flight_ids is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("cannot select a flight scope without flight_id metadata")
    sequence_values = rows["sequence_id"].astype(str).to_numpy()
    flight_values = flight_ids.astype(str).to_numpy()
    mask = (sequence_values == sequence_id) & (flight_values == flight_id)
    return rows.iloc[np.flatnonzero(mask)].drop(
        columns=["flight_id"],
        errors="ignore",
    )


def _build_flight_scoped_oracle_gap(
    frames: dict[str, pd.DataFrame],
    flights: dict[str, pd.Series | None],
    *,
    max_time_delta_s: float | None,
) -> pd.DataFrame:
    """Evaluate each physical flight independently and combine public rows."""

    pieces: list[pd.DataFrame] = []
    for sequence_id, flight_id in _scope_keys(frames, flights):
        rows = _ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP(
            _rows_for_scope(
                frames["candidates"],
                flights["candidates"],
                sequence_id=sequence_id,
                flight_id=flight_id,
            ),
            _rows_for_scope(
                frames["selected"],
                flights["selected"],
                sequence_id=sequence_id,
                flight_id=flight_id,
            ),
            _rows_for_scope(
                frames["truth"],
                flights["truth"],
                sequence_id=sequence_id,
                flight_id=flight_id,
            ),
            max_time_delta_s=max_time_delta_s,
        )
        scoped_rows = pd.DataFrame(rows).copy()
        if scoped_rows.empty:
            continue
        if "flight_id" in scoped_rows.columns:  # pragma: no cover - legacy guard
            raise RuntimeError("candidate oracle gap unexpectedly returned flight_id")
        insert_at = int(scoped_rows.columns.get_loc("sequence_id")) + 1
        scoped_rows.insert(insert_at, "flight_id", flight_id)
        pieces.append(scoped_rows)

    if pieces:
        return pd.concat(pieces, ignore_index=True)

    return _ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP(
        frames["candidates"],
        frames["selected"],
        frames["truth"],
        max_time_delta_s=max_time_delta_s,
    )


@wraps(_ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP)
def build_candidate_oracle_gap(
    candidates: Any,
    selected: Any,
    truth: Any,
    *,
    max_time_delta_s: float | None = 0.5,
) -> Any:
    """Build validated oracle-gap rows, isolated by physical flight when possible."""

    normalized_gate = _normalize_max_time_delta_s(max_time_delta_s)
    frames, flights = _normalized_oracle_gap_inputs(
        candidates,
        selected,
        truth,
    )
    nonempty = [name for name, rows in frames.items() if not rows.empty]
    if nonempty and all(flights[name] is not None for name in nonempty):
        return _build_flight_scoped_oracle_gap(
            frames,
            flights,
            max_time_delta_s=normalized_gate,
        )

    ambiguous = _ambiguous_sequence_ids(frames, flights)
    if ambiguous:
        raise ValueError(
            "cannot evaluate pooled candidate oracle-gap rows without complete "
            "flight_id metadata on candidates, selected, and truth; ambiguous "
            f"sequence_id values: {ambiguous}"
        )

    return _ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP(
        frames["candidates"],
        frames["selected"],
        frames["truth"],
        max_time_delta_s=normalized_gate,
    )


_IMPL._finite_candidate_rows = _finite_candidate_rows
_IMPL._finite_truth_rows = _finite_truth_rows
_IMPL.build_candidate_oracle_gap = build_candidate_oracle_gap

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)

globals()["_coerce_real_numeric_columns"] = _coerce_real_numeric_columns
globals()["_finite_candidate_rows"] = _finite_candidate_rows
globals()["_finite_truth_rows"] = _finite_truth_rows
globals()["_normalize_max_time_delta_s"] = _normalize_max_time_delta_s
globals()["_canonical_flight_id"] = _canonical_flight_id
globals()["_complete_flight_ids"] = _complete_flight_ids
globals()["_normalized_oracle_gap_inputs"] = _normalized_oracle_gap_inputs
globals()["_ambiguous_sequence_ids"] = _ambiguous_sequence_ids
globals()["_scope_keys"] = _scope_keys
globals()["_rows_for_scope"] = _rows_for_scope
globals()["_build_flight_scoped_oracle_gap"] = _build_flight_scoped_oracle_gap
globals()["build_candidate_oracle_gap"] = build_candidate_oracle_gap
__doc__ = _IMPL.__doc__
__all__ = [_name for _name in dir(_IMPL) if not _name.startswith("__")]
