"""Keep single-track oracle coverage within one sequence."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
import inspect
from typing import Any

import pandas as pd
from pandas.api.types import is_scalar


_compact_coverage = import_module("raft_uav.evaluation.oracle_coverage")
_detailed_coverage = import_module("raft_uav.evaluation.oracle_candidate_coverage")
_ORIGINAL_BUILD_COMPACT_COVERAGE = (
    _compact_coverage.build_oracle_candidate_coverage
)
_ORIGINAL_BUILD_DETAILED_COVERAGE = (
    _detailed_coverage.build_oracle_candidate_coverage_diagnostics
)
_COMPACT_SIGNATURE = inspect.signature(_ORIGINAL_BUILD_COMPACT_COVERAGE)
_DETAILED_SIGNATURE = inspect.signature(_ORIGINAL_BUILD_DETAILED_COVERAGE)
_SEQUENCE_COLUMN_CANDIDATES = ("sequence_id", "flight_id")
_MISSING_SEQUENCE_TEXT = frozenset({"", "nan", "none", "<na>", "nat"})


def _canonical_sequence_id(value: object) -> str | None:
    """Return a stable scalar sequence identifier, or ``None`` when missing."""

    if not is_scalar(value):
        raise ValueError("sequence identifiers must be scalar")
    if value is None or bool(pd.isna(value)):
        return None
    text = str(value).strip()
    return None if text.casefold() in _MISSING_SEQUENCE_TEXT else text


def _sequence_keys(
    frame: pd.DataFrame,
    *,
    diagnostic: str,
    role: str,
) -> pd.Series:
    """Return validated canonical sequence identifiers aligned with ``frame``."""

    columns = [
        column
        for column in _SEQUENCE_COLUMN_CANDIDATES
        if column in frame.columns
    ]
    if not columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype=object)

    by_column: dict[str, pd.Series] = {}
    for column in columns:
        try:
            by_column[column] = (
                frame[column].map(_canonical_sequence_id).astype(object)
            )
        except ValueError as exc:
            raise ValueError(
                f"{diagnostic} requires scalar {column} values on {role} rows"
            ) from exc

    keys = by_column[columns[0]].copy()
    for column in columns[1:]:
        other = by_column[column]
        conflicts = keys.notna() & other.notna() & keys.ne(other)
        if bool(conflicts.any()):
            raise ValueError(
                f"{diagnostic} requires matching sequence_id and flight_id "
                f"on {role} rows"
            )
        keys = keys.where(keys.notna(), other)
    return keys


def _single_sequence_inputs(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    diagnostic: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate one radar sequence and restrict truth to the matching sequence."""

    radar_rows = pd.DataFrame(radar).copy()
    truth_rows = pd.DataFrame(truth).copy()
    if radar_rows.empty or truth_rows.empty:
        return radar_rows, truth_rows

    radar_keys = _sequence_keys(
        radar_rows,
        diagnostic=diagnostic,
        role="radar",
    )
    truth_keys = _sequence_keys(
        truth_rows,
        diagnostic=diagnostic,
        role="truth",
    )
    radar_ids = sorted({str(value) for value in radar_keys.dropna().tolist()})
    truth_ids = sorted({str(value) for value in truth_keys.dropna().tolist()})

    if len(radar_ids) > 1:
        raise ValueError(
            f"{diagnostic} requires radar rows from one sequence; found {radar_ids}"
        )
    if bool(radar_ids) != bool(truth_ids):
        raise ValueError(
            f"{diagnostic} requires sequence_id or flight_id on both radar and "
            "truth or neither"
        )
    if not radar_ids:
        return radar_rows, truth_rows
    if radar_keys.isna().any():
        raise ValueError(
            f"{diagnostic} requires a sequence identifier on every radar row"
        )
    if truth_keys.isna().any():
        raise ValueError(
            f"{diagnostic} requires a sequence identifier on every truth row"
        )

    sequence_id = radar_ids[0]
    matching_truth = truth_keys.eq(sequence_id).fillna(False)
    if not bool(matching_truth.any()):
        raise ValueError(
            f"{diagnostic} radar sequence {sequence_id!r} is absent from truth"
        )
    return radar_rows, truth_rows.loc[matching_truth].copy()


@wraps(_ORIGINAL_BUILD_COMPACT_COVERAGE)
def build_oracle_candidate_coverage(*args: Any, **kwargs: Any) -> Any:
    """Build compact coverage only from sequence-consistent radar and truth."""

    bound = _COMPACT_SIGNATURE.bind(*args, **kwargs)
    bound.apply_defaults()
    radar, truth = _single_sequence_inputs(
        bound.arguments["radar"],
        bound.arguments["truth"],
        diagnostic="oracle candidate coverage",
    )
    bound.arguments["radar"] = radar
    bound.arguments["truth"] = truth
    return _ORIGINAL_BUILD_COMPACT_COVERAGE(*bound.args, **bound.kwargs)


@wraps(_ORIGINAL_BUILD_DETAILED_COVERAGE)
def build_oracle_candidate_coverage_diagnostics(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Build detailed coverage only from sequence-consistent radar and truth."""

    bound = _DETAILED_SIGNATURE.bind(*args, **kwargs)
    bound.apply_defaults()
    radar, truth = _single_sequence_inputs(
        bound.arguments["radar"],
        bound.arguments["truth"],
        diagnostic="oracle candidate coverage diagnostics",
    )
    bound.arguments["radar"] = radar
    bound.arguments["truth"] = truth
    return _ORIGINAL_BUILD_DETAILED_COVERAGE(*bound.args, **bound.kwargs)


def install() -> None:
    """Install sequence scoping on public and legacy implementation paths."""

    if getattr(_compact_coverage, "_sequence_scope_patch_applied", False):
        return

    _compact_coverage.build_oracle_candidate_coverage = (
        build_oracle_candidate_coverage
    )
    compact_implementation = getattr(_compact_coverage, "_IMPL", None)
    if compact_implementation is not None:
        compact_implementation.build_oracle_candidate_coverage = (
            build_oracle_candidate_coverage
        )

    _detailed_coverage.build_oracle_candidate_coverage_diagnostics = (
        build_oracle_candidate_coverage_diagnostics
    )
    detailed_implementation = getattr(_detailed_coverage, "_IMPL", None)
    if detailed_implementation is not None:
        detailed_implementation.build_oracle_candidate_coverage_diagnostics = (
            build_oracle_candidate_coverage_diagnostics
        )

    _compact_coverage._sequence_scope_patch_applied = True
    _detailed_coverage._sequence_scope_patch_applied = True


install()
