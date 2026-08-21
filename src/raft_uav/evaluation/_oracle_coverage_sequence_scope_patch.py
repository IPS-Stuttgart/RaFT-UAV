"""Keep single-track oracle coverage within one physical flight/sequence scope."""

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
_SCOPE_FIELDS = ("sequence_id", "flight_id")
_MISSING_SEQUENCE_TEXT = frozenset({"", "nan", "none", "<na>", "nat"})


def _canonical_scope_id(value: object) -> str | None:
    """Return a stable scalar scope identifier, or ``None`` when missing."""

    if not is_scalar(value):
        raise ValueError("scope identifiers must be scalar")
    if value is None or bool(pd.isna(value)):
        return None
    text = str(value).strip()
    return None if text.casefold() in _MISSING_SEQUENCE_TEXT else text


def _scope_keys(
    frame: pd.DataFrame,
    *,
    diagnostic: str,
    role: str,
) -> dict[str, pd.Series]:
    """Return populated, validated scope identifiers aligned with ``frame``."""

    by_field: dict[str, pd.Series] = {}
    for field in _SCOPE_FIELDS:
        if field not in frame.columns:
            continue
        try:
            keys = frame[field].map(_canonical_scope_id).astype(object)
        except ValueError as exc:
            raise ValueError(
                f"{diagnostic} requires scalar {field} values on {role} rows"
            ) from exc
        if not bool(keys.notna().any()):
            continue
        if bool(keys.isna().any()):
            raise ValueError(
                f"{diagnostic} requires a {field} on every {role} row"
            )
        by_field[field] = keys
    return by_field


def _single_radar_scope(
    radar_scope: dict[str, pd.Series],
    *,
    diagnostic: str,
) -> dict[str, str]:
    """Return the one scope value represented by radar for every known field."""

    values: dict[str, str] = {}
    for field, keys in radar_scope.items():
        unique = tuple(dict.fromkeys(str(value) for value in keys.tolist()))
        if len(unique) > 1:
            raise ValueError(
                f"{diagnostic} requires radar rows from one {field}; "
                f"found {list(unique)!r}"
            )
        if unique:
            values[field] = unique[0]
    return values


def _filter_truth_to_shared_scope(
    truth_rows: pd.DataFrame,
    truth_scope: dict[str, pd.Series],
    radar_values: dict[str, str],
    shared_fields: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter truth to all scope dimensions known on both inputs."""

    matching = pd.Series(True, index=truth_rows.index, dtype=bool)
    for field in shared_fields:
        matching &= truth_scope[field].eq(radar_values[field]).fillna(False)
    return truth_rows.loc[matching].copy(), matching


def _single_sequence_inputs(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    diagnostic: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate one radar scope and restrict truth to that physical scope."""

    radar_rows = pd.DataFrame(radar).copy()
    truth_rows = pd.DataFrame(truth).copy()
    if radar_rows.empty or truth_rows.empty:
        return radar_rows, truth_rows

    radar_scope = _scope_keys(
        radar_rows,
        diagnostic=diagnostic,
        role="radar",
    )
    truth_scope = _scope_keys(
        truth_rows,
        diagnostic=diagnostic,
        role="truth",
    )
    if bool(radar_scope) != bool(truth_scope):
        raise ValueError(
            f"{diagnostic} requires sequence_id or flight_id on both radar and "
            "truth or neither"
        )
    if not radar_scope:
        return radar_rows, truth_rows

    radar_values = _single_radar_scope(radar_scope, diagnostic=diagnostic)
    shared_fields = tuple(
        field
        for field in _SCOPE_FIELDS
        if field in radar_scope and field in truth_scope
    )

    if shared_fields:
        matching_truth, matching_mask = _filter_truth_to_shared_scope(
            truth_rows,
            truth_scope,
            radar_values,
            shared_fields,
        )
        if matching_truth.empty:
            scope_text = ", ".join(
                f"{field}={radar_values[field]!r}" for field in shared_fields
            )
            raise ValueError(
                f"{diagnostic} radar scope {scope_text} is absent from truth"
            )

        for field, keys in truth_scope.items():
            if field in shared_fields:
                continue
            matching_keys = keys.loc[matching_mask]
            if int(matching_keys.nunique(dropna=True)) > 1:
                raise ValueError(
                    f"{diagnostic} cannot align one-sided {field} metadata: "
                    "the matching truth scope contains multiple values"
                )
        return radar_rows, matching_truth

    # Historical inputs sometimes expose the same physical identifier under
    # different column names. Preserve that compatibility only when each side
    # has exactly one populated scope field; once a same-named field exists,
    # sequence_id and flight_id are independent dimensions and are never
    # compared to one another.
    if len(radar_scope) == 1 and len(truth_scope) == 1:
        radar_field = next(iter(radar_scope))
        truth_field = next(iter(truth_scope))
        radar_value = radar_values[radar_field]
        matching = truth_scope[truth_field].eq(radar_value).fillna(False)
        if not bool(matching.any()):
            raise ValueError(
                f"{diagnostic} radar identifier {radar_value!r} is absent from truth"
            )
        return radar_rows, truth_rows.loc[matching].copy()

    raise ValueError(
        f"{diagnostic} has no common sequence_id or flight_id scope between "
        "radar and truth"
    )


@wraps(_ORIGINAL_BUILD_COMPACT_COVERAGE)
def build_oracle_candidate_coverage(*args: Any, **kwargs: Any) -> Any:
    """Build compact coverage only from scope-consistent radar and truth."""

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
    """Build detailed coverage only from scope-consistent radar and truth."""

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
