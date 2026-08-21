"""Compatibility fixes for candidate-oracle target exports.

The maintained implementation lives in the sibling ``candidate_oracle_targets.py``
module. This package preserves the public import path while rejecting malformed
truth-matching time gates, oracle-label thresholds, candidate-score controls,
and non-finite candidate-score values before they can silently widen, empty, or
corrupt the training export. It also keeps distinct floating-point thresholds
distinct in output labels, retains the final finite truth snapshot for every
physical-flight timestamp, and prevents candidate/truth matching across flights
that reuse the same local sequence labels and timestamps.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_targets.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_targets_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate-oracle targets from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_ORACLE_TARGETS = _IMPL.build_candidate_oracle_targets
_ORIGINAL_NORMALIZE_TRUTH_COLUMNS = _IMPL.normalize_truth_columns
_MISSING_SCOPE_TEXT = frozenset({"", "nan", "none", "null", "<na>", "nat"})
_SCOPE_TOKEN_PREFIX = "__raft_uav_candidate_oracle_scope__"


def _validated_numeric_tuple(
    values: Any,
    *,
    name: str,
    strictly_positive: bool,
) -> tuple[float, ...]:
    """Return a tuple of finite scalar thresholds with the requested sign."""

    requirement = "positive" if strictly_positive else "non-negative"
    message = f"{name} must contain only finite {requirement} scalars"
    if values is None or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(message)
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(message) from exc

    normalized: list[float] = []
    for value in items:
        number = optional_float(value)
        invalid_sign = (
            number is not None
            and (number <= 0.0 if strictly_positive else number < 0.0)
        )
        if number is None or invalid_sign:
            raise ValueError(message)
        normalized.append(number)
    return tuple(normalized)


def _validated_score_column(value: Any, *, name: str) -> str:
    """Return one normalized, non-empty candidate-score column name."""

    message = f"{name} must be a non-empty string"
    if not isinstance(value, str):
        raise ValueError(message)
    normalized = value.strip()
    if not normalized:
        raise ValueError(message)
    return normalized


def _validated_score_columns(values: Any) -> tuple[str, ...]:
    """Return normalized fallback score-column names."""

    message = "fallback_score_columns must contain only non-empty strings"
    if values is None or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(message)
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(message) from exc
    try:
        return tuple(
            _validated_score_column(value, name="fallback_score_columns")
            for value in items
        )
    except ValueError as exc:
        raise ValueError(message) from exc


def _validated_config(
    config: _IMPL.CandidateOracleTargetConfig | None,
) -> _IMPL.CandidateOracleTargetConfig:
    """Return a config with valid matching, label, and score controls."""

    if config is None:
        resolved = _IMPL.CandidateOracleTargetConfig()
    elif not isinstance(config, _IMPL.CandidateOracleTargetConfig):
        raise TypeError(
            "config must be a CandidateOracleTargetConfig instance or None"
        )
    else:
        resolved = config

    max_delta = optional_float(resolved.max_truth_time_delta_s)
    if max_delta is None or max_delta < 0.0:
        raise ValueError(
            "max_truth_time_delta_s must be a finite non-negative scalar"
        )
    score_column = _validated_score_column(
        resolved.score_column,
        name="score_column",
    )
    fallback_score_columns = _validated_score_columns(
        resolved.fallback_score_columns
    )
    soft_tau_m = _validated_numeric_tuple(
        resolved.soft_tau_m,
        name="soft_tau_m",
        strictly_positive=True,
    )
    good_thresholds_m = _validated_numeric_tuple(
        resolved.good_thresholds_m,
        name="good_thresholds_m",
        strictly_positive=False,
    )
    return replace(
        resolved,
        max_truth_time_delta_s=max_delta,
        score_column=score_column,
        fallback_score_columns=fallback_score_columns,
        soft_tau_m=soft_tau_m,
        good_thresholds_m=good_thresholds_m,
    )


def _optional_candidate_score(value: object) -> float | None:
    """Recover real values from pandas columns upcast to complex dtype."""

    if isinstance(value, (complex, np.complexfloating)):
        imaginary = float(np.imag(value))
        if not np.isfinite(imaginary) or imaginary != 0.0:
            return None
        value = np.real(value)
    return optional_float(value)


def _candidate_score(
    rows: pd.DataFrame,
    *,
    config: _IMPL.CandidateOracleTargetConfig,
) -> pd.Series:
    """Use the first finite real score in the configured fallback chain."""

    columns = (config.score_column, *config.fallback_score_columns)
    result = pd.Series(float("nan"), index=rows.index, dtype=float)
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.Series(
            [_optional_candidate_score(value) for value in rows[column]],
            index=rows.index,
            dtype=float,
        )
        result = result.where(result.notna(), values)
    return result.fillna(0.0).astype(float)


def _threshold_label(value: float) -> str:
    """Return a column-safe shortest round-trip floating-point label."""

    text = repr(float(value))
    if text.endswith(".0") and "e" not in text.lower():
        text = text[:-2]
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _scope_identifier(value: object, *, column: str) -> str | None:
    """Return one normalized opaque scope identifier without lossy coercion."""

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{column} values must be scalar identifiers")
        value = value.item()
    if not pd.api.types.is_scalar(value):
        raise ValueError(f"{column} values must be scalar identifiers")
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    text = str(value).strip()
    return None if text.casefold() in _MISSING_SCOPE_TEXT else text


def _normalized_scope_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Normalize one scope column while preserving missing values explicitly."""

    return pd.Series(
        [_scope_identifier(value, column=column) for value in frame[column]],
        index=frame.index,
        dtype=object,
    )


def _flight_metadata_state(frame: pd.DataFrame) -> tuple[str, pd.Series | None]:
    """Classify flight metadata as absent, complete, or partially populated."""

    if "flight_id" not in frame.columns:
        return "absent", None
    values = _normalized_scope_values(frame, "flight_id")
    populated = values.notna()
    if not bool(populated.any()):
        return "absent", values
    if bool(populated.all()):
        return "complete", values
    return "partial", values


def _joint_scope_keys(frame: pd.DataFrame) -> list[tuple[str, str | None]]:
    """Return normalized joint sequence/flight keys for every row."""

    sequence = _normalized_scope_values(frame, "sequence_id")
    if bool(sequence.isna().any()):
        raise ValueError("sequence_id values must be complete after normalization")
    flight = _normalized_scope_values(frame, "flight_id")
    return [
        (str(sequence_value), flight_value)
        for sequence_value, flight_value in zip(sequence, flight, strict=True)
    ]


def _scope_token_maps(
    *key_groups: list[tuple[str, str | None]],
) -> tuple[dict[tuple[str, str | None], str], dict[str, tuple[str, str | None]]]:
    """Create deterministic collision-free internal tokens for joint scope keys."""

    unique = {key for keys in key_groups for key in keys}
    ordered = sorted(unique, key=lambda key: (key[0], key[1] is None, key[1] or ""))
    key_to_token = {
        key: f"{_SCOPE_TOKEN_PREFIX}{index:012d}"
        for index, key in enumerate(ordered)
    }
    return key_to_token, {token: key for key, token in key_to_token.items()}


def _apply_scope_tokens(
    frame: pd.DataFrame,
    keys: list[tuple[str, str | None]],
    key_to_token: dict[tuple[str, str | None], str],
) -> pd.DataFrame:
    """Replace public identifiers with an internal joint-scope sequence token."""

    out = frame.copy()
    out["sequence_id"] = [key_to_token[key] for key in keys]
    out["flight_id"] = [pd.NA if key[1] is None else key[1] for key in keys]
    return out


def _restore_scope_tokens(
    frame: pd.DataFrame,
    token_to_key: dict[str, tuple[str, str | None]],
) -> pd.DataFrame:
    """Restore public sequence and flight identifiers on a result table."""

    out = pd.DataFrame(frame).copy()
    if out.empty or "sequence_id" not in out.columns:
        return out
    tokens = out["sequence_id"].astype(str)
    unknown = sorted(set(tokens).difference(token_to_key))
    if unknown:
        raise RuntimeError(
            f"candidate-oracle targets returned unknown internal scopes: {unknown}"
        )
    keys = [token_to_key[token] for token in tokens]
    out["sequence_id"] = [key[0] for key in keys]
    flight = pd.Series(
        [pd.NA if key[1] is None else key[1] for key in keys],
        index=out.index,
        dtype=object,
    )
    if "flight_id" in out.columns:
        out["flight_id"] = flight
    else:
        location = out.columns.get_loc("sequence_id") + 1
        out.insert(location, "flight_id", flight)
    return out


def _sequence_to_flight(
    frame: pd.DataFrame,
    flight_values: pd.Series,
) -> dict[str, str]:
    """Return one flight per normalized sequence or reject an ambiguous pool."""

    sequence = _normalized_scope_values(frame, "sequence_id")
    if bool(sequence.isna().any()):
        raise ValueError("sequence_id values must be complete after normalization")
    mapping: dict[str, str] = {}
    for sequence_value, flight_value in zip(sequence, flight_values, strict=True):
        assert sequence_value is not None
        assert flight_value is not None
        previous = mapping.setdefault(str(sequence_value), str(flight_value))
        if previous != str(flight_value):
            raise ValueError(
                "one-sided flight_id metadata is ambiguous because one sequence "
                "contains multiple physical flights"
            )
    return mapping


def _attach_sequence_flights(
    frame: pd.DataFrame,
    sequence_to_flight: dict[str, str],
) -> pd.DataFrame:
    """Attach an unambiguous one-sided flight identifier to result rows."""

    out = pd.DataFrame(frame).copy()
    if out.empty or "sequence_id" not in out.columns:
        return out
    sequence = out["sequence_id"].map(
        lambda value: _scope_identifier(value, column="sequence_id")
    )
    flight = sequence.map(sequence_to_flight)
    if "flight_id" in out.columns:
        out["flight_id"] = flight
    else:
        location = out.columns.get_loc("sequence_id") + 1
        out.insert(location, "flight_id", flight)
    return out


def _authoritative_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    """Retain the final finite input row for each physical-flight truth timestamp."""

    rows = pd.DataFrame(truth).copy()
    if rows.empty:
        return rows
    marker = "_candidate_oracle_truth_input_order"
    while marker in rows.columns:
        marker = f"_{marker}"
    rows[marker] = np.arange(len(rows), dtype=np.int64)
    normalized = _ORIGINAL_NORMALIZE_TRUTH_COLUMNS(rows)
    if normalized.empty:
        return normalized.drop(columns=[marker], errors="ignore")

    state, flight_values = _flight_metadata_state(normalized)
    if state == "partial":
        raise ValueError("truth flight_id metadata must be complete or absent")
    scope_columns = ["sequence_id"]
    if state == "complete":
        assert flight_values is not None
        normalized["flight_id"] = flight_values
        scope_columns.append("flight_id")
    return (
        normalized.sort_values([*scope_columns, "time_s", marker], kind="mergesort")
        .drop_duplicates([*scope_columns, "time_s"], keep="last")
        .sort_values([*scope_columns, "time_s"], kind="mergesort")
        .drop(columns=[marker])
        .reset_index(drop=True)
    )


def _summary_payload_with_scope(
    target_rows: pd.DataFrame,
    frame_summary: pd.DataFrame,
    *,
    config: _IMPL.CandidateOracleTargetConfig,
) -> dict[str, Any]:
    """Return legacy summary metrics with physical-flight attribution."""

    payload = _IMPL._summary_payload(target_rows, frame_summary, config=config)
    if frame_summary.empty or "flight_id" not in frame_summary.columns:
        return payload
    flights = _normalized_scope_values(frame_summary, "flight_id")
    if not bool(flights.notna().any()):
        return payload
    if bool(flights.isna().any()):
        raise RuntimeError("candidate-oracle frame summary lost physical-flight scope")

    scoped = frame_summary.copy()
    scoped["flight_id"] = flights
    by_sequence: list[dict[str, Any]] = []
    for (sequence_id, flight_id), group in scoped.groupby(
        ["sequence_id", "flight_id"],
        sort=True,
        dropna=False,
    ):
        record = _IMPL._summary_record(group, sequence_id=str(sequence_id))
        record["flight_id"] = str(flight_id)
        by_sequence.append(record)
    payload["by_sequence"] = by_sequence
    return _IMPL._jsonable(payload)


def build_candidate_oracle_targets(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    config: _IMPL.CandidateOracleTargetConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build oracle labels only inside a complete physical-flight scope."""

    validated_config = _validated_config(config)
    candidate_rows = _IMPL.normalize_candidate_columns(pd.DataFrame(candidates).copy())
    authoritative_truth = _authoritative_truth_rows(truth)
    if candidate_rows.empty or authoritative_truth.empty:
        return _ORIGINAL_BUILD_CANDIDATE_ORACLE_TARGETS(
            candidate_rows,
            authoritative_truth,
            config=validated_config,
        )

    candidate_state, candidate_flights = _flight_metadata_state(candidate_rows)
    truth_state, truth_flights = _flight_metadata_state(authoritative_truth)
    if "partial" in {candidate_state, truth_state}:
        raise ValueError(
            "candidate and truth flight_id metadata must each be complete or absent"
        )

    token_to_key: dict[str, tuple[str, str | None]] = {}
    sequence_to_flight: dict[str, str] = {}
    scoped_candidates = candidate_rows
    scoped_truth = authoritative_truth
    if candidate_state == "complete" and truth_state == "complete":
        candidate_keys = _joint_scope_keys(candidate_rows)
        truth_keys = _joint_scope_keys(authoritative_truth)
        key_to_token, token_to_key = _scope_token_maps(candidate_keys, truth_keys)
        scoped_candidates = _apply_scope_tokens(
            candidate_rows,
            candidate_keys,
            key_to_token,
        )
        scoped_truth = _apply_scope_tokens(
            authoritative_truth,
            truth_keys,
            key_to_token,
        )
    elif candidate_state == "complete":
        assert candidate_flights is not None
        sequence_to_flight = _sequence_to_flight(candidate_rows, candidate_flights)
    elif truth_state == "complete":
        assert truth_flights is not None
        sequence_to_flight = _sequence_to_flight(authoritative_truth, truth_flights)

    target_rows, frame_summary, _ = _ORIGINAL_BUILD_CANDIDATE_ORACLE_TARGETS(
        scoped_candidates,
        scoped_truth,
        config=validated_config,
    )
    if token_to_key:
        target_rows = _restore_scope_tokens(target_rows, token_to_key)
        frame_summary = _restore_scope_tokens(frame_summary, token_to_key)
    elif sequence_to_flight:
        target_rows = _attach_sequence_flights(target_rows, sequence_to_flight)
        frame_summary = _attach_sequence_flights(frame_summary, sequence_to_flight)

    summary = _summary_payload_with_scope(
        target_rows,
        frame_summary,
        config=validated_config,
    )
    return target_rows, frame_summary, summary


_IMPL._candidate_score = _candidate_score
_IMPL._threshold_label = _threshold_label
_IMPL.build_candidate_oracle_targets = build_candidate_oracle_targets

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_BUILD_CANDIDATE_ORACLE_TARGETS"] = (
    _ORIGINAL_BUILD_CANDIDATE_ORACLE_TARGETS
)
globals()["_ORIGINAL_NORMALIZE_TRUTH_COLUMNS"] = _ORIGINAL_NORMALIZE_TRUTH_COLUMNS
globals()["_MISSING_SCOPE_TEXT"] = _MISSING_SCOPE_TEXT
globals()["_SCOPE_TOKEN_PREFIX"] = _SCOPE_TOKEN_PREFIX
globals()["_validated_numeric_tuple"] = _validated_numeric_tuple
globals()["_validated_score_column"] = _validated_score_column
globals()["_validated_score_columns"] = _validated_score_columns
globals()["_validated_config"] = _validated_config
globals()["_optional_candidate_score"] = _optional_candidate_score
globals()["_candidate_score"] = _candidate_score
globals()["_threshold_label"] = _threshold_label
globals()["_scope_identifier"] = _scope_identifier
globals()["_normalized_scope_values"] = _normalized_scope_values
globals()["_flight_metadata_state"] = _flight_metadata_state
globals()["_joint_scope_keys"] = _joint_scope_keys
globals()["_scope_token_maps"] = _scope_token_maps
globals()["_apply_scope_tokens"] = _apply_scope_tokens
globals()["_restore_scope_tokens"] = _restore_scope_tokens
globals()["_sequence_to_flight"] = _sequence_to_flight
globals()["_attach_sequence_flights"] = _attach_sequence_flights
globals()["_authoritative_truth_rows"] = _authoritative_truth_rows
globals()["_summary_payload_with_scope"] = _summary_payload_with_scope
globals()["build_candidate_oracle_targets"] = build_candidate_oracle_targets

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
