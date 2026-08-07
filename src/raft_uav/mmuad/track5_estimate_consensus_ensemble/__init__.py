"""Compatibility fixes for Track 5 consensus input and template validation.

The maintained implementation lives in the sibling
``track5_estimate_consensus_ensemble.py`` module. This package preserves the
public import path while rejecting ambiguous estimate labels, making template
alias lookup insensitive to surrounding whitespace, and rejecting ambiguous
columns or malformed rows before sequence/time alignment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_estimate_consensus_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_consensus_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load Track 5 consensus ensemble implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _normalized_column_name(value: object) -> str:
    """Return the whitespace-insensitive, case-folded column name."""

    return str(value).strip().casefold()


def _first_present(rows: Any, names: tuple[str, ...]) -> Any | None:
    """Return the unique original column matching one of the supplied aliases."""

    aliases = {_normalized_column_name(name) for name in names}
    matching_columns = [
        column
        for column in rows.columns
        if _normalized_column_name(column) in aliases
    ]
    if len(matching_columns) > 1:
        rendered = ", ".join(repr(str(column)) for column in matching_columns)
        raise ValueError(
            "template contains ambiguous columns matching "
            f"{tuple(names)!r}: {rendered}"
        )
    if matching_columns:
        return matching_columns[0]
    return None


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    """Normalize every requested template row or reject the first malformed row."""

    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(
        rows,
        ("sequence_id", "Sequence", "sequence", "seq"),
    )
    time_column = _first_present(
        rows,
        ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"),
    )
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")

    sequence_ids: list[str] = []
    timestamps: list[float] = []
    for row_label, sequence_value, time_value in zip(
        rows.index,
        rows[sequence_column],
        rows[time_column],
        strict=True,
    ):
        try:
            sequence_id = _IMPL.parse_official_sequence_cell(sequence_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "template contains an invalid sequence identifier at "
                f"row {row_label!r}: {sequence_value!r}"
            ) from exc
        timestamp = optional_float(time_value)
        if timestamp is None:
            raise ValueError(
                f"template contains an invalid timestamp at row {row_label!r}: "
                f"{time_value!r}"
            )
        sequence_ids.append(sequence_id)
        timestamps.append(timestamp)

    return (
        pd.DataFrame(
            {
                "sequence_id": sequence_ids,
                "time_s": timestamps,
            }
        )
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


def _template_time_matches(values: pd.Series, target: float):
    """Match rows copied from the template by exact timestamp identity."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric == float(target)


def _validate_unique_estimate_labels(estimate_inputs: Any) -> tuple[Any, ...]:
    """Materialize estimate inputs and reject ambiguous normalized labels."""

    loaded_inputs = tuple(estimate_inputs)
    seen_labels: dict[str, str] = {}
    for raw_label, _, _ in loaded_inputs:
        raw_text = str(raw_label)
        safe_label = _IMPL._safe_label(raw_text)
        previous = seen_labels.get(safe_label)
        if previous is not None:
            if previous == raw_text:
                raise ValueError(f"estimate input label {safe_label!r} is duplicated")
            raise ValueError(
                "estimate input labels collide after normalization: "
                f"{previous!r} and {raw_text!r} both normalize to {safe_label!r}"
            )
        seen_labels[safe_label] = raw_text
    return loaded_inputs


_ORIGINAL_BUILD_TRACK5_CONSENSUS_ESTIMATE_ENSEMBLE = (
    _IMPL.build_track5_consensus_estimate_ensemble
)


def _build_track5_consensus_estimate_ensemble(
    estimate_inputs: Any,
    template: pd.DataFrame,
    *,
    consensus_radius_m: float = 5.0,
    fallback_policy: str = "max-weight",
    min_consensus_weight_fraction: float = 0.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the ensemble after validating normalized input-label identity."""

    loaded_inputs = _validate_unique_estimate_labels(estimate_inputs)
    return _ORIGINAL_BUILD_TRACK5_CONSENSUS_ESTIMATE_ENSEMBLE(
        loaded_inputs,
        template,
        consensus_radius_m=consensus_radius_m,
        fallback_policy=fallback_policy,
        min_consensus_weight_fraction=min_consensus_weight_fraction,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )


_IMPL._template_time_matches = _template_time_matches
_IMPL._first_present = _first_present
_IMPL._normalize_template_rows = _normalize_template_rows
_IMPL.build_track5_consensus_estimate_ensemble = (
    _build_track5_consensus_estimate_ensemble
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep the patched helpers visible to tests and exploratory callers.
globals()["_template_time_matches"] = _template_time_matches
globals()["_normalized_column_name"] = _normalized_column_name
globals()["_first_present"] = _first_present
globals()["_normalize_template_rows"] = _normalize_template_rows
globals()["_validate_unique_estimate_labels"] = _validate_unique_estimate_labels
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
