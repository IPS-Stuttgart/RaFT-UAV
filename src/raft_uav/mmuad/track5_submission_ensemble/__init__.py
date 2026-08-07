"""Compatibility fixes for Track 5 submission ensembling.

The maintained implementation lives in the sibling
``track5_submission_ensemble.py`` module. This package preserves the public
import path while rejecting ambiguous CSV headers, preventing malformed
normalized numeric and classification values from being silently dropped or
truncated, rejecting missing normalized sequence identifiers, and keeping
weighted ensemble arithmetic finite for very large non-negative weights.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.class_probability_context import OFFICIAL_CLASS_LABELS

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_submission_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_submission_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load Track 5 submission-ensemble implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_READ_TRACK5_SUBMISSION_CSV = _IMPL._read_track5_submission_csv
_ORIGINAL_NORMALIZE_INTERNAL_SUBMISSION_ROWS = (
    _IMPL._normalize_internal_submission_rows
)
_ORIGINAL_ENSEMBLE_TRACK5_SUBMISSIONS = _IMPL.ensemble_track5_submissions


def _normalized_column_name(value: object) -> str:
    """Return the whitespace-insensitive, case-folded column name."""

    return str(value).strip().casefold()


def _normalized_column_lookup(rows: pd.DataFrame) -> dict[str, Any]:
    """Return unique normalized columns or reject ambiguous physical names."""

    lookup: dict[str, Any] = {}
    for column in rows.columns:
        normalized = _normalized_column_name(column)
        if normalized in lookup:
            first = lookup[normalized]
            raise ValueError(
                "ambiguous Track 5 submission columns after whitespace/case "
                f"normalization for {normalized!r}: {first!r}, {column!r}"
            )
        lookup[normalized] = column
    return lookup


def _validate_physical_submission_header(source: Any) -> None:
    """Reject duplicate physical CSV headers before pandas mangles their names."""

    rewind_position: int | None = None
    if not isinstance(source, (str, Path)):
        try:
            rewind_position = int(source.tell())
        except (AttributeError, OSError, TypeError, ValueError):
            return

    try:
        try:
            physical_header = pd.read_csv(
                source,
                header=None,
                nrows=1,
                dtype=str,
                keep_default_na=False,
            )
        except TypeError:
            physical_header = pd.read_csv(source, header=None, nrows=1)
    finally:
        if rewind_position is not None:
            try:
                source.seek(rewind_position)
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Track 5 submission stream could not be rewound after header validation"
                ) from exc

    if physical_header.empty:
        return
    _normalized_column_lookup(
        pd.DataFrame(columns=physical_header.iloc[0].tolist())
    )


def _read_track5_submission_csv(source: Any) -> pd.DataFrame:
    """Read Track 5 CSV data after validating its physical header."""

    _validate_physical_submission_header(source)
    rows = _ORIGINAL_READ_TRACK5_SUBMISSION_CSV(source)
    _normalized_column_lookup(rows)
    return rows


def _invalid_row_summary(index: pd.Index, invalid: np.ndarray) -> str:
    positions = np.flatnonzero(np.asarray(invalid, dtype=bool))
    labels = [repr(index[int(position)]) for position in positions[:5]]
    suffix = ", ..." if len(positions) > 5 else ""
    return ", ".join(labels) + suffix


def _raise_invalid_normalized_rows(
    *,
    source_path: Path,
    index: pd.Index,
    invalid: np.ndarray,
    field: str,
) -> None:
    count = int(np.count_nonzero(invalid))
    sample = _invalid_row_summary(index, invalid)
    raise ValueError(
        f"{source_path} contains {count} invalid normalized Track 5 {field} "
        f"row(s) at index/indices {sample}"
    )


def _normalize_internal_submission_rows(
    rows: pd.DataFrame,
    *,
    source_path: Path,
) -> pd.DataFrame:
    """Reject malformed normalized rows before the legacy loader can drop them."""

    frame = pd.DataFrame(rows).copy()
    lookup = _normalized_column_lookup(frame)
    sequence_column = lookup.get("sequence_id")
    if sequence_column is not None:
        sequence_text = frame[sequence_column].astype("string").str.strip()
        invalid_sequence = (sequence_text.isna() | sequence_text.eq("")).fillna(True)
        if bool(invalid_sequence.any()):
            _raise_invalid_normalized_rows(
                source_path=source_path,
                index=frame.index,
                invalid=invalid_sequence.to_numpy(dtype=bool),
                field="sequence_id",
            )

    classification_column = _IMPL._normalized_classification_column(lookup)
    measurement_columns = ("time_s", "state_x_m", "state_y_m", "state_z_m")
    if classification_column is not None and all(
        column in lookup for column in measurement_columns
    ):
        measurements = pd.DataFrame(
            {
                column: pd.to_numeric(frame[lookup[column]], errors="coerce")
                for column in measurement_columns
            },
            index=frame.index,
        )
        for column in measurement_columns:
            invalid = ~np.isfinite(measurements[column].to_numpy(dtype=float))
            if bool(invalid.any()):
                _raise_invalid_normalized_rows(
                    source_path=source_path,
                    index=frame.index,
                    invalid=invalid,
                    field=column,
                )

        raw_classes = frame[classification_column]
        normalized_classes = _IMPL._predicted_class_labels(raw_classes)
        valid_classes = normalized_classes.isin(OFFICIAL_CLASS_LABELS)
        if not bool(valid_classes.all()):
            examples = sorted(
                {repr(value) for value in raw_classes.loc[~valid_classes].tolist()}
            )
            raise ValueError(
                "invalid normalized Track 5 Classification values in "
                f"{source_path}: {', '.join(examples)}"
            )

    return _ORIGINAL_NORMALIZE_INTERNAL_SUBMISSION_ROWS(
        frame,
        source_path=source_path,
    )


def ensemble_track5_submissions(
    submissions: Iterable[object],
    *,
    class_policy: str = "weighted-vote",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensemble submissions without overflowing finite non-negative weights.

    Only relative weights affect the ensemble. Converting them to probabilities
    before the legacy implementation prevents both the weight sum and weighted
    position numerator from overflowing. Zero-weight grid entries remain valid,
    while the complete weight vector must retain positive mass. Diagnostic sums
    and vote margins are converted back to the original weight scale afterwards.
    """

    inputs = tuple(submissions)
    if not inputs:
        return _ORIGINAL_ENSEMBLE_TRACK5_SUBMISSIONS(
            inputs,
            class_policy=class_policy,
        )

    weights = np.asarray([float(item.weight) for item in inputs], dtype=float)
    if not np.isfinite(weights).all() or bool(np.any(weights < 0.0)):
        raise ValueError("submission weights must be non-negative and finite")

    scale = float(np.max(weights))
    if scale <= 0.0:
        raise ValueError("submission weights must have positive finite mass")
    scaled_weights = weights / scale
    scaled_total = float(np.sum(scaled_weights))
    if not np.isfinite(scaled_total) or scaled_total <= 0.0:
        raise ValueError("submission weights must have positive finite mass")
    normalized_weights = scaled_weights / scaled_total

    normalized_inputs = tuple(
        _IMPL.SubmissionInput(
            label=item.label,
            path=item.path,
            weight=float(weight),
        )
        for item, weight in zip(inputs, normalized_weights, strict=True)
    )
    estimates, diagnostics = _ORIGINAL_ENSEMBLE_TRACK5_SUBMISSIONS(
        normalized_inputs,
        class_policy=class_policy,
    )

    raw_total = scale * scaled_total
    if "ensemble_weight_sum" in estimates.columns:
        estimates["ensemble_weight_sum"] = raw_total
    if "weight_sum" in diagnostics.columns:
        diagnostics["weight_sum"] = raw_total
    if "classification_vote_margin" in diagnostics.columns:
        margins = diagnostics["classification_vote_margin"].to_numpy(dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            margins = (margins * scale) * scaled_total
        diagnostics["classification_vote_margin"] = margins

    return estimates, diagnostics


_IMPL._normalized_column_lookup = _normalized_column_lookup
_IMPL._read_track5_submission_csv = _read_track5_submission_csv
_IMPL._normalize_internal_submission_rows = _normalize_internal_submission_rows
_IMPL.ensemble_track5_submissions = ensemble_track5_submissions

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalized_column_name"] = _normalized_column_name
globals()["_normalized_column_lookup"] = _normalized_column_lookup
globals()["_validate_physical_submission_header"] = (
    _validate_physical_submission_header
)
globals()["_read_track5_submission_csv"] = _read_track5_submission_csv
globals()["_invalid_row_summary"] = _invalid_row_summary
globals()["_raise_invalid_normalized_rows"] = _raise_invalid_normalized_rows
globals()["_normalize_internal_submission_rows"] = (
    _normalize_internal_submission_rows
)
globals()["ensemble_track5_submissions"] = ensemble_track5_submissions

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
