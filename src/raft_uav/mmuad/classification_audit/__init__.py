"""Compatibility fixes for MMUAD Track 5 classification audits.

The maintained implementation lives in the sibling ``classification_audit.py``
module. This package preserves the public import path while requiring every
classification row to contain a valid official class identifier and scoring
truth/results through an exact one-to-one row alignment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "classification_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._classification_audit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load classification audit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD = _IMPL.build_mmuad_classification_audit
_MISSING_ROW = "<missing-row>"
_ALIGNMENT_METRIC_COLUMNS = (
    "row_count",
    "correct_row_count",
    "matched_row_count",
    "missing_prediction_row_count",
    "extra_prediction_row_count",
    "row_key_parity",
    "per_sequence_accuracy",
)


def _valid_class_series(values: pd.Series) -> bool:
    """Return whether every row contains a valid official class identifier."""

    normalized = pd.Series(values, copy=False)
    if normalized.empty or bool(normalized.isna().any()):
        return False
    return all(_IMPL._valid_class_id(value) for value in normalized.tolist())


def _row_keys(rows: pd.DataFrame, *, side: str) -> list[tuple[Any, ...]]:
    """Return unique one-to-one keys for sequence/timestamp rows."""

    timestamps = pd.to_numeric(rows["Timestamp"], errors="coerce").to_numpy(dtype=float)
    occurrences: dict[tuple[str, float], int] = {}
    keys: list[tuple[Any, ...]] = []
    for position, (sequence, timestamp) in enumerate(
        zip(rows["Sequence"].astype(str), timestamps, strict=True)
    ):
        sequence_text = str(sequence)
        if not np.isfinite(timestamp):
            # Keep malformed timestamps unique and side-specific. In particular,
            # a missing truth timestamp must never match a missing result timestamp.
            keys.append((f"invalid-{side}", sequence_text, int(position), 0))
            continue
        base = (sequence_text, float(timestamp))
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        keys.append(("valid", *base, occurrence))
    return keys


def _attach_row_keys(rows: pd.DataFrame, *, side: str) -> pd.DataFrame:
    keyed = rows.copy()
    keyed["_row_key"] = pd.Series(
        _row_keys(keyed, side=side),
        index=keyed.index,
        dtype=object,
    )
    return keyed


def _build_row_alignment(truth: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Align truth and predictions one-to-one without matching invalid timestamps."""

    truth_rows = _attach_row_keys(_IMPL._class_rows(truth), side="truth")
    result_rows = _attach_row_keys(_IMPL._class_rows(results), side="result")
    result_lookup = {
        key: row
        for key, (_, row) in zip(
            result_rows["_row_key"],
            result_rows.iterrows(),
            strict=True,
        )
    }
    records: list[dict[str, Any]] = []
    for _, truth_row in truth_rows.iterrows():
        key = truth_row["_row_key"]
        result_row = result_lookup.pop(key, None)
        if result_row is None:
            records.append(
                {
                    "sequence": str(truth_row["Sequence"]),
                    "truth_class": truth_row["Classification"],
                    "predicted_class": None,
                    "match_state": "truth_only",
                    "correct": False,
                }
            )
            continue
        truth_class = truth_row["Classification"]
        predicted_class = result_row["Classification"]
        records.append(
            {
                "sequence": str(truth_row["Sequence"]),
                "truth_class": truth_class,
                "predicted_class": predicted_class,
                "match_state": "both",
                "correct": bool(
                    truth_class is not None
                    and predicted_class is not None
                    and _IMPL._classes_equal(truth_class, predicted_class)
                ),
            }
        )
    for result_row in result_lookup.values():
        records.append(
            {
                "sequence": str(result_row["Sequence"]),
                "truth_class": None,
                "predicted_class": result_row["Classification"],
                "match_state": "result_only",
                "correct": False,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "sequence",
            "truth_class",
            "predicted_class",
            "match_state",
            "correct",
        ],
    )


def _sequence_alignment_metrics(alignment: pd.DataFrame) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for sequence, group in alignment.groupby("sequence", sort=True):
        states = group["match_state"]
        scored_count = int(len(group))
        correct_count = int(group["correct"].sum())
        missing_count = int(states.eq("truth_only").sum())
        extra_count = int(states.eq("result_only").sum())
        metrics[str(sequence)] = {
            "row_count": scored_count,
            "correct_row_count": correct_count,
            "matched_row_count": int(states.eq("both").sum()),
            "missing_prediction_row_count": missing_count,
            "extra_prediction_row_count": extra_count,
            "row_key_parity": missing_count == 0 and extra_count == 0,
            "per_sequence_accuracy": (
                float(correct_count / scored_count) if scored_count else np.nan
            ),
        }
    return metrics


def _update_sequence_frame(
    frame: pd.DataFrame,
    metrics: dict[str, dict[str, Any]],
    *,
    append_missing_sequences: bool,
) -> pd.DataFrame:
    updated = frame.copy()
    if "sequence" not in updated.columns:
        return updated

    sequences = updated["sequence"].astype(str)
    for column in _ALIGNMENT_METRIC_COLUMNS:
        updated[column] = [metrics.get(sequence, {}).get(column, np.nan) for sequence in sequences]
    updated["sequence_accuracy"] = updated["per_sequence_accuracy"]

    if not append_missing_sequences:
        return updated

    existing = set(sequences.tolist())
    missing_sequences = sorted(set(metrics) - existing)
    if not missing_sequences:
        return updated

    appended_rows: list[dict[str, Any]] = []
    for sequence in missing_sequences:
        row = {column: np.nan for column in updated.columns}
        row["sequence"] = sequence
        row.update(metrics[sequence])
        row["sequence_accuracy"] = metrics[sequence]["per_sequence_accuracy"]
        appended_rows.append(row)
    return pd.concat(
        [updated, pd.DataFrame.from_records(appended_rows, columns=updated.columns)],
        ignore_index=True,
    )


def _aligned_confusion_matrix(
    alignment: pd.DataFrame,
    *,
    class_names: dict[int, str],
) -> pd.DataFrame:
    columns = [
        "ground_truth_class",
        "predicted_class",
        "count",
        "ground_truth_class_string",
        "predicted_class_string",
        "sequence_count",
    ]
    if alignment.empty:
        return pd.DataFrame(columns=columns)
    rows = alignment.copy()
    rows["ground_truth_class"] = [
        _MISSING_ROW if state == "result_only" else _IMPL._class_text(value)
        for state, value in zip(rows["match_state"], rows["truth_class"], strict=True)
    ]
    rows["predicted_class_text"] = [
        _MISSING_ROW if state == "truth_only" else _IMPL._class_text(value)
        for state, value in zip(rows["match_state"], rows["predicted_class"], strict=True)
    ]
    grouped = (
        rows.groupby(["ground_truth_class", "predicted_class_text"], dropna=False)["sequence"]
        .agg(row_count="count", sequence_count=lambda values: int(values.nunique()))
        .reset_index()
    )
    records: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        truth_text = str(row["ground_truth_class"])
        predicted_text = str(row["predicted_class_text"])
        truth_value = _IMPL._normal_class_value(truth_text)
        predicted_value = _IMPL._normal_class_value(predicted_text)
        records.append(
            {
                "ground_truth_class": truth_text,
                "predicted_class": predicted_text,
                "count": int(row["row_count"]),
                "ground_truth_class_string": (
                    ""
                    if truth_text == _MISSING_ROW
                    else _IMPL._class_string(truth_value, class_names)
                ),
                "predicted_class_string": (
                    ""
                    if predicted_text == _MISSING_ROW
                    else _IMPL._class_string(predicted_value, class_names)
                ),
                "sequence_count": int(row["sequence_count"]),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["ground_truth_class", "predicted_class"]
    ).reset_index(drop=True)


def _default_label_explains_score(summary: dict[str, Any], current_accuracy: float) -> bool:
    constant_accuracy = summary.get("constant_prediction_accuracy")
    constant_prediction = str(summary.get("submission_constant_prediction", ""))
    return bool(
        constant_prediction
        and np.isfinite(current_accuracy)
        and constant_accuracy is not None
        and np.isfinite(float(constant_accuracy))
        and abs(current_accuracy - float(constant_accuracy)) <= 1.0e-12
    )


def build_mmuad_classification_audit(
    *,
    truth: pd.DataFrame,
    results: pd.DataFrame,
    training_truth: pd.DataFrame | None = None,
    class_map: Mapping[str, Any] | None = None,
    class_names: dict[int, str] | None = None,
) -> Any:
    """Build an audit whose accuracy cannot ignore missing or extra result rows."""

    resolved_class_names = dict(class_names or _IMPL.OFFICIAL_TRACK5_CLASS_NAMES)
    audit = _ORIGINAL_BUILD(
        truth=truth,
        results=results,
        training_truth=training_truth,
        class_map=class_map,
        class_names=resolved_class_names,
    )
    alignment = _build_row_alignment(truth, results)
    metrics = _sequence_alignment_metrics(alignment)
    summary = dict(audit.summary)

    scored_count = int(len(alignment))
    correct_count = int(alignment["correct"].sum()) if scored_count else 0
    matched_count = int(alignment["match_state"].eq("both").sum())
    missing_count = int(alignment["match_state"].eq("truth_only").sum())
    extra_count = int(alignment["match_state"].eq("result_only").sum())
    row_key_parity = missing_count == 0 and extra_count == 0

    if row_key_parity:
        # Preserve every legacy table and metric for already well-formed
        # submissions. The compatibility layer only changes malformed row sets.
        sequence_summary = audit.sequence_class_summary.copy()
        classification_audit = audit.classification_audit.copy()
        confusion = audit.confusion_matrix.copy()
        current_accuracy = float(summary.get("current_accuracy", np.nan))
        default_explains = bool(summary.get("default_label_explains_score", False))
    else:
        sequence_summary = _update_sequence_frame(
            audit.sequence_class_summary,
            metrics,
            append_missing_sequences=True,
        )
        classification_audit = _update_sequence_frame(
            audit.classification_audit,
            metrics,
            append_missing_sequences=False,
        )
        confusion = _aligned_confusion_matrix(alignment, class_names=resolved_class_names)
        current_accuracy = float(correct_count / scored_count) if scored_count else np.nan
        default_explains = _default_label_explains_score(summary, current_accuracy)
        if "default_label_explains_score" in sequence_summary.columns:
            sequence_summary["default_label_explains_score"] = default_explains

    summary.update(
        {
            "current_accuracy": current_accuracy,
            "classification_scored_row_count": scored_count,
            "matched_row_count": matched_count,
            "missing_prediction_row_count": missing_count,
            "extra_prediction_row_count": extra_count,
            "row_key_parity": row_key_parity,
            "default_label_explains_score": default_explains,
        }
    )
    return _IMPL.MmuadClassificationAudit(
        classification_audit=classification_audit,
        confusion_matrix=confusion,
        sequence_class_summary=sequence_summary,
        summary=summary,
    )


_IMPL._valid_class_series = _valid_class_series
_IMPL.build_mmuad_classification_audit = build_mmuad_classification_audit

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_valid_class_series"] = _valid_class_series
globals()["build_mmuad_classification_audit"] = build_mmuad_classification_audit
globals()["_build_row_alignment"] = _build_row_alignment

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
