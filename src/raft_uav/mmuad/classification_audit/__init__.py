"""Compatibility wrapper that scores classification on complete truth support.

The maintained implementation lives in the sibling ``classification_audit.py``
module. This package preserves the public import path while ensuring omitted
truth sequences cannot inflate the reported classification accuracy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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

_LEGACY_BUILD_MMUAD_CLASSIFICATION_AUDIT = _IMPL.build_mmuad_classification_audit


def _row_accuracy_from_sequence_truth(
    result_rows: pd.DataFrame,
    truth_by_sequence: dict[str, dict[str, Any]],
) -> float:
    """Score every truth row from its sequence prediction, counting omissions wrong."""

    predicted_by_sequence = _IMPL._sequence_class_stats(
        result_rows,
        prefix="predicted",
    )
    total = 0
    correct = 0
    for sequence, truth_stats in truth_by_sequence.items():
        truth_class = truth_stats.get("truth_class")
        truth_row_count = int(truth_stats.get("truth_row_count", 0) or 0)
        if truth_class is None or truth_row_count <= 0:
            continue
        total += truth_row_count
        predicted_class = predicted_by_sequence.get(sequence, {}).get("predicted_class")
        if predicted_class is not None and _IMPL._classes_equal(
            predicted_class,
            truth_class,
        ):
            correct += truth_row_count
    return float(correct / total) if total else np.nan


def _classification_coverage(
    truth_rows: pd.DataFrame,
    result_rows: pd.DataFrame,
) -> dict[str, int | float]:
    """Return truth-support coverage from available per-sequence predictions."""

    truth_by_sequence = _IMPL._sequence_class_stats(truth_rows, prefix="truth")
    predicted_by_sequence = _IMPL._sequence_class_stats(
        result_rows,
        prefix="predicted",
    )
    truth_row_count = 0
    matched_truth_row_count = 0
    truth_sequence_count = 0
    matched_sequence_count = 0
    for sequence, truth_stats in truth_by_sequence.items():
        truth_class = truth_stats.get("truth_class")
        sequence_rows = int(truth_stats.get("truth_row_count", 0) or 0)
        if truth_class is None or sequence_rows <= 0:
            continue
        truth_sequence_count += 1
        truth_row_count += sequence_rows
        predicted_class = predicted_by_sequence.get(sequence, {}).get("predicted_class")
        if predicted_class is not None:
            matched_sequence_count += 1
            matched_truth_row_count += sequence_rows
    return {
        "classification_truth_row_count": truth_row_count,
        "classification_matched_truth_row_count": matched_truth_row_count,
        "classification_missing_truth_row_count": truth_row_count - matched_truth_row_count,
        "classification_truth_sequence_count": truth_sequence_count,
        "classification_matched_sequence_count": matched_sequence_count,
        "classification_missing_sequence_count": truth_sequence_count
        - matched_sequence_count,
        "classification_coverage_fraction": (
            float(matched_truth_row_count / truth_row_count)
            if truth_row_count
            else np.nan
        ),
    }


def build_mmuad_classification_audit(
    *,
    truth: pd.DataFrame,
    results: pd.DataFrame,
    training_truth: pd.DataFrame | None = None,
    class_map=None,
    class_names: dict[int, str] | None = None,
):
    """Build an audit whose accuracy and baselines use the same truth support."""

    audit = _LEGACY_BUILD_MMUAD_CLASSIFICATION_AUDIT(
        truth=truth,
        results=results,
        training_truth=training_truth,
        class_map=class_map,
        class_names=class_names,
    )
    summary = dict(audit.summary)
    summary.update(
        _classification_coverage(
            _IMPL._class_rows(truth),
            _IMPL._class_rows(results),
        )
    )
    return _IMPL.MmuadClassificationAudit(
        classification_audit=audit.classification_audit,
        confusion_matrix=audit.confusion_matrix,
        sequence_class_summary=audit.sequence_class_summary,
        summary=summary,
    )


_IMPL._row_accuracy_from_sequence_truth = _row_accuracy_from_sequence_truth
_IMPL.build_mmuad_classification_audit = build_mmuad_classification_audit

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_row_accuracy_from_sequence_truth"] = _row_accuracy_from_sequence_truth
globals()["_classification_coverage"] = _classification_coverage
globals()["build_mmuad_classification_audit"] = build_mmuad_classification_audit

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
