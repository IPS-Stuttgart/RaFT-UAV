"""Reject conflicting duplicate sequence labels before calibration map coercion."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_PATCH_MARKER = "_raft_uav_rejects_conflicting_calibration_labels"


def _conflicting_sequence_labels(
    module: Any,
    labels: object,
    *,
    label_column: str | None = None,
) -> dict[str, list[str]]:
    """Return sequence identifiers assigned more than one distinct class label."""

    if isinstance(labels, Mapping):
        return {}
    rows = pd.DataFrame(labels).copy()
    try:
        sequence_column = module.resolve_sequence_column(rows)
    except ValueError:
        return {}
    resolved_label_column = label_column
    if resolved_label_column is None:
        resolved_label_column = next(
            (
                candidate
                for candidate in module._LABEL_COLUMNS
                if candidate in rows.columns
            ),
            None,
        )
    if resolved_label_column is None or resolved_label_column not in rows.columns:
        return {}

    pairs = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].map(str),
            "class_label": rows[resolved_label_column].map(str),
        }
    )
    conflicts: dict[str, list[str]] = {}
    for sequence_id, group in pairs.groupby("sequence_id", sort=False):
        distinct_labels = sorted(set(group["class_label"]))
        if len(distinct_labels) > 1:
            conflicts[str(sequence_id)] = distinct_labels
    return conflicts


def _reject_conflicting_sequence_labels(
    module: Any,
    labels: object,
    *,
    label_column: str | None = None,
) -> None:
    conflicts = _conflicting_sequence_labels(
        module,
        labels,
        label_column=label_column,
    )
    if not conflicts:
        return
    details = ", ".join(
        f"{sequence_id}={conflicts[sequence_id]!r}"
        for sequence_id in sorted(conflicts)
    )
    raise ValueError(
        "class-label table contains conflicting labels for sequence identifiers: "
        + details
    )


def _wrap_normalize_label_map(module: Any) -> None:
    original: Callable[..., Any] = module.normalize_label_map
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def validated(
        labels: object,
        *,
        label_column: str | None = None,
    ) -> dict[str, str]:
        _reject_conflicting_sequence_labels(
            module,
            labels,
            label_column=label_column,
        )
        return original(labels, label_column=label_column)

    setattr(validated, _PATCH_MARKER, True)
    module.normalize_label_map = validated


def _wrap_label_file_loader(module: Any) -> None:
    original: Callable[..., Any] = module._load_labels_preserving_ids
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def validated(path: Path) -> dict[str, str]:
        rows = module._read_csv_as_strings(Path(path))
        _reject_conflicting_sequence_labels(module, rows)
        return original(path)

    setattr(validated, _PATCH_MARKER, True)
    module._load_labels_preserving_ids = validated


def install() -> None:
    """Install conflict guards on programmatic and CSV calibration labels."""

    from raft_uav.mmuad import class_probability_calibration

    _wrap_normalize_label_map(class_probability_calibration)
    _wrap_label_file_loader(class_probability_calibration)
