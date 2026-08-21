"""Reject malformed timestamps in valid Track 5 template-resampling rows."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import pandas as pd

from raft_uav.mmuad.submission import parse_official_sequence_cell
from raft_uav.numeric import optional_float

_PATCH_MARKER = "_raft_uav_validates_track5_resample_template_timestamps"


def _validate_template_timestamps(template: pd.DataFrame, *, resample_module: Any) -> None:
    """Require a valid timestamp whenever a template row has a valid sequence id."""

    rows = pd.DataFrame(template).copy()
    if rows.empty:
        return
    sequence_column = resample_module._first_present(
        rows,
        resample_module.SEQUENCE_ALIASES,
    )
    time_column = resample_module._first_present(
        rows,
        resample_module.TIME_ALIASES,
    )
    if sequence_column is None or time_column is None:
        return

    for row_label, sequence_value, time_value in zip(
        rows.index,
        rows[sequence_column],
        rows[time_column],
        strict=True,
    ):
        try:
            parse_official_sequence_cell(sequence_value)
        except (TypeError, ValueError):
            # Preserve the resampler's established cleanup of missing sequence rows.
            continue
        if optional_float(time_value) is None:
            raise ValueError(
                f"template contains an invalid timestamp at row {row_label!r}: "
                f"{time_value!r}"
            )


def install() -> None:
    """Install strict timestamp validation at the resampling boundary."""

    from raft_uav.mmuad import track5_template_resample as resample_module

    original: Callable[..., pd.DataFrame] = resample_module._normalize_template_rows
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
        normalized = original(template)
        _validate_template_timestamps(template, resample_module=resample_module)
        return normalized

    setattr(normalize_template_rows, _PATCH_MARKER, True)
    resample_module._normalize_template_rows = normalize_template_rows
    implementation = getattr(resample_module, "_IMPL", None)
    if implementation is not None:
        implementation._normalize_template_rows = normalize_template_rows
