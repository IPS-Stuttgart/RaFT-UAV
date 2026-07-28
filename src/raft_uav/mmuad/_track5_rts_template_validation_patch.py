"""Reject malformed Track 5 RTS template rows before smoothing."""

from __future__ import annotations

from typing import Any

import pandas as pd

from raft_uav.mmuad.submission import parse_official_sequence_cell
from raft_uav.numeric import optional_float


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    """Normalize every requested template row or fail with row context."""

    from raft_uav.mmuad import track5_rts_ensemble as rts

    rows = pd.DataFrame(template).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s"])
    sequence_column = rts._first_present(
        rows,
        ("sequence_id", "Sequence", "sequence", "seq"),
    )
    time_column = rts._first_present(
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
            sequence_id = parse_official_sequence_cell(sequence_value)
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


def install() -> None:
    """Install strict RTS template validation on public and legacy modules."""

    from raft_uav.mmuad import track5_rts_ensemble as rts

    rts._normalize_template_rows = _normalize_template_rows
    implementation: Any = getattr(rts, "_IMPL", None)
    if implementation is not None:
        implementation._normalize_template_rows = _normalize_template_rows
