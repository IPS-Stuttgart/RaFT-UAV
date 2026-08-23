"""Validate Track 5 RTS template rows and clone duplicate output rows safely."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import parse_official_sequence_cell
from raft_uav.numeric import optional_float

_TEMPLATE_KEY = ("sequence_id", "time_s")


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    """Normalize every requested template row or fail with row context."""

    from raft_uav.mmuad import track5_rts_ensemble as rts

    rows = pd.DataFrame(template).copy()
    if rows.empty:
        return pd.DataFrame(columns=list(_TEMPLATE_KEY))
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
        .sort_values(list(_TEMPLATE_KEY), kind="mergesort")
        .reset_index(drop=True)
    )


def _exact_template_time_matches(values: pd.Series, target: float) -> np.ndarray:
    """Match the exact timestamp copied into rows during template resampling."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.isfinite(numeric) & (numeric == float(target))


def _expand_unique_rows_to_template(
    rows: pd.DataFrame,
    template_rows: pd.DataFrame,
    *,
    role: str,
) -> pd.DataFrame:
    """Clone one computed row onto every matching requested template row."""

    result = pd.DataFrame(rows).copy()
    missing = [column for column in _TEMPLATE_KEY if column not in result.columns]
    if missing:
        raise RuntimeError(f"{role} rows are missing template keys: {missing!r}")

    source_index = pd.MultiIndex.from_frame(result.loc[:, list(_TEMPLATE_KEY)])
    if source_index.has_duplicates:
        raise RuntimeError(f"{role} rows contain duplicate unique-template keys")

    target_index = pd.MultiIndex.from_frame(
        template_rows.loc[:, list(_TEMPLATE_KEY)]
    )
    row_positions = source_index.get_indexer(target_index)
    if bool(np.any(row_positions < 0)):
        missing_keys = target_index[row_positions < 0].tolist()
        raise RuntimeError(
            f"{role} rows are missing requested template keys: {missing_keys!r}"
        )

    expanded = result.iloc[row_positions].reset_index(drop=True)
    for column in _TEMPLATE_KEY:
        expanded[column] = template_rows[column].to_numpy(copy=True)
    return expanded


def _duplicate_safe_build(
    base: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
) -> Callable[..., tuple[pd.DataFrame, pd.DataFrame]]:
    """Compute each physical timestamp once, then clone requested output rows."""

    if getattr(base, "_raft_uav_duplicate_template_safe", False):
        return base

    @wraps(base)
    def build(
        estimate_inputs: Any,
        template: pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        template_rows = _normalize_template_rows(template)
        duplicated = template_rows.duplicated(
            subset=list(_TEMPLATE_KEY),
            keep=False,
        )
        if not bool(duplicated.any()):
            return base(estimate_inputs, template, *args, **kwargs)

        unique_template = (
            template_rows.drop_duplicates(
                subset=list(_TEMPLATE_KEY),
                keep="first",
            )
            .reset_index(drop=True)
        )
        unique_estimates, unique_diagnostics = base(
            estimate_inputs,
            unique_template,
            *args,
            **kwargs,
        )
        estimates = _expand_unique_rows_to_template(
            unique_estimates,
            template_rows,
            role="RTS estimate",
        )
        diagnostics = _expand_unique_rows_to_template(
            unique_diagnostics,
            template_rows,
            role="RTS diagnostic",
        )
        return estimates, diagnostics

    build._raft_uav_duplicate_template_safe = True  # type: ignore[attr-defined]
    return build


def install() -> None:
    """Install strict template validation and duplicate-safe RTS construction."""

    from raft_uav.mmuad import track5_rts_ensemble as rts

    rts._normalize_template_rows = _normalize_template_rows
    rts._time_matches = _exact_template_time_matches
    implementation: Any = getattr(rts, "_IMPL", None)
    if implementation is not None:
        implementation._normalize_template_rows = _normalize_template_rows
        implementation._time_matches = _exact_template_time_matches

    duplicate_safe_build = _duplicate_safe_build(
        rts.build_track5_rts_ensemble
    )
    rts.build_track5_rts_ensemble = duplicate_safe_build
    if implementation is not None:
        implementation.build_track5_rts_ensemble = duplicate_safe_build
