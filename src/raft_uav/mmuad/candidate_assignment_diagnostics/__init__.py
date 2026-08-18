"""Compatibility fixes for candidate-assignment parsing and physical-flight scoping."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from raft_uav.numeric import optional_int as _safe_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_assignment_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_assignment_diagnostics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load candidate-assignment diagnostics implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_ASSIGNMENT_DIAGNOSTICS = (
    _IMPL.build_candidate_assignment_diagnostics
)
_SCOPE_TOKEN_PREFIX = "__raft_uav_candidate_assignment_scope_"
_MISSING_SCOPE_TEXT = frozenset({"", "nan", "none", "<na>", "nat"})


def _assignment_weights(group: pd.DataFrame) -> np.ndarray:
    """Return finite normalized assignment weights.

    Malformed, negative, NaN, and infinite weights carry no usable probability
    mass. If no positive finite mass remains, fall back to a uniform distribution.
    """

    if "mixture_final_weight" in group.columns:
        weights = pd.to_numeric(
            group["mixture_final_weight"], errors="coerce"
        ).to_numpy(dtype=float)
    elif "mixture_dominant" in group.columns:
        weights = np.asarray(
            [
                _IMPL._parse_mixture_dominant_flag(value)
                for value in group["mixture_dominant"]
            ],
            dtype=float,
        )
    else:
        weights = np.ones(len(group), dtype=float)

    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    scale = float(np.max(weights, initial=0.0))
    if scale <= 0.0:
        return np.ones(len(group), dtype=float) / max(float(len(group)), 1.0)

    scaled = weights / scale
    scaled_total = float(np.sum(scaled))
    if not np.isfinite(scaled_total) or scaled_total <= 0.0:
        return np.ones(len(group), dtype=float) / max(float(len(group)), 1.0)
    if scale <= 1.0e-12 / scaled_total:
        return np.ones(len(group), dtype=float) / max(float(len(group)), 1.0)
    return scaled / scaled_total


def _canonical_scope_id(value: object, *, field: str) -> str | None:
    """Return one normalized scalar scope identifier or ``None`` when missing."""

    if not is_scalar(value):
        raise ValueError(f"{field} values must be scalar")
    if value is None:
        return None
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return None
    text = str(value).strip()
    return None if text.casefold() in _MISSING_SCOPE_TEXT else text


def _final_truth_snapshots(
    truth: pd.DataFrame,
    *,
    scope_by_flight: bool = False,
) -> pd.DataFrame:
    """Return the final finite truth row for each applicable scope timestamp."""

    rows = pd.DataFrame(truth).copy()
    order_column = "__raft_uav_truth_input_order__"
    while order_column in rows.columns:
        order_column = f"_{order_column}"
    rows[order_column] = np.arange(len(rows), dtype=np.int64)
    normalized = _IMPL.normalize_truth_columns(rows)
    if normalized.empty:
        return normalized.drop(columns=[order_column], errors="ignore")
    identity_columns = ["sequence_id", "time_s"]
    if scope_by_flight and "flight_id" in normalized.columns:
        identity_columns.insert(1, "flight_id")
    return (
        normalized.sort_values(
            ["sequence_id", "time_s", order_column],
            kind="mergesort",
        )
        .drop_duplicates(subset=identity_columns, keep="last")
        .drop(columns=[order_column])
        .reset_index(drop=True)
    )


def _scope_rows_by_flight(
    assignments: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[str, str | None]] | None]:
    """Replace sequence IDs with joint sequence/flight tokens when possible."""

    assignment_rows = pd.DataFrame(assignments).copy()
    truth_rows = pd.DataFrame(truth).copy()
    if (
        assignment_rows.empty
        or truth_rows.empty
        or "flight_id" not in assignment_rows.columns
        or "flight_id" not in truth_rows.columns
    ):
        return assignment_rows, truth_rows, None
    if (
        "sequence_id" not in assignment_rows.columns
        or "sequence_id" not in truth_rows.columns
    ):
        return assignment_rows, truth_rows, None

    def scope_keys(frame: pd.DataFrame) -> list[tuple[str, str | None]]:
        keys: list[tuple[str, str | None]] = []
        for sequence_id, flight_id in zip(
            frame["sequence_id"].tolist(),
            frame["flight_id"].tolist(),
            strict=True,
        ):
            keys.append(
                (
                    str(sequence_id),
                    _canonical_scope_id(flight_id, field="flight_id"),
                )
            )
        return keys

    assignment_keys = scope_keys(assignment_rows)
    truth_keys = scope_keys(truth_rows)
    tokens: dict[tuple[str, str | None], str] = {}
    metadata: dict[str, tuple[str, str | None]] = {}

    def token_for(key: tuple[str, str | None]) -> str:
        token = tokens.get(key)
        if token is None:
            token = f"{_SCOPE_TOKEN_PREFIX}{len(tokens)}"
            tokens[key] = token
            metadata[token] = key
        return token

    assignment_rows["sequence_id"] = [token_for(key) for key in assignment_keys]
    truth_rows["sequence_id"] = [token_for(key) for key in truth_keys]
    return assignment_rows, truth_rows, metadata


def _restore_scope_columns(
    rows: pd.DataFrame,
    metadata: dict[str, tuple[str, str | None]] | None,
) -> pd.DataFrame:
    """Restore public sequence and flight identifiers after temporary scoping."""

    frame = pd.DataFrame(rows).copy()
    if metadata is None or frame.empty or "sequence_id" not in frame.columns:
        return frame

    restored_sequence: list[str] = []
    restored_flight: list[object] = []
    for value in frame["sequence_id"].astype(str).tolist():
        if value == "__pooled__":
            restored_sequence.append(value)
            restored_flight.append("__pooled__")
            continue
        scope = metadata.get(value)
        if scope is None:  # pragma: no cover - internal contract guard
            raise RuntimeError(
                f"candidate assignment returned unknown internal scope token {value!r}"
            )
        sequence_id, flight_id = scope
        restored_sequence.append(sequence_id)
        restored_flight.append(pd.NA if flight_id is None else flight_id)
    frame["sequence_id"] = restored_sequence
    frame["flight_id"] = restored_flight
    return frame


def build_candidate_assignment_diagnostics(
    assignments: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    config: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build diagnostics within each physical-flight scope when available."""

    assignment_rows = pd.DataFrame(assignments).copy()
    raw_truth = pd.DataFrame(truth).copy()
    scope_by_flight = (
        "flight_id" in assignment_rows.columns and "flight_id" in raw_truth.columns
    )
    truth_rows = _final_truth_snapshots(
        raw_truth,
        scope_by_flight=scope_by_flight,
    )
    scoped_assignments, scoped_truth, metadata = _scope_rows_by_flight(
        assignment_rows,
        truth_rows,
    )
    frames, summary = _ORIGINAL_BUILD_CANDIDATE_ASSIGNMENT_DIAGNOSTICS(
        scoped_assignments,
        scoped_truth,
        config=config,
    )
    return (
        _restore_scope_columns(frames, metadata),
        _restore_scope_columns(summary, metadata),
    )


# Candidate ranks are integer identifiers. The legacy float round-trip silently
# truncated fractional values and lost precision for integers above 2**53.
_IMPL._safe_int = _safe_int
_IMPL._assignment_weights = _assignment_weights
_IMPL._canonical_scope_id = _canonical_scope_id
_IMPL._final_truth_snapshots = _final_truth_snapshots
_IMPL._scope_rows_by_flight = _scope_rows_by_flight
_IMPL._restore_scope_columns = _restore_scope_columns
_IMPL.build_candidate_assignment_diagnostics = build_candidate_assignment_diagnostics

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_safe_int"] = _safe_int
globals()["_assignment_weights"] = _assignment_weights
globals()["_canonical_scope_id"] = _canonical_scope_id
globals()["_final_truth_snapshots"] = _final_truth_snapshots
globals()["_scope_rows_by_flight"] = _scope_rows_by_flight
globals()["_restore_scope_columns"] = _restore_scope_columns
globals()["build_candidate_assignment_diagnostics"] = (
    build_candidate_assignment_diagnostics
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
