"""Compatibility fixes for backward association repair.

The maintained implementation lives in the sibling ``runtime_modes.py`` module.
This package preserves the public import path while using timestamp grouping when
frame-index metadata are incomplete and scoping repair anchors by sequence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "runtime_modes.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.research._runtime_modes_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load runtime modes from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)


def _has_complete_frame_index(frame: pd.DataFrame) -> bool:
    """Return whether every row has a numeric finite frame index."""

    if frame.empty or "frame_index" not in frame.columns:
        return False
    values = pd.to_numeric(frame["frame_index"], errors="coerce").to_numpy(
        dtype=float
    )
    return bool(np.isfinite(values).all())


def _frame_groups(
    frame: pd.DataFrame,
    *,
    use_frame_index: bool | None = None,
) -> list[tuple[tuple[str, int | float], pd.DataFrame]]:
    """Group all valid frames with a key compatible with selected-row keys."""

    if frame.empty:
        return []
    if use_frame_index is None:
        use_frame_index = _has_complete_frame_index(frame)

    work = frame.copy()
    key_column = "_runtime_mode_frame_key"
    if use_frame_index:
        values = pd.to_numeric(work["frame_index"], errors="coerce")
        work[key_column] = values.to_numpy(dtype=float)
        key_kind = "frame_index"
    else:
        values = pd.to_numeric(work["time_s"], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        work = work.loc[finite].copy()
        work[key_column] = values.loc[finite].to_numpy(dtype=float)
        key_kind = "time_s"

    groups: list[tuple[tuple[str, int | float], pd.DataFrame]] = []
    for value, rows in work.groupby(key_column, sort=True):
        key = (
            ("frame_index", int(float(value)))
            if key_kind == "frame_index"
            else ("time_s", round(float(value), 9))
        )
        groups.append((key, rows.drop(columns=key_column).copy()))
    return groups


def _row_key(
    row: pd.Series | object,
    *,
    use_frame_index: bool | None = None,
) -> tuple[str, int | float]:
    """Return a tagged frame key using the requested grouping mode."""

    if isinstance(row, pd.Series):
        frame_index = row.get("frame_index", np.nan)
        time_s = row["time_s"]
    else:
        frame_index = getattr(row, "frame_index", np.nan)
        time_s = getattr(row, "time_s")

    if use_frame_index is None:
        try:
            use_frame_index = bool(np.isfinite(float(frame_index)))
        except (TypeError, ValueError):
            use_frame_index = False
    if use_frame_index:
        return ("frame_index", int(float(frame_index)))
    return ("time_s", round(float(time_s), 9))


def _backward_repair_one_sequence(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    max_gap_s: float,
    max_repair_distance_m: float,
) -> pd.DataFrame:
    """Repair one sequence using a frame key valid for both input tables."""

    selected = selected.sort_values("time_s", kind="mergesort").reset_index(drop=True)
    repaired = [row.copy() for _, row in selected.iterrows()]
    use_frame_index = _has_complete_frame_index(selected) and (
        _has_complete_frame_index(candidates)
    )
    candidate_groups = _frame_groups(
        candidates,
        use_frame_index=use_frame_index,
    )
    selected_keys = {
        _row_key(row, use_frame_index=use_frame_index)
        for _, row in selected.iterrows()
    }
    for left, right in zip(
        selected.iloc[:-1].itertuples(index=False),
        selected.iloc[1:].itertuples(index=False),
    ):
        left_time = float(left.time_s)
        right_time = float(right.time_s)
        gap_s = right_time - left_time
        if gap_s <= 0.0 or gap_s > float(max_gap_s):
            continue
        left_pos = np.array(
            [left.east_m, left.north_m, left.up_m],
            dtype=float,
        )
        right_pos = np.array(
            [right.east_m, right.north_m, right.up_m],
            dtype=float,
        )
        if not np.isfinite(left_pos).all() or not np.isfinite(right_pos).all():
            continue
        for key, frame in candidate_groups:
            if key in selected_keys:
                continue
            time_values = pd.to_numeric(frame["time_s"], errors="coerce")
            time_s = float(time_values.median())
            if not left_time < time_s < right_time:
                continue
            alpha = (time_s - left_time) / gap_s
            target = (1.0 - alpha) * left_pos + alpha * right_pos
            positions = (
                frame.loc[:, _LEGACY.PositionColumns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
            finite = np.isfinite(positions).all(axis=1)
            if not finite.any():
                continue
            distances = np.full(len(frame), np.inf, dtype=float)
            distances[finite] = np.linalg.norm(
                positions[finite] - target.reshape(1, 3),
                axis=1,
            )
            best_idx = int(np.argmin(distances))
            if float(distances[best_idx]) <= float(max_repair_distance_m):
                repaired_row = frame.iloc[best_idx].copy()
                repaired_row["association_mode"] = "backward-repair"
                repaired_row["association_score"] = float(distances[best_idx])
                repaired_row["association_repaired"] = True
                repaired.append(repaired_row)
                selected_keys.add(key)
    return (
        pd.DataFrame(repaired)
        .sort_values("time_s", kind="mergesort")
        .reset_index(drop=True)
    )


def backward_repair_associations(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    max_gap_s: float = 10.0,
    max_repair_distance_m: float = 200.0,
) -> pd.DataFrame:
    """Repair gaps with sequence-local anchors and complete frame-key fallback."""

    if selected.empty or candidates.empty:
        return selected.copy()
    selected_rows = pd.DataFrame(selected).copy()
    candidate_rows = pd.DataFrame(candidates).copy()
    if "sequence_id" not in selected_rows.columns or "sequence_id" not in candidate_rows.columns:
        return _backward_repair_one_sequence(
            selected_rows,
            candidate_rows,
            max_gap_s=max_gap_s,
            max_repair_distance_m=max_repair_distance_m,
        )

    sequence_key_column = "_runtime_mode_sequence_key"
    selected_rows[sequence_key_column] = _LEGACY._sequence_keys(
        selected_rows["sequence_id"]
    )
    candidate_rows[sequence_key_column] = _LEGACY._sequence_keys(
        candidate_rows["sequence_id"]
    )
    repaired_parts: list[pd.DataFrame] = []
    for sequence_key in pd.unique(selected_rows[sequence_key_column]):
        selected_mask = _LEGACY._sequence_mask(
            selected_rows[sequence_key_column],
            sequence_key,
        )
        candidate_mask = _LEGACY._sequence_mask(
            candidate_rows[sequence_key_column],
            sequence_key,
        )
        sequence_selected = selected_rows.loc[selected_mask].drop(
            columns=sequence_key_column
        )
        sequence_candidates = candidate_rows.loc[candidate_mask].drop(
            columns=sequence_key_column
        )
        if sequence_candidates.empty:
            repaired_parts.append(sequence_selected)
            continue
        repaired_parts.append(
            _backward_repair_one_sequence(
                sequence_selected,
                sequence_candidates,
                max_gap_s=max_gap_s,
                max_repair_distance_m=max_repair_distance_m,
            )
        )
    return (
        pd.concat(repaired_parts, ignore_index=True, sort=False)
        .sort_values(["sequence_id", "time_s"], kind="mergesort")
        .reset_index(drop=True)
    )


_LEGACY._has_complete_frame_index = _has_complete_frame_index
_LEGACY._frame_groups = _frame_groups
_LEGACY._row_key = _row_key
_LEGACY._backward_repair_one_sequence = _backward_repair_one_sequence
_LEGACY.backward_repair_associations = backward_repair_associations

for _name in dir(_LEGACY):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_LEGACY, _name)
globals()["_has_complete_frame_index"] = _has_complete_frame_index
globals()["_frame_groups"] = _frame_groups
globals()["_row_key"] = _row_key
globals()["_backward_repair_one_sequence"] = _backward_repair_one_sequence
globals()["backward_repair_associations"] = backward_repair_associations

__doc__ = _LEGACY.__doc__
__all__ = [
    name
    for name in dir(_LEGACY)
    if not (name.startswith("__") and name.endswith("__"))
]
