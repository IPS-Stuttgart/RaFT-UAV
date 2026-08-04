"""Reject malformed explicit multi-object tracker configurations.

The startup hook also canonicalizes equal-confidence detections before the MOT
tracker assigns persistent output identities. This makes exact optimal-assignment
ties independent of DataFrame row order.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_validates_explicit_mot_config"
_ORDER_HELPER_PREFIX = "_raft_uav_mot_order_"


def _finite_real_order_values(values: pd.Series) -> pd.Series:
    """Return finite real sort keys without changing the candidate payload."""

    parsed: list[float] = []
    for value in values.tolist():
        try:
            if np.ma.is_masked(value):
                raise TypeError
            array = np.asarray(value)
            if array.ndim != 0 or np.iscomplexobj(array):
                raise TypeError
            number = float(array.item())
        except (TypeError, ValueError, OverflowError):
            number = float("inf")
        parsed.append(number if np.isfinite(number) else float("inf"))
    return pd.Series(parsed, index=values.index, dtype=float)


def _text_order_values(values: pd.Series) -> pd.Series:
    """Return stable text fallback keys for optional candidate metadata."""

    return values.where(values.notna(), "").astype(str).str.strip()


def _ordered_candidate_frame(candidates: Any, mot: Any) -> Any:
    """Canonicalize within-frame detection order for deterministic MOT ties."""

    rows = getattr(candidates, "rows", None)
    if not isinstance(rows, pd.DataFrame) or rows.empty:
        return candidates

    ordered = rows.copy().reset_index(drop=True)
    sort_keys = pd.DataFrame(index=ordered.index)
    sort_columns: list[str] = []
    ascending: list[bool] = []

    for name in ("sequence_id",):
        key = f"{_ORDER_HELPER_PREFIX}{name}"
        values = ordered[name] if name in ordered.columns else pd.Series("", index=ordered.index)
        sort_keys[key] = _text_order_values(values)
        sort_columns.append(key)
        ascending.append(True)

    for name in ("time_s",):
        key = f"{_ORDER_HELPER_PREFIX}{name}"
        values = (
            ordered[name]
            if name in ordered.columns
            else pd.Series(np.nan, index=ordered.index)
        )
        sort_keys[key] = _finite_real_order_values(values)
        sort_columns.append(key)
        ascending.append(True)

    confidence_key = f"{_ORDER_HELPER_PREFIX}confidence"
    sort_keys[confidence_key] = mot._mot_confidence_values(ordered)
    sort_columns.append(confidence_key)
    ascending.append(False)

    for name in ("x_m", "y_m", "z_m"):
        key = f"{_ORDER_HELPER_PREFIX}{name}"
        values = (
            ordered[name]
            if name in ordered.columns
            else pd.Series(np.nan, index=ordered.index)
        )
        sort_keys[key] = _finite_real_order_values(values)
        sort_columns.append(key)
        ascending.append(True)

    for name in ("source", "track_id", "class_name"):
        key = f"{_ORDER_HELPER_PREFIX}{name}"
        values = ordered[name] if name in ordered.columns else pd.Series("", index=ordered.index)
        sort_keys[key] = _text_order_values(values)
        sort_columns.append(key)
        ascending.append(True)

    row_order = sort_keys.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).index.to_numpy()
    return mot.CandidateFrame(ordered.iloc[row_order].reset_index(drop=True))


def install() -> None:
    """Install strict config validation and deterministic candidate ordering."""

    import raft_uav.mmuad as mmuad
    from raft_uav.mmuad import mot

    original: Callable[..., Any] = mot.run_mmuad_multi_object_tracker
    if getattr(original, _PATCH_MARKER, False):
        setattr(mmuad, "run_mmuad_multi_object_tracker", original)
        return

    @wraps(original)
    def validated(
        candidates: Any,
        truth: Any = None,
        *,
        config: Any = None,
    ) -> Any:
        if config is not None and not isinstance(config, mot.MultiObjectTrackerConfig):
            raise TypeError("config must be a MultiObjectTrackerConfig or None")
        ordered_candidates = _ordered_candidate_frame(candidates, mot)
        return original(ordered_candidates, truth, config=config)

    setattr(validated, _PATCH_MARKER, True)
    setattr(mot, "run_mmuad_multi_object_tracker", validated)
    setattr(mmuad, "run_mmuad_multi_object_tracker", validated)
