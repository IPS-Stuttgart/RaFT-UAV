"""Keep Track 5 jerk diagnostics aligned and safely aggregate persisted flags."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_preserves_jerk_window_support"
_MANIFEST_PATCH_MARKER = "_raft_uav_normalizes_jerk_manifest_flags"
_INVALID_BOOLEAN = object()
_TRUE_BOOLEAN_TOKENS = frozenset({"true", "t", "yes", "y", "on"})
_FALSE_BOOLEAN_TOKENS = frozenset({"false", "f", "no", "n", "off"})
_MISSING_BOOLEAN_TOKENS = frozenset({"", "nan", "none", "null", "<na>", "nat"})


def _install_window_support(track5_jerk_limit: Any) -> None:
    """Install support-aware row attribution for Track 5 jerk windows."""

    original: Callable[..., np.ndarray] = track5_jerk_limit._row_jerk_proxy
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def aligned(times: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        count = len(times)
        row_jerk = np.full(count, np.nan, dtype=float)
        d3 = track5_jerk_limit._third_derivative_matrix(times)
        if d3.size == 0:
            return row_jerk

        jerk_windows = d3 @ np.asarray(xyz, dtype=float)
        norms = np.linalg.norm(jerk_windows, axis=1)
        for coefficients, norm in zip(d3, norms, strict=True):
            for row_index in np.flatnonzero(coefficients):
                current = row_jerk[row_index]
                if np.isnan(current) or norm > current:
                    row_jerk[row_index] = float(norm)
        return row_jerk

    setattr(aligned, _PATCH_MARKER, True)
    track5_jerk_limit._row_jerk_proxy = aligned
    implementation: Any = getattr(track5_jerk_limit, "_IMPL", None)
    if implementation is not None and hasattr(implementation, "_row_jerk_proxy"):
        implementation._row_jerk_proxy = aligned


def _boolean_number(value: float) -> bool | object:
    """Parse one finite numeric Boolean representation."""

    if np.isnan(value):
        return False
    if not np.isfinite(value):
        return _INVALID_BOOLEAN
    if value == 0.0:
        return False
    if value == 1.0:
        return True
    return _INVALID_BOOLEAN


def _applied_flag_value(value: object) -> bool | object:
    """Parse one persisted ``jerk_limit_applied`` cell without truthiness casts."""

    if value is None or value is pd.NA or np.ma.is_masked(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in _TRUE_BOOLEAN_TOKENS:
            return True
        if text in _FALSE_BOOLEAN_TOKENS or text in _MISSING_BOOLEAN_TOKENS:
            return False
        try:
            return _boolean_number(float(text))
        except (TypeError, ValueError, OverflowError):
            return _INVALID_BOOLEAN

    try:
        array = np.asanyarray(value)
    except (TypeError, ValueError):
        return _INVALID_BOOLEAN
    if array.ndim != 0 or np.iscomplexobj(array):
        return _INVALID_BOOLEAN
    if np.ma.isMaskedArray(array) and bool(np.ma.getmaskarray(array).any()):
        return False

    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    try:
        if bool(pd.isna(scalar)):
            return False
    except (TypeError, ValueError):
        return _INVALID_BOOLEAN
    try:
        return _boolean_number(float(scalar))
    except (TypeError, ValueError, OverflowError):
        return _INVALID_BOOLEAN


def _normalized_applied_flags(values: pd.Series) -> pd.Series:
    """Return strict Boolean flags while preserving the diagnostics row index."""

    series = pd.Series(values, copy=False)
    normalized: list[bool] = []
    invalid: list[tuple[object, object]] = []
    for index, value in series.items():
        parsed = _applied_flag_value(value)
        if parsed is _INVALID_BOOLEAN:
            invalid.append((index, value))
            normalized.append(False)
        else:
            normalized.append(bool(parsed))

    if invalid:
        preview = ", ".join(
            f"index {index!r}: {value!r}" for index, value in invalid[:5]
        )
        suffix = "" if len(invalid) <= 5 else f"; plus {len(invalid) - 5} more"
        raise ValueError(
            "jerk_limit_applied contains invalid Boolean values: "
            f"{preview}{suffix}"
        )
    return pd.Series(
        normalized,
        index=series.index,
        name=series.name,
        dtype=bool,
    )


def _install_manifest_flag_guard(track5_jerk_limit: Any) -> None:
    """Normalize persisted applied flags before writing manifest aggregates."""

    original: Callable[..., dict[str, Any]] = (
        track5_jerk_limit.write_track5_jerk_limit_outputs
    )
    if getattr(original, _MANIFEST_PATCH_MARKER, False):
        return

    @wraps(original)
    def guarded(
        *,
        repaired: pd.DataFrame,
        diagnostics: pd.DataFrame,
        output_dir: Any,
        input_submission_path: Any,
        template: pd.DataFrame | None = None,
        manifest: dict[str, Any] | None = None,
        require_leaderboard_ready: bool = False,
    ) -> dict[str, Any]:
        normalized_diagnostics = pd.DataFrame(diagnostics).copy()
        if "jerk_limit_applied" in normalized_diagnostics.columns:
            normalized_diagnostics["jerk_limit_applied"] = _normalized_applied_flags(
                normalized_diagnostics["jerk_limit_applied"]
            )
        return original(
            repaired=repaired,
            diagnostics=normalized_diagnostics,
            output_dir=output_dir,
            input_submission_path=input_submission_path,
            template=template,
            manifest=manifest,
            require_leaderboard_ready=require_leaderboard_ready,
        )

    setattr(guarded, _MANIFEST_PATCH_MARKER, True)
    track5_jerk_limit.write_track5_jerk_limit_outputs = guarded
    implementation: Any = getattr(track5_jerk_limit, "_IMPL", None)
    if implementation is not None and hasattr(
        implementation,
        "write_track5_jerk_limit_outputs",
    ):
        implementation.write_track5_jerk_limit_outputs = guarded


def install() -> None:
    """Install Track 5 jerk-window and persisted-manifest diagnostics guards."""

    from raft_uav.mmuad import track5_jerk_limit

    _install_window_support(track5_jerk_limit)
    _install_manifest_flag_guard(track5_jerk_limit)
