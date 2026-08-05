"""Compatibility wrapper preserving distinct source-calibration path fractions.

The maintained implementation lives in the sibling
``source_calibration_path_ensemble.py`` module. This package preserves the
public import path while treating only exact ``0`` and ``1`` fractions as
calibration-path endpoints, assigning round-trip-safe branch tokens to every
distinct normalized fraction, and rejecting lossy pseudo-numeric fractions.
"""

from __future__ import annotations

from collections.abc import Iterable
import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame

_IMPL_PATH = Path(__file__).resolve().parent.parent / "source_calibration_path_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._source_calibration_path_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load source-calibration path ensemble implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _contains_fraction(values: Sequence[float], target: float) -> bool:
    """Return whether an exact normalized endpoint is present."""

    return any(float(value) == float(target) for value in values)


def _fraction_token(value: float) -> str:
    """Return a round-trip-safe branch token for one normalized float."""

    return f"{float(value):.17g}".replace("-", "m").replace(".", "p")


def _coerce_fraction(value: object) -> float:
    """Return one finite real fraction without lossy scalar coercion."""

    error = "calibration fractions must be finite real scalars"
    current = value
    seen_arrays: set[int] = set()
    while isinstance(current, np.ndarray):
        if current.ndim != 0 or np.ma.is_masked(current) or np.iscomplexobj(current):
            raise ValueError(error)
        marker = id(current)
        if marker in seen_arrays:
            raise ValueError(error)
        seen_arrays.add(marker)
        current = current.item()
    if (
        np.ma.is_masked(current)
        or isinstance(current, (bool, np.bool_))
        or np.iscomplexobj(current)
    ):
        raise ValueError(error)
    try:
        number = float(current)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number):
        raise ValueError(error)
    return number


def _normalize_fractions(values: Iterable[object]) -> tuple[float, ...]:
    """Normalize a non-empty iterable of finite real fractions in ``[0, 1]``."""

    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(
            "calibration fractions must be an iterable of finite real scalars"
        )
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(
            "calibration fractions must be an iterable of finite real scalars"
        ) from exc
    if not raw_values:
        raise ValueError("at least one calibration fraction is required")
    fractions = tuple(_coerce_fraction(value) for value in raw_values)
    for value in fractions:
        if value < 0.0 or value > 1.0:
            raise ValueError("calibration fractions must lie in [0, 1]")
    return tuple(sorted(set(fractions)))


def _annotate_fraction(
    rows: pd.DataFrame,
    *,
    fraction: float,
    branch: str | None,
) -> pd.DataFrame:
    """Annotate one exact path fraction without collapsing near-zero values."""

    out = rows.copy()
    if branch is not None:
        branch = _IMPL._branch_label(branch)
        out["candidate_branch"] = branch
        out["mmuad_source_calibration_branch"] = branch
    out[_IMPL.CALIBRATION_FRACTION_COLUMN] = float(fraction)
    out[_IMPL.INTERPOLATED_COLUMN] = bool(0.0 < float(fraction) < 1.0)
    calibrated = bool(float(fraction) > 0.0)
    out["mmuad_candidate_branch_is_calibrated"] = calibrated
    if not calibrated:
        out["mmuad_source_calibration_applied"] = False

    original_xyz = out[list(_IMPL.ORIGINAL_XYZ_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    current_xyz = out[["x_m", "y_m", "z_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    delta = current_xyz.to_numpy(float) - original_xyz.to_numpy(float)
    out["mmuad_calibration_dx_m"] = delta[:, 0]
    out["mmuad_calibration_dy_m"] = delta[:, 1]
    out["mmuad_calibration_dz_m"] = delta[:, 2]
    out["mmuad_calibration_displacement_m"] = np.linalg.norm(delta, axis=1)

    alpha = pd.to_numeric(
        out.get(
            "mmuad_source_calibration_alpha",
            pd.Series(np.nan, index=out.index),
        ),
        errors="coerce",
    )
    out[_IMPL.EFFECTIVE_ALPHA_COLUMN] = alpha.to_numpy(float) * float(fraction)
    if fraction == 0.0:
        out[_IMPL.EFFECTIVE_ALPHA_COLUMN] = 0.0
    return out


def build_source_calibration_path_ensemble(
    candidates: CandidateFrame | pd.DataFrame,
    calibration_payload: dict[str, Any],
    *,
    fractions: Sequence[float] = _IMPL.DEFAULT_CALIBRATION_FRACTIONS,
    mode: str | None = None,
    raw_branch: str = "raw",
    calibrated_branch: str | None = None,
    intermediate_branch_prefix: str | None = None,
    keep_unapplied_calibrated: bool = False,
    branch_track_ids: bool = True,
) -> CandidateFrame:
    """Return every requested finite path fraction as a distinct hypothesis."""

    normalized_fractions = _normalize_fractions(fractions)
    union = _IMPL.build_source_calibration_branch_union(
        candidates,
        calibration_payload,
        mode=mode,
        raw_branch=raw_branch,
        calibrated_branch=calibrated_branch,
        keep_unapplied_calibrated=keep_unapplied_calibrated,
        branch_track_ids=False,
    ).rows
    if union.empty:
        return CandidateFrame(_IMPL._empty_ensemble_rows(union))

    calibrated_mask = _IMPL._boolean_series(
        union.get("mmuad_candidate_branch_is_calibrated", False),
        union.index,
    )
    raw_rows = union.loc[~calibrated_mask].copy()
    calibrated_rows = union.loc[calibrated_mask].copy()
    calibrated_label = _IMPL._calibrated_label(
        calibrated_rows,
        calibration_payload,
        mode,
    )
    prefix = _IMPL._branch_label(
        intermediate_branch_prefix or f"{calibrated_label}_path"
    )

    parts: list[pd.DataFrame] = []
    if _contains_fraction(normalized_fractions, 0.0):
        parts.append(_annotate_fraction(raw_rows, fraction=0.0, branch=None))
    for fraction in normalized_fractions:
        if fraction == 0.0 or fraction == 1.0:
            continue
        branch = f"{prefix}_f{_fraction_token(fraction)}"
        parts.append(
            _IMPL._interpolate_calibrated_rows(
                calibrated_rows,
                fraction=float(fraction),
                branch=branch,
            )
        )
    if _contains_fraction(normalized_fractions, 1.0):
        parts.append(_annotate_fraction(calibrated_rows, fraction=1.0, branch=None))

    if not parts:
        return CandidateFrame(_IMPL._empty_ensemble_rows(union))
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.loc[
        np.isfinite(out[["x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    ].copy()
    if branch_track_ids:
        out["track_id"] = [
            _IMPL._qualified_track_id(original, branch, origin)
            for original, branch, origin in zip(
                out[_IMPL.ORIGINAL_TRACK_ID_COLUMN],
                out["candidate_branch"],
                out[_IMPL.ORIGIN_ROW_COLUMN],
                strict=False,
            )
        ]
    return CandidateFrame(_IMPL.normalize_candidate_columns(out))


_IMPL._contains_fraction = _contains_fraction
_IMPL._fraction_token = _fraction_token
_IMPL._coerce_fraction = _coerce_fraction
_IMPL._normalize_fractions = _normalize_fractions
_IMPL._annotate_fraction = _annotate_fraction
_IMPL.build_source_calibration_path_ensemble = build_source_calibration_path_ensemble

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_contains_fraction"] = _contains_fraction
globals()["_fraction_token"] = _fraction_token
globals()["_coerce_fraction"] = _coerce_fraction
globals()["_normalize_fractions"] = _normalize_fractions
globals()["_annotate_fraction"] = _annotate_fraction
globals()["build_source_calibration_path_ensemble"] = (
    build_source_calibration_path_ensemble
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
