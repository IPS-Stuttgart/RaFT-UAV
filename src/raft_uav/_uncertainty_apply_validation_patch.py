"""Validate uncertainty application, covariance fallbacks, and finite features."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_APPLY_PATCH_MARKER = "_raft_uav_validates_empty_uncertainty_apply"
_COVARIANCE_PATCH_MARKER = "_raft_uav_validates_uncertainty_fallback_covariance"
_FEATURE_PATCH_MARKER = "_raft_uav_stabilizes_uncertainty_velocity_norm"
_FLOAT_MAX = np.finfo(float).max


def _validated_fallback_covariance(value: object, *, dim: int) -> np.ndarray:
    """Return one finite symmetric positive-definite fallback covariance."""

    error = (
        f"fallback must be a finite symmetric positive-definite {dim}x{dim} "
        "covariance matrix"
    )
    if np.ma.is_masked(value):
        raise ValueError(error)
    try:
        array = np.asanyarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if np.ma.isMaskedArray(array) and bool(np.ma.getmaskarray(array).any()):
        raise ValueError(error)
    if np.iscomplexobj(array) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(error)
    if array.dtype == object:
        for item in array.flat:
            if np.ma.is_masked(item) or isinstance(
                item,
                (bool, np.bool_, complex, np.complexfloating),
            ):
                raise ValueError(error)
    try:
        covariance = np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if covariance.shape != (dim, dim) or not np.isfinite(covariance).all():
        raise ValueError(error)
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(error)
    covariance = 0.5 * (covariance + covariance.T)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(error) from exc
    if eigenvalues.size != dim or float(np.min(eigenvalues)) <= 0.0:
        raise ValueError(error)
    return covariance


def _stable_velocity_norm(
    east_mps: np.ndarray,
    north_mps: np.ndarray,
    down_mps: np.ndarray,
) -> np.ndarray:
    """Return finite overflow-stable velocity norms for finite components."""

    components = np.column_stack([east_mps, north_mps, down_mps])
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.hypot.reduce(components, axis=1)
    return np.nan_to_num(
        norms,
        nan=0.0,
        posinf=_FLOAT_MAX,
        neginf=_FLOAT_MAX,
    )


def install() -> None:
    """Install uncertainty source-head, covariance, and feature validation."""

    from raft_uav import uncertainty as uncertainty_module

    model_class = uncertainty_module.HeteroscedasticUncertaintyModel
    original_apply: Callable[..., Any] = model_class.apply
    if not getattr(original_apply, _APPLY_PATCH_MARKER, False):

        @wraps(original_apply)
        def validated_apply(self, frame, *, source):
            self._heads(source)
            return original_apply(self, frame, source=source)

        setattr(validated_apply, _APPLY_PATCH_MARKER, True)
        model_class.apply = validated_apply

    original_covariance_from_row: Callable[..., Any] = uncertainty_module.covariance_from_row
    if not getattr(
        original_covariance_from_row,
        _COVARIANCE_PATCH_MARKER,
        False,
    ):

        @wraps(original_covariance_from_row)
        def validated_covariance_from_row(
            row,
            dim,
            fallback,
            *,
            prefixes=("association_cov", "cov"),
        ):
            if dim not in (2, 3):
                return original_covariance_from_row(
                    row,
                    dim,
                    fallback,
                    prefixes=prefixes,
                )
            validated_fallback = _validated_fallback_covariance(fallback, dim=dim)
            return original_covariance_from_row(
                row,
                dim,
                validated_fallback,
                prefixes=prefixes,
            )

        setattr(
            validated_covariance_from_row,
            _COVARIANCE_PATCH_MARKER,
            True,
        )
        uncertainty_module.covariance_from_row = validated_covariance_from_row

    legacy = uncertainty_module._legacy
    original_feature_frame: Callable[..., Any] = legacy._feature_frame
    if not getattr(original_feature_frame, _FEATURE_PATCH_MARKER, False):

        @wraps(original_feature_frame)
        def stable_feature_frame(frame, source):
            if source != "radar":
                return original_feature_frame(frame, source)

            with np.errstate(over="ignore", invalid="ignore"):
                out = original_feature_frame(frame, source)
            east_mps = legacy._num(
                frame,
                ("velocity_east_mps", "v_east_mps"),
                0.0,
            )
            north_mps = legacy._num(
                frame,
                ("velocity_north_mps", "v_north_mps"),
                0.0,
            )
            down_mps = legacy._num(
                frame,
                ("velocity_down_mps", "v_down_mps"),
                0.0,
            )
            out = out.copy()
            out["velocity_norm"] = _stable_velocity_norm(
                east_mps,
                north_mps,
                down_mps,
            )
            return out

        setattr(stable_feature_frame, _FEATURE_PATCH_MARKER, True)
        legacy._feature_frame = stable_feature_frame
        uncertainty_module._feature_frame = stable_feature_frame
