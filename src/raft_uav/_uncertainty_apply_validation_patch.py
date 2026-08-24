"""Validate uncertainty application and covariance fallback inputs."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_APPLY_PATCH_MARKER = "_raft_uav_validates_empty_uncertainty_apply"
_COVARIANCE_PATCH_MARKER = "_raft_uav_validates_uncertainty_fallback_covariance"


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


def install() -> None:
    """Install uncertainty source-head and covariance-fallback validation."""

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
