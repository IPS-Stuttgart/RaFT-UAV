"""Validate uncertainty inputs and covariance fallbacks at package startup."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

_APPLY_PATCH_MARKER = "_raft_uav_validates_empty_uncertainty_apply"
_COVARIANCE_PATCH_MARKER = "_raft_uav_validates_uncertainty_covariance_output"


def install() -> None:
    """Install uncertainty input and covariance validation guards."""

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

    original_covariance: Callable[..., Any] = uncertainty_module.covariance_from_row
    if not getattr(original_covariance, _COVARIANCE_PATCH_MARKER, False):

        @wraps(original_covariance)
        def validated_covariance_from_row(
            row,
            dim,
            fallback,
            *,
            prefixes=("association_cov", "cov"),
        ):
            covariance = original_covariance(row, dim, fallback, prefixes=prefixes)
            checked = uncertainty_module._finite_positive_definite_covariance(covariance)
            if checked is None or checked.shape != (dim, dim):
                raise ValueError(
                    f"resolved covariance must be a finite {dim}x{dim} "
                    "positive-definite covariance matrix"
                )
            return checked

        setattr(validated_covariance_from_row, _COVARIANCE_PATCH_MARKER, True)
        uncertainty_module.covariance_from_row = validated_covariance_from_row
