"""Validate uncertainty source heads before empty-input fast paths."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

_PATCH_MARKER = "_raft_uav_validates_empty_uncertainty_apply"


def install() -> None:
    """Install source-head validation ahead of uncertainty application fast paths."""

    from raft_uav import uncertainty as uncertainty_module

    model_class = uncertainty_module.HeteroscedasticUncertaintyModel
    original: Callable[..., Any] = model_class.apply
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def validated_apply(self, frame, *, source):
        self._heads(source)
        return original(self, frame, source=source)

    setattr(validated_apply, _PATCH_MARKER, True)
    model_class.apply = validated_apply
