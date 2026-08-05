"""Validate stress configuration provenance labels."""

from __future__ import annotations

import numpy as np

from . import perturbations as _IMPL

_ORIGINAL_POST_INIT = _IMPL.PerturbationConfig.__post_init__


def _normalized_name(value: object) -> str:
    """Return a trimmed scalar string suitable for artifact provenance."""

    seen_array_ids: set[int] = set()
    while isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError("name must be a non-blank string scalar")
        array_id = id(value)
        if array_id in seen_array_ids:
            raise ValueError("name must be a non-blank string scalar")
        seen_array_ids.add(array_id)
        value = value.item()

    if np.ma.is_masked(value) or not isinstance(value, (str, np.str_)):
        raise ValueError("name must be a non-blank string scalar")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("name must be a non-blank string scalar")
    return normalized


def _post_init(self: _IMPL.PerturbationConfig) -> None:
    object.__setattr__(self, "name", _normalized_name(self.name))
    _ORIGINAL_POST_INIT(self)


_IMPL.PerturbationConfig.__post_init__ = _post_init
