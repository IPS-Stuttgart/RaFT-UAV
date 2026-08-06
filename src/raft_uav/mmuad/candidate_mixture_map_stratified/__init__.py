"""Compatibility validation for stratified mixture top-K configuration.

The maintained implementation lives in the sibling
``candidate_mixture_map_stratified.py`` module. This package preserves the
public import path while requiring exact integer quota values before selection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from raft_uav.numeric import optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_mixture_map_stratified.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_map_stratified_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load stratified mixture implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _validated_integer(value: object, *, name: str, minimum: int) -> int:
    """Return an exact integer meeting ``minimum`` or raise a field error."""

    parsed = optional_int(value)
    if parsed is None or parsed < minimum:
        requirement = "positive integer" if minimum == 1 else "non-negative integer"
        raise ValueError(f"{name} must be a {requirement}")
    return parsed


def _validate_config(config: object) -> None:
    """Reject fractional, Boolean, non-finite, and non-scalar quota values."""

    _validated_integer(config.top_k, name="top_k", minimum=1)
    for name in (
        "min_per_branch",
        "min_per_source",
        "min_per_source_branch",
    ):
        _validated_integer(getattr(config, name), name=name, minimum=0)


_IMPL._validate_config = _validate_config

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_integer"] = _validated_integer
globals()["_validate_config"] = _validate_config

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
