"""Compatibility package validating group-top-K selection controls.

The maintained implementation lives in the sibling
``candidate_mixture_group_topk.py`` module. This package preserves the public
import path while rejecting malformed integer controls before ``int(...)`` can
silently truncate them or leak implementation-specific exceptions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_mixture_group_topk.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_topk_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate mixture group top-K from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

HypothesisGroupTopKConfig = _IMPL.HypothesisGroupTopKConfig
GROUP_SCORE_MODES = _IMPL.GROUP_SCORE_MODES


def _finite_scalar(value: Any, *, name: str) -> float:
    """Return one finite non-Boolean scalar with a stable validation error."""

    message = f"{name} must be a finite scalar"
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def _non_negative_integer(value: Any, *, name: str) -> int:
    """Return an integer-valued scalar greater than or equal to zero."""

    message = f"{name} must be a non-negative integer"
    try:
        number = _finite_scalar(value, name=name)
    except ValueError as exc:
        raise ValueError(message) from exc
    if not number.is_integer() or number < 0.0:
        raise ValueError(message)
    return int(number)


def _validate_selection_config(config: HypothesisGroupTopKConfig) -> None:
    """Validate group selection controls without lossy integer coercion."""

    _non_negative_integer(config.group_top_k, name="group_top_k")
    _non_negative_integer(
        config.max_siblings_per_group,
        name="max_siblings_per_group",
    )
    if config.group_score_mode not in GROUP_SCORE_MODES:
        raise ValueError(
            f"group_score_mode must be one of {GROUP_SCORE_MODES}, "
            f"got {config.group_score_mode!r}"
        )


_IMPL._finite_scalar = _finite_scalar
_IMPL._non_negative_integer = _non_negative_integer
_IMPL._validate_selection_config = _validate_selection_config

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_scalar"] = _finite_scalar
globals()["_non_negative_integer"] = _non_negative_integer
globals()["_validate_selection_config"] = _validate_selection_config

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
