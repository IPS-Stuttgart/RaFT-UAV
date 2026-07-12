"""Compatibility wrapper rejecting non-finite spatial group top-K settings.

The maintained implementation lives in the sibling
``candidate_mixture_group_spatial_topk.py`` module. This package preserves the
public import path while ensuring that NaN and infinite diversity parameters are
rejected before they can produce non-finite group-selection utilities.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_group_spatial_topk.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_spatial_topk_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load spatial group top-K implementation from "
        f"{_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_VALIDATE_SELECTION_CONFIG = _IMPL._validate_selection_config
_FINITE_SELECTION_FIELDS = (
    "diversity_weight",
    "diversity_scale_m",
    "diversity_cap_m",
)


def _validate_selection_config(config: Any) -> None:
    """Reject non-finite diversity settings before applying range checks."""

    for field_name in _FINITE_SELECTION_FIELDS:
        try:
            value = float(getattr(config, field_name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
    _ORIGINAL_VALIDATE_SELECTION_CONFIG(config)


_IMPL._validate_selection_config = _validate_selection_config

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_selection_config"] = _validate_selection_config
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
