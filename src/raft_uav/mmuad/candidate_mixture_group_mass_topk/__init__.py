"""Compatibility fix for posterior-mass MMUAD group selection.

The maintained implementation lives in the sibling
``candidate_mixture_group_mass_topk.py`` module. This package preserves the
public import path while rejecting non-finite posterior temperatures before
they can silently collapse the posterior to the softmax fallback.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_mixture_group_mass_topk.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_mass_topk_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load posterior-mass group top-K implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_VALIDATE_SELECTION_CONFIG = _IMPL._validate_selection_config


def _validate_selection_config(config: Any) -> None:
    """Reject NaN and infinite posterior temperatures before range checks."""

    try:
        posterior_temperature = float(config.posterior_temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("posterior_temperature must be numeric") from exc
    if not np.isfinite(posterior_temperature):
        raise ValueError("posterior_temperature must be finite")
    _ORIGINAL_VALIDATE_SELECTION_CONFIG(config)


_FIXED_VALIDATE_SELECTION_CONFIG = _validate_selection_config
_IMPL._validate_selection_config = _FIXED_VALIDATE_SELECTION_CONFIG

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_selection_config"] = _FIXED_VALIDATE_SELECTION_CONFIG
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
