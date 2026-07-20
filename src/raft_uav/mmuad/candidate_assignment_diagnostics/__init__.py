"""Compatibility fixes for candidate-assignment diagnostic parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int as _safe_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_assignment_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_assignment_diagnostics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load candidate-assignment diagnostics implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _assignment_weights(group: pd.DataFrame) -> np.ndarray:
    """Return finite normalized assignment weights.

    Malformed, negative, NaN, and infinite weights carry no usable probability
    mass. If no positive finite mass remains, fall back to a uniform distribution.
    """

    if "mixture_final_weight" in group.columns:
        weights = pd.to_numeric(
            group["mixture_final_weight"], errors="coerce"
        ).to_numpy(dtype=float)
    elif "mixture_dominant" in group.columns:
        weights = np.asarray(
            [
                _IMPL._parse_mixture_dominant_flag(value)
                for value in group["mixture_dominant"]
            ],
            dtype=float,
        )
    else:
        weights = np.ones(len(group), dtype=float)

    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= 1.0e-12:
        return np.ones(len(group), dtype=float) / max(float(len(group)), 1.0)
    return weights / total


# Candidate ranks are integer identifiers. The legacy float round-trip silently
# truncated fractional values and lost precision for integers above 2**53.
_IMPL._safe_int = _safe_int
_IMPL._assignment_weights = _assignment_weights

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_safe_int"] = _safe_int
globals()["_assignment_weights"] = _assignment_weights

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
