"""Compatibility wrapper for Track 5 trajectory regularization.

The maintained implementation lives in the sibling
``track5_trajectory_regularizer.py`` module. This package preserves the public
import path while ensuring that reported robust weights correspond to the final
smoothed trajectory rather than the state from the preceding IRLS iteration.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_trajectory_regularizer.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_trajectory_regularizer_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        f"cannot load Track 5 trajectory-regularizer implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_ROBUST_SMOOTH_SEQUENCE = _IMPL._robust_smooth_sequence


def _robust_smooth_sequence(
    times: np.ndarray,
    observed: np.ndarray,
    *,
    finite: np.ndarray,
    smoothness_weight: float,
    huber_delta_m: float,
    iterations: int,
    observation_sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return final-state residuals and Huber weights from the same iterate."""

    state, residual, _ = _ORIGINAL_ROBUST_SMOOTH_SEQUENCE(
        times,
        observed,
        finite=finite,
        smoothness_weight=smoothness_weight,
        huber_delta_m=huber_delta_m,
        iterations=iterations,
        observation_sigma_m=observation_sigma_m,
    )
    final_weights = np.zeros(len(residual), dtype=float)
    finite_mask = np.asarray(finite, dtype=bool)
    final_residual = np.asarray(residual, dtype=float)[finite_mask]
    final_weights[finite_mask] = np.minimum(
        1.0,
        float(huber_delta_m) / np.maximum(final_residual, 1.0e-12),
    )
    return state, residual, final_weights


_IMPL._robust_smooth_sequence = _robust_smooth_sequence

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_ROBUST_SMOOTH_SEQUENCE"] = _ORIGINAL_ROBUST_SMOOTH_SEQUENCE
globals()["_robust_smooth_sequence"] = _robust_smooth_sequence

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
