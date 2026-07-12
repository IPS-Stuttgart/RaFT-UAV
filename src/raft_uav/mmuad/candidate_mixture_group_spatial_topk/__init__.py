"""Compatibility fixes for spatially diverse MMUAD group top-K selection.

The maintained implementation lives in the sibling
``candidate_mixture_group_spatial_topk.py`` module.  This package preserves the
public import path while aligning candidate-uncertainty handling with the core
mixture-MAP implementation: non-finite, zero, and negative predicted sigmas are
missing values and must fall back to ``default_sigma_m`` rather than being
clipped to the minimum uncertainty.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_mixture_group_spatial_topk.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_group_spatial_topk_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load spatial group top-K implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _candidate_unary_utility(
    prepared: pd.DataFrame,
    *,
    mixture_config: Any,
) -> np.ndarray:
    """Return the state-independent score/uncertainty utility.

    A predicted sigma is usable only when it is finite and strictly positive.
    Invalid values represent missing uncertainty estimates and therefore use the
    configured default.  Clipping a negative value directly would otherwise turn
    it into ``sigma_min_m`` and make a malformed candidate look maximally certain.
    """

    normalized_score = pd.to_numeric(
        prepared["mixture_group_base_normalized_score"],
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    if mixture_config.sigma_column in prepared.columns:
        sigma = pd.to_numeric(
            prepared[mixture_config.sigma_column],
            errors="coerce",
        ).to_numpy(float)
    else:
        sigma = np.full(len(prepared), float(mixture_config.default_sigma_m))

    default_sigma = float(mixture_config.default_sigma_m)
    valid = np.isfinite(sigma) & (sigma > 0.0)
    sigma = np.where(valid, sigma, default_sigma)
    sigma = np.clip(
        sigma,
        float(mixture_config.sigma_min_m),
        float(mixture_config.sigma_max_m),
    )
    temperature = max(float(mixture_config.temperature), 1.0e-12)
    return (
        float(mixture_config.score_weight) * normalized_score / temperature
        - float(mixture_config.sigma_log_weight) * np.log(sigma)
    )


# Preserve the correction before exporting the legacy implementation.  The
# export includes a helper with the same name, so reading the global name after
# ``globals().update(...)`` would otherwise reinstall the buggy legacy helper.
_FIXED_CANDIDATE_UNARY_UTILITY = _candidate_unary_utility

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
_IMPL._candidate_unary_utility = _FIXED_CANDIDATE_UNARY_UTILITY
globals()["_candidate_unary_utility"] = _FIXED_CANDIDATE_UNARY_UTILITY

__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
