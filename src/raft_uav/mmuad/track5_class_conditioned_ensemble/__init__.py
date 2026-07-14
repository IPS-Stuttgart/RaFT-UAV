"""Compatibility wrapper for robust Track 5 class-ensemble handling.

The maintained implementation lives in the sibling
``track5_class_conditioned_ensemble.py`` module. This package preserves the
public import path while making template-header lookup robust and validating
class-conditioned weight configurations before they reach the ensemble.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_class_conditioned_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_class_conditioned_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load Track 5 class-conditioned ensemble implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _first_present(rows: Any, names: tuple[str, ...]) -> Any | None:
    """Return the original column whose stripped, case-folded name matches."""

    by_normalized_name = {
        str(column).strip().casefold(): column for column in rows.columns
    }
    for name in names:
        if name in rows.columns:
            return name
        found = by_normalized_name.get(str(name).strip().casefold())
        if found is not None:
            return found
    return None


def _normalized_weight_map(raw: Any, inputs: tuple[Any, ...]) -> dict[str, float]:
    """Validate and normalize a class-conditioned estimate-weight mapping."""

    if not isinstance(raw, dict):
        raise ValueError("weight map must be an object")

    labels = tuple(str(item.label) for item in inputs)
    if not raw:
        raw = {label: 1.0 for label in labels}

    unknown = sorted((key for key in raw if key not in labels), key=str)
    if unknown:
        raise ValueError(f"weight map has unknown estimate labels: {unknown}")

    weights: dict[str, float] = {}
    for label in labels:
        candidate = raw.get(label, 0.0)
        try:
            value = float(candidate)
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise ValueError(
                f"estimate weight must be finite and non-negative for {label}: "
                f"{candidate!r}"
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"estimate weight must be finite and non-negative for {label}: {value}"
            )
        weights[label] = value

    scale = max(weights.values(), default=0.0)
    if scale <= 0.0:
        raise ValueError("weight map must contain at least one positive weight")

    scaled = {label: value / scale for label, value in weights.items()}
    total = sum(scaled.values())
    return {label: value / total for label, value in scaled.items()}


_IMPL._first_present = _first_present
_IMPL._normalized_weight_map = _normalized_weight_map

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep patched private helpers visible to tests and exploratory callers.
globals()["_first_present"] = _first_present
globals()["_normalized_weight_map"] = _normalized_weight_map
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
