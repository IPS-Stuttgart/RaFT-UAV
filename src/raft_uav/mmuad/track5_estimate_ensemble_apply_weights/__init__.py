"""Compatibility package validating applied Track 5 ensemble configuration.

The maintained implementation lives in the sibling
``track5_estimate_ensemble_apply_weights.py`` module. This package preserves the
public import path while rejecting malformed top-level JSON payloads and
Boolean or non-scalar pseudo-numbers before they can be coerced into ensemble
weights, trim fractions, or nearest-time limits.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_estimate_ensemble_apply_weights.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_ensemble_apply_weights_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load applied ensemble-weight implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

EstimateInput = _IMPL.EstimateInput
_ORIGINAL_APPLY = _IMPL.apply_ensemble_weight_config
_ORIGINAL_WRITE = _IMPL.write_apply_weights_outputs


class _ApplyWeightsModule(ModuleType):
    """Module proxy that keeps runtime monkeypatches visible to legacy globals."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "_IMPL":
            return
        implementation = self.__dict__.get("_IMPL")
        if implementation is not None and hasattr(implementation, name):
            setattr(implementation, name, value)


def _validate_weight_value(value: Any, *, label: str) -> float:
    """Return one finite non-negative real scalar weight."""

    weight = optional_float(value)
    if weight is None or weight < 0.0:
        raise ValueError(f"weight for {label!r} must be finite and non-negative")
    return weight


def _select_trim_fraction(
    override: float | None,
    weight_config: dict[str, Any],
    *,
    default: float = 0.2,
) -> float:
    """Return a finite real scalar trim fraction in the supported interval."""

    raw_value = override if override is not None else weight_config.get("trim_fraction", default)
    if raw_value is None:
        raw_value = default
    selected = optional_float(raw_value)
    if selected is None or not 0.0 <= selected < 0.5:
        raise ValueError("trim_fraction must be finite and in [0, 0.5)")
    return selected


def _validate_max_nearest_time_delta_s(value: Any) -> float | None:
    """Return a finite non-negative real scalar time tolerance."""

    if value is None:
        return None
    delta = optional_float(value)
    if delta is None or delta < 0.0:
        raise ValueError("max_nearest_time_delta_s must be finite and non-negative")
    return delta


def load_ensemble_weight_config(path: Path) -> dict[str, Any]:
    """Load an ensemble-weight config after validating its JSON container."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("weight config must be a JSON object")
    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError("weight config must contain a non-empty 'weights' object")
    parsed_weights = _IMPL._normalize_weight_mapping(raw_weights)
    if sum(parsed_weights.values()) <= 0.0:
        raise ValueError("weight config sum must be positive")
    payload["weights"] = parsed_weights
    return payload


def apply_ensemble_weight_config(
    estimate_specs: Iterable[str | EstimateInput],
    weight_config: dict[str, Any],
    *,
    missing_weight_policy: str = "error",
    default_missing_weight: float = 0.0,
) -> list[EstimateInput]:
    """Apply selected weights after validating the in-memory config container."""

    if not isinstance(weight_config, dict):
        raise ValueError("weight config must be a mapping")
    return _ORIGINAL_APPLY(
        estimate_specs,
        weight_config,
        missing_weight_policy=missing_weight_policy,
        default_missing_weight=default_missing_weight,
    )


def write_apply_weights_outputs(**kwargs: Any) -> dict[str, Path]:
    """Write applied-weight outputs after validating the config container."""

    weight_config = kwargs.get("weight_config")
    if not isinstance(weight_config, dict):
        raise ValueError("weight config must be a mapping")
    return _ORIGINAL_WRITE(**kwargs)


_IMPL._validate_weight_value = _validate_weight_value
_IMPL._select_trim_fraction = _select_trim_fraction
_IMPL._validate_max_nearest_time_delta_s = _validate_max_nearest_time_delta_s
_IMPL.load_ensemble_weight_config = load_ensemble_weight_config
_IMPL.apply_ensemble_weight_config = apply_ensemble_weight_config
_IMPL.write_apply_weights_outputs = write_apply_weights_outputs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["EstimateInput"] = EstimateInput
globals()["_validate_weight_value"] = _validate_weight_value
globals()["_select_trim_fraction"] = _select_trim_fraction
globals()["_validate_max_nearest_time_delta_s"] = _validate_max_nearest_time_delta_s
globals()["load_ensemble_weight_config"] = load_ensemble_weight_config
globals()["apply_ensemble_weight_config"] = apply_ensemble_weight_config
globals()["write_apply_weights_outputs"] = write_apply_weights_outputs

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
sys.modules[__name__].__class__ = _ApplyWeightsModule
