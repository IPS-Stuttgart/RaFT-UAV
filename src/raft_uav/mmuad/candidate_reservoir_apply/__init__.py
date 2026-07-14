"""Compatibility wrapper with strict train-selected reservoir config validation.

The maintained implementation lives in the sibling ``candidate_reservoir_apply.py``
module. This package preserves the public import path while ensuring frozen
training controls are applied exactly rather than being silently truncated or
made non-finite during inference.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_apply.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_apply_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate reservoir application from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_REQUIRED_CONFIG_KEYS = tuple(_IMPL._REQUIRED_CONFIG_KEYS)
_INTEGER_CONFIG_KEYS = (
    "global_top_n",
    "per_source_top_n",
    "per_branch_top_n",
    "max_candidates_per_frame",
)
_ORIGINAL_APPLY_CONFIG = _IMPL.apply_train_selected_reservoir_config


def _scalar_value(value: Any, *, message: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    message = f"{name} must be a finite non-negative integer"
    value = _scalar_value(value, message=message)
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def _finite_float(value: Any, *, name: str) -> float:
    message = f"{name} must contain finite numeric values"
    value = _scalar_value(value, message=message)
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric):
        raise ValueError(message)
    return numeric


def _finite_float_mapping(value: Any, *, name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {
        str(key): _finite_float(item, name=f"{name}[{key!r}]")
        for key, item in value.items()
    }


def _optional_unit_interval(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    numeric = _finite_float(value, name=name)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return numeric


def _normalize_train_selected_reservoir_config(config: Any) -> dict[str, Any]:
    """Return an exact, finite version of one frozen reservoir configuration."""

    if not isinstance(config, dict):
        raise ValueError("candidate reservoir config JSON must contain an object")
    payload = dict(config)
    missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in payload]
    if missing:
        raise ValueError(f"candidate reservoir config missing required keys: {missing}")

    schema_version = _nonnegative_integer(
        payload.get("schema_version", 1),
        name="schema_version",
    )
    if schema_version != 1:
        raise ValueError(f"unsupported candidate reservoir config schema: {schema_version}")
    payload["schema_version"] = schema_version

    for key in _INTEGER_CONFIG_KEYS:
        payload[key] = _nonnegative_integer(payload[key], name=key)
    payload["branch_score_offsets"] = _finite_float_mapping(
        payload.get("branch_score_offsets", {}),
        name="branch_score_offsets",
    )
    payload["source_score_offsets"] = _finite_float_mapping(
        payload.get("source_score_offsets", {}),
        name="source_score_offsets",
    )
    payload["score_floor_quantile"] = _optional_unit_interval(
        payload.get("score_floor_quantile"),
        name="score_floor_quantile",
    )
    return payload


def load_train_selected_reservoir_config(path: Path) -> dict[str, Any]:
    """Load a frozen config while rejecting lossy and non-finite controls."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return _normalize_train_selected_reservoir_config(payload)


def apply_train_selected_reservoir_config(
    candidates,
    config: dict[str, Any],
    *,
    cap_mode: str = "score",
    diversity_min_per_source: int = 1,
    diversity_min_per_branch: int = 1,
    spatial_diversity_weight: float = 1.0,
    spatial_diversity_scale_m: float = 10.0,
    spatial_distance_cap_m: float = 50.0,
):
    """Apply an exact frozen reservoir config without lossy numeric coercion."""

    normalized = _normalize_train_selected_reservoir_config(config)
    return _ORIGINAL_APPLY_CONFIG(
        candidates,
        normalized,
        cap_mode=cap_mode,
        diversity_min_per_source=diversity_min_per_source,
        diversity_min_per_branch=diversity_min_per_branch,
        spatial_diversity_weight=spatial_diversity_weight,
        spatial_diversity_scale_m=spatial_diversity_scale_m,
        spatial_distance_cap_m=spatial_distance_cap_m,
    )


_IMPL.load_train_selected_reservoir_config = load_train_selected_reservoir_config
_IMPL.apply_train_selected_reservoir_config = apply_train_selected_reservoir_config

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_scalar_value"] = _scalar_value
globals()["_nonnegative_integer"] = _nonnegative_integer
globals()["_finite_float"] = _finite_float
globals()["_finite_float_mapping"] = _finite_float_mapping
globals()["_optional_unit_interval"] = _optional_unit_interval
globals()["_normalize_train_selected_reservoir_config"] = (
    _normalize_train_selected_reservoir_config
)
globals()["load_train_selected_reservoir_config"] = (
    load_train_selected_reservoir_config
)
globals()["apply_train_selected_reservoir_config"] = (
    apply_train_selected_reservoir_config
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
