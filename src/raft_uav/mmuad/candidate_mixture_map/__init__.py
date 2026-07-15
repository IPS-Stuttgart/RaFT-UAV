"""Compatibility fixes for candidate-mixture MAP.

The maintained implementation lives in the sibling ``candidate_mixture_map.py``
module. This package keeps the public import path while preserving opaque IDs in
CSV inputs, retaining complete candidate frames when target-template times fall
outside the configured matching tolerance, and validating numerical controls
before inference.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_mixture_map.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_map_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate-mixture implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_VALIDATE_CONFIG = _IMPL._validate_config


class _PandasCsvProxy:
    """Delegate pandas operations while preserving identifiers in plain CSV reads."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def read_csv(self, path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if args or kwargs:
            rows = self._module.read_csv(path, *args, **kwargs)
            out = rows.copy()
            out.columns = [str(column).strip() for column in out.columns]
            return out
        return read_estimate_csv(Path(path))


def _finite_scalar(value: Any, *, field: str) -> float:
    """Return a finite non-Boolean scalar with a field-specific error."""

    message = f"{field} must be a finite scalar"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric):
        raise ValueError(message)
    return numeric


def _integer_scalar(
    value: Any,
    *,
    field: str,
    minimum: int,
) -> int:
    """Return an integer-equivalent scalar satisfying the lower bound."""

    qualifier = "non-negative" if minimum == 0 else "positive"
    message = f"{field} must be a {qualifier} finite integer"
    try:
        numeric = _finite_scalar(value, field=field)
    except ValueError as exc:
        raise ValueError(message) from exc
    if numeric < float(minimum) or not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def _validate_config(config: Any) -> None:
    """Reject malformed controls before candidate preparation or numerical work."""

    _integer_scalar(config.top_k, field="top_k", minimum=0)
    _integer_scalar(config.iterations, field="iterations", minimum=1)

    finite_fields = (
        "default_sigma_m",
        "sigma_min_m",
        "sigma_max_m",
        "score_weight",
        "temperature",
        "sigma_log_weight",
        "huber_delta",
        "smoothness_weight",
        "anchor_weight",
        "tolerance_m",
        "target_time_tolerance_s",
        "uniform_weight_floor",
        "branch_balance",
        "source_balance",
        "responsibility_floor",
        "min_measurement_precision",
        "max_measurement_precision",
    )
    numeric = {
        field: _finite_scalar(getattr(config, field), field=field)
        for field in finite_fields
    }
    if not (
        0.0
        < numeric["min_measurement_precision"]
        <= numeric["max_measurement_precision"]
    ):
        raise ValueError(
            "measurement precision bounds must satisfy "
            "0 < min_measurement_precision <= max_measurement_precision"
        )
    _ORIGINAL_VALIDATE_CONFIG(config)


def _target_time_candidate_groups(
    sequence_rows: pd.DataFrame,
    *,
    candidate_times: np.ndarray,
    target_times: np.ndarray,
    tolerance_s: float,
) -> list[tuple[float, pd.DataFrame]]:
    """Match target times without collapsing a nearest timestamp to one row.

    Candidate tables commonly contain several hypotheses at each timestamp. If
    no timestamp lies inside the tolerance window, the legacy fallback selected
    one row by positional index. Keep every hypothesis from the nearest timestamp
    instead, matching the grouped behavior used when a timestamp is in tolerance.
    """

    groups: list[tuple[float, pd.DataFrame]] = []
    if len(sequence_rows) == 0 or len(target_times) == 0:
        return groups
    tolerance = max(float(tolerance_s), 0.0)
    for target_time in target_times:
        left = int(np.searchsorted(candidate_times, target_time - tolerance, side="left"))
        right = int(np.searchsorted(candidate_times, target_time + tolerance, side="right"))
        if right <= left:
            nearest = int(np.argmin(np.abs(candidate_times - target_time)))
            nearest_time = float(candidate_times[nearest])
            left = int(np.searchsorted(candidate_times, nearest_time, side="left"))
            right = int(np.searchsorted(candidate_times, nearest_time, side="right"))
        groups.append((float(target_time), sequence_rows.iloc[left:right]))
    return groups


_IMPL.pd = _PandasCsvProxy(pd)
_IMPL._validate_config = _validate_config
_IMPL._target_time_candidate_groups = _target_time_candidate_groups

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_scalar"] = _finite_scalar
globals()["_integer_scalar"] = _integer_scalar
globals()["_validate_config"] = _validate_config
globals()["_target_time_candidate_groups"] = _target_time_candidate_groups

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
