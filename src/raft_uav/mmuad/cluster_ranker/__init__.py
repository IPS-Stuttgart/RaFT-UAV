"""Compatibility package that normalizes serialized cluster-ranker targets."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "cluster_ranker.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._cluster_ranker_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load cluster-ranker implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_TRAIN_CLUSTER_RANKER = _IMPL.train_cluster_ranker
_LEGACY_BINARY_AUC = _IMPL._binary_auc
_INVALID_BINARY_TARGET = object()
_TRUE_TARGET_TOKENS = frozenset({"true", "t", "yes", "y", "on"})
_FALSE_TARGET_TOKENS = frozenset({"false", "f", "no", "n", "off"})
_MISSING_TARGET_TOKENS = frozenset({"", "nan", "none", "null", "<na>", "nat"})


def train_cluster_ranker(
    features: pd.DataFrame,
    *,
    model_type: str = "logistic",
    target_column: str = "good_cluster",
    learning_rate: float = 0.05,
    iterations: int = 600,
    l2: float = 1.0e-3,
    random_state: int = 13,
    n_estimators: int = 200,
    score_distance_scale_m: float = 10.0,
) -> Any:
    """Train after parsing persisted classifier targets as actual booleans."""

    normalized = pd.DataFrame(features).copy()
    normalized_model_type = str(model_type)
    actual_target = _IMPL._actual_target_column(
        normalized,
        model_type=normalized_model_type,
        target_column=target_column,
    )
    if actual_target not in normalized.columns:
        raise ValueError(f"cluster-ranker target column is missing: {actual_target!r}")
    if not normalized_model_type.endswith("-regressor"):
        normalized[actual_target] = _normalize_binary_targets(
            normalized[actual_target],
            target_column=actual_target,
        )
    return _LEGACY_TRAIN_CLUSTER_RANKER(
        normalized,
        model_type=normalized_model_type,
        target_column=target_column,
        learning_rate=learning_rate,
        iterations=iterations,
        l2=l2,
        random_state=random_state,
        n_estimators=n_estimators,
        score_distance_scale_m=score_distance_scale_m,
    )


def _binary_auc(scores: pd.Series, labels: pd.Series) -> float:
    normalized = _normalize_binary_targets(
        pd.Series(labels),
        target_column="labels",
    )
    valid = normalized.notna()
    if not bool(valid.any()):
        return float("nan")
    score_series = pd.Series(scores)
    return float(
        _LEGACY_BINARY_AUC(
            score_series.loc[valid],
            normalized.loc[valid],
        )
    )


def _normalize_binary_targets(
    values: pd.Series,
    *,
    target_column: str,
) -> pd.Series:
    normalized: list[object] = []
    invalid: list[tuple[object, object]] = []
    for index, value in values.items():
        parsed = _binary_target_value(value)
        if parsed is _INVALID_BINARY_TARGET:
            invalid.append((index, value))
            normalized.append(pd.NA)
        else:
            normalized.append(parsed)
    if invalid:
        preview = ", ".join(
            f"index {index!r}: {value!r}" for index, value in invalid[:5]
        )
        suffix = "" if len(invalid) <= 5 else f"; plus {len(invalid) - 5} more"
        raise ValueError(
            f"binary target column {target_column!r} contains invalid values: "
            f"{preview}{suffix}"
        )
    return pd.Series(normalized, index=values.index, dtype="boolean")


def _binary_target_value(value: object) -> object:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, (int, np.integer, float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return pd.NA
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
        return _INVALID_BINARY_TARGET
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _MISSING_TARGET_TOKENS:
            return pd.NA
        if text in _TRUE_TARGET_TOKENS:
            return True
        if text in _FALSE_TARGET_TOKENS:
            return False
        try:
            numeric = float(text)
        except ValueError:
            return _INVALID_BINARY_TARGET
        if not np.isfinite(numeric):
            return pd.NA
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
        return _INVALID_BINARY_TARGET
    try:
        return pd.NA if bool(pd.isna(value)) else _INVALID_BINARY_TARGET
    except (TypeError, ValueError):
        return _INVALID_BINARY_TARGET


_IMPL.train_cluster_ranker = train_cluster_ranker
_IMPL._binary_auc = _binary_auc

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["train_cluster_ranker"] = train_cluster_ranker
globals()["_binary_auc"] = _binary_auc
globals()["_normalize_binary_targets"] = _normalize_binary_targets
globals()["_binary_target_value"] = _binary_target_value

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
