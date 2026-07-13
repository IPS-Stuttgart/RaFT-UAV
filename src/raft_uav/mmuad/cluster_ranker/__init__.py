"""Compatibility wrapper with strict cluster-ranker integer controls.

The maintained implementation lives in the sibling ``cluster_ranker.py`` module.
This package preserves the public import path while preventing lossy coercion of
training and LOSO evaluation controls.
"""

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
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load cluster-ranker implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRAIN_CLUSTER_RANKER = _IMPL.train_cluster_ranker
_ORIGINAL_EVALUATE_CLUSTER_RANKER_LOSO = _IMPL.evaluate_cluster_ranker_loso
_ORIGINAL_MAKE_SKLEARN_ESTIMATOR = _IMPL._make_sklearn_estimator
_ESTIMATOR_COUNT_MODELS = {
    "random-forest-classifier",
    "hist-gradient-boosting-classifier",
    "random-forest-regressor",
    "hist-gradient-boosting-regressor",
}


def _integer_control(
    value: object,
    *,
    name: str,
    minimum: int,
    qualifier: str,
) -> int:
    """Return an exact integer control without Boolean or fractional coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a {qualifier} integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a {qualifier} integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(numeric)


def _positive_integer(value: object, *, name: str) -> int:
    return _integer_control(value, name=name, minimum=1, qualifier="positive")


def _nonnegative_integer(value: object, *, name: str) -> int:
    return _integer_control(value, name=name, minimum=0, qualifier="nonnegative")


def _validated_training_counts(
    *,
    model_type: object,
    iterations: object,
    n_estimators: object,
) -> tuple[str, object, object]:
    selected_model = str(model_type)
    validated_iterations = (
        _positive_integer(iterations, name="iterations")
        if selected_model == "logistic"
        else iterations
    )
    validated_n_estimators = (
        _positive_integer(n_estimators, name="n_estimators")
        if selected_model in _ESTIMATOR_COUNT_MODELS
        else n_estimators
    )
    return selected_model, validated_iterations, validated_n_estimators


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
):
    """Train after validating integer controls used by the selected model."""

    selected_model, validated_iterations, validated_n_estimators = (
        _validated_training_counts(
            model_type=model_type,
            iterations=iterations,
            n_estimators=n_estimators,
        )
    )
    return _ORIGINAL_TRAIN_CLUSTER_RANKER(
        features,
        model_type=selected_model,
        target_column=target_column,
        learning_rate=learning_rate,
        iterations=validated_iterations,
        l2=l2,
        random_state=random_state,
        n_estimators=validated_n_estimators,
        score_distance_scale_m=score_distance_scale_m,
    )


def evaluate_cluster_ranker_loso(
    features: pd.DataFrame,
    *,
    model_type: str = "logistic",
    target_column: str = "good_cluster",
    learning_rate: float = 0.05,
    iterations: int = 600,
    random_state: int = 13,
    n_estimators: int = 200,
    score_distance_scale_m: float = 10.0,
    min_train_sequences: int = 1,
    protocol: str = "LOSO public-validation diagnostic, not submission-valid",
):
    """Evaluate LOSO after validating all active integer controls."""

    selected_model, validated_iterations, validated_n_estimators = (
        _validated_training_counts(
            model_type=model_type,
            iterations=iterations,
            n_estimators=n_estimators,
        )
    )
    validated_minimum = _nonnegative_integer(
        min_train_sequences,
        name="min_train_sequences",
    )
    return _ORIGINAL_EVALUATE_CLUSTER_RANKER_LOSO(
        features,
        model_type=selected_model,
        target_column=target_column,
        learning_rate=learning_rate,
        iterations=validated_iterations,
        random_state=random_state,
        n_estimators=validated_n_estimators,
        score_distance_scale_m=score_distance_scale_m,
        min_train_sequences=validated_minimum,
        protocol=protocol,
    )


def _make_sklearn_estimator(
    *,
    model_type: str,
    random_state: int,
    n_estimators: int,
) -> tuple[Any, str]:
    """Construct an estimator without truncating its requested iteration count."""

    count = (
        _positive_integer(n_estimators, name="n_estimators")
        if str(model_type) in _ESTIMATOR_COUNT_MODELS
        else n_estimators
    )
    return _ORIGINAL_MAKE_SKLEARN_ESTIMATOR(
        model_type=model_type,
        random_state=random_state,
        n_estimators=count,
    )


_IMPL.train_cluster_ranker = train_cluster_ranker
_IMPL.evaluate_cluster_ranker_loso = evaluate_cluster_ranker_loso
_IMPL._make_sklearn_estimator = _make_sklearn_estimator

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_integer_control"] = _integer_control
globals()["_positive_integer"] = _positive_integer
globals()["_nonnegative_integer"] = _nonnegative_integer
globals()["_validated_training_counts"] = _validated_training_counts
globals()["train_cluster_ranker"] = train_cluster_ranker
globals()["evaluate_cluster_ranker_loso"] = evaluate_cluster_ranker_loso
globals()["_make_sklearn_estimator"] = _make_sklearn_estimator

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
