"""Compatibility fixes for cluster-ranker training and truth labeling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float
from raft_uav.numeric import optional_int

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
_LEGACY_LABEL_CLUSTER_FEATURES_AGAINST_TRUTH = (
    _IMPL.label_cluster_features_against_truth
)
_LEGACY_BINARY_AUC = _IMPL._binary_auc
_LEGACY_RANKER_PREDICTION_SUMMARY = _IMPL._ranker_prediction_summary
_INVALID_BINARY_TARGET = object()
_TRUE_TARGET_TOKENS = frozenset({"true", "t", "yes", "y", "on"})
_FALSE_TARGET_TOKENS = frozenset({"false", "f", "no", "n", "off"})
_MISSING_TARGET_TOKENS = frozenset({"", "nan", "none", "null", "<na>", "nat"})
_TRUTH_LABEL_COLUMNS = (
    "good_cluster_2m",
    "good_cluster_5m",
    "good_cluster_10m",
    "good_cluster_20m",
    "good_cluster",
)
_MAX_RANDOM_STATE = int(np.iinfo(np.uint32).max)


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
    """Train after validating controls and parsing persisted binary targets."""

    normalized_learning_rate = _validated_positive_real(
        learning_rate,
        name="learning_rate",
    )
    normalized_iterations = _validated_positive_int(
        iterations,
        name="iterations",
    )
    normalized_l2 = _validated_nonnegative_real(
        l2,
        name="l2",
    )
    normalized_random_state = _validated_random_state(random_state)
    normalized_n_estimators = _validated_positive_int(
        n_estimators,
        name="n_estimators",
    )
    normalized_score_distance_scale_m = _validated_positive_real(
        score_distance_scale_m,
        name="score_distance_scale_m",
    )

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
        learning_rate=normalized_learning_rate,
        iterations=normalized_iterations,
        l2=normalized_l2,
        random_state=normalized_random_state,
        n_estimators=normalized_n_estimators,
        score_distance_scale_m=normalized_score_distance_scale_m,
    )


def label_cluster_features_against_truth(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    good_threshold_m: float = 5.0,
    max_truth_time_delta_s: float = 0.5,
) -> pd.DataFrame:
    """Attach truth labels using the authoritative final same-time truth row."""

    distance_gate = _validated_nonnegative_gate(
        good_threshold_m,
        name="good_threshold_m",
    )
    time_gate = _validated_nonnegative_gate(
        max_truth_time_delta_s,
        name="max_truth_time_delta_s",
    )
    authoritative_truth = _authoritative_truth_rows(truth)
    labeled = _LEGACY_LABEL_CLUSTER_FEATURES_AGAINST_TRUTH(
        features,
        authoritative_truth,
        good_threshold_m=distance_gate,
        max_truth_time_delta_s=time_gate,
    )
    if "truth_matched" not in labeled.columns:
        return labeled

    matched = (
        pd.Series(labeled["truth_matched"], index=labeled.index)
        .fillna(False)
        .astype(bool)
    )
    for column in _TRUTH_LABEL_COLUMNS:
        if column not in labeled.columns:
            continue
        labeled[column] = pd.Series(
            labeled[column],
            index=labeled.index,
            dtype="boolean",
        )
        labeled.loc[~matched, column] = pd.NA
    return labeled


def _authoritative_truth_rows(truth: Any) -> pd.DataFrame:
    """Keep the final finite row for each normalized sequence timestamp."""

    rows = _IMPL._truth_rows(truth).reset_index(drop=True)
    if rows.empty:
        return rows
    rows["_truth_input_order"] = np.arange(len(rows), dtype=np.int64)
    return (
        rows.sort_values(
            ["sequence_id", "time_s", "_truth_input_order"],
            kind="mergesort",
        )
        .drop_duplicates(["sequence_id", "time_s"], keep="last")
        .drop(columns="_truth_input_order")
        .reset_index(drop=True)
    )


def _validated_nonnegative_gate(value: object, *, name: str) -> float:
    """Return a finite non-negative scalar gate or raise a stable error."""

    normalized = optional_float(value)
    if normalized is None or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return normalized


def _validated_positive_real(value: object, *, name: str) -> float:
    """Return a finite positive real scalar or raise a stable error."""

    normalized = optional_float(value)
    if normalized is None or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return normalized


def _validated_nonnegative_real(value: object, *, name: str) -> float:
    """Return a finite non-negative real scalar or raise a stable error."""

    normalized = optional_float(value)
    if normalized is None or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return normalized


def _validated_positive_int(value: object, *, name: str) -> int:
    """Return an exact positive integer or raise a stable error."""

    normalized = optional_int(value)
    if normalized is None or normalized <= 0:
        raise ValueError(f"{name} must be an exact positive integer")
    return normalized


def _validated_random_state(value: object) -> int:
    """Return an exact seed accepted by NumPy and scikit-learn."""

    normalized = optional_int(value)
    if normalized is None or not 0 <= normalized <= _MAX_RANDOM_STATE:
        raise ValueError(
            "random_state must be an exact integer in "
            f"[0, {_MAX_RANDOM_STATE}]"
        )
    return normalized


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


def _ranker_prediction_summary(
    rows: pd.DataFrame,
    *,
    sequence: str,
    split: str,
    protocol: str,
) -> dict[str, Any]:
    """Summarize predictions after preserving serialized target semantics."""

    normalized = pd.DataFrame(rows).copy()
    if "good_cluster" in normalized.columns:
        normalized["good_cluster"] = _normalize_binary_targets(
            normalized["good_cluster"],
            target_column="good_cluster",
        )
    return _LEGACY_RANKER_PREDICTION_SUMMARY(
        normalized,
        sequence=sequence,
        split=split,
        protocol=protocol,
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
    if isinstance(value, (int, np.integer)):
        if value == 0:
            return False
        if value == 1:
            return True
        return _INVALID_BINARY_TARGET
    if isinstance(value, (float, np.floating)):
        if bool(np.isnan(value)):
            return pd.NA
        if not bool(np.isfinite(value)):
            return _INVALID_BINARY_TARGET
        if value == 0.0:
            return False
        if value == 1.0:
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
        if np.isnan(numeric):
            return pd.NA
        if not np.isfinite(numeric):
            return _INVALID_BINARY_TARGET
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
_IMPL.label_cluster_features_against_truth = label_cluster_features_against_truth
_IMPL._binary_auc = _binary_auc
_IMPL._ranker_prediction_summary = _ranker_prediction_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["train_cluster_ranker"] = train_cluster_ranker
globals()["label_cluster_features_against_truth"] = (
    label_cluster_features_against_truth
)
globals()["_authoritative_truth_rows"] = _authoritative_truth_rows
globals()["_validated_nonnegative_gate"] = _validated_nonnegative_gate
globals()["_validated_positive_real"] = _validated_positive_real
globals()["_validated_nonnegative_real"] = _validated_nonnegative_real
globals()["_validated_positive_int"] = _validated_positive_int
globals()["_validated_random_state"] = _validated_random_state
globals()["_binary_auc"] = _binary_auc
globals()["_ranker_prediction_summary"] = _ranker_prediction_summary
globals()["_normalize_binary_targets"] = _normalize_binary_targets
globals()["_binary_target_value"] = _binary_target_value

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
