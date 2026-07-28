"""Compatibility fixes for candidate-oracle target exports.

The maintained implementation lives in the sibling ``candidate_oracle_targets.py``
module. This package preserves the public import path while rejecting malformed
truth-matching time gates, oracle-label thresholds, candidate-score controls,
and non-finite candidate-score values before they can silently widen, empty, or
corrupt the training export, and while keeping distinct floating-point thresholds
distinct in output column labels.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_targets.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_targets_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate-oracle targets from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_ORACLE_TARGETS = _IMPL.build_candidate_oracle_targets


def _validated_numeric_tuple(
    values: Any,
    *,
    name: str,
    strictly_positive: bool,
) -> tuple[float, ...]:
    """Return a tuple of finite scalar thresholds with the requested sign."""

    requirement = "positive" if strictly_positive else "non-negative"
    message = f"{name} must contain only finite {requirement} scalars"
    if values is None or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(message)
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(message) from exc

    normalized: list[float] = []
    for value in items:
        number = optional_float(value)
        invalid_sign = (
            number is not None
            and (number <= 0.0 if strictly_positive else number < 0.0)
        )
        if number is None or invalid_sign:
            raise ValueError(message)
        normalized.append(number)
    return tuple(normalized)


def _validated_score_column(value: Any, *, name: str) -> str:
    """Return one normalized, non-empty candidate-score column name."""

    message = f"{name} must be a non-empty string"
    if not isinstance(value, str):
        raise ValueError(message)
    normalized = value.strip()
    if not normalized:
        raise ValueError(message)
    return normalized


def _validated_score_columns(values: Any) -> tuple[str, ...]:
    """Return normalized fallback score-column names."""

    message = "fallback_score_columns must contain only non-empty strings"
    if values is None or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(message)
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(message) from exc
    try:
        return tuple(
            _validated_score_column(value, name="fallback_score_columns")
            for value in items
        )
    except ValueError as exc:
        raise ValueError(message) from exc


def _validated_config(
    config: _IMPL.CandidateOracleTargetConfig | None,
) -> _IMPL.CandidateOracleTargetConfig:
    """Return a config with valid matching, label, and score controls."""

    if config is None:
        resolved = _IMPL.CandidateOracleTargetConfig()
    elif not isinstance(config, _IMPL.CandidateOracleTargetConfig):
        raise TypeError(
            "config must be a CandidateOracleTargetConfig instance or None"
        )
    else:
        resolved = config

    max_delta = optional_float(resolved.max_truth_time_delta_s)
    if max_delta is None or max_delta < 0.0:
        raise ValueError(
            "max_truth_time_delta_s must be a finite non-negative scalar"
        )
    score_column = _validated_score_column(
        resolved.score_column,
        name="score_column",
    )
    fallback_score_columns = _validated_score_columns(
        resolved.fallback_score_columns
    )
    soft_tau_m = _validated_numeric_tuple(
        resolved.soft_tau_m,
        name="soft_tau_m",
        strictly_positive=True,
    )
    good_thresholds_m = _validated_numeric_tuple(
        resolved.good_thresholds_m,
        name="good_thresholds_m",
        strictly_positive=False,
    )
    return replace(
        resolved,
        max_truth_time_delta_s=max_delta,
        score_column=score_column,
        fallback_score_columns=fallback_score_columns,
        soft_tau_m=soft_tau_m,
        good_thresholds_m=good_thresholds_m,
    )


def _optional_candidate_score(value: object) -> float | None:
    """Recover real values from pandas columns upcast to complex dtype."""

    if isinstance(value, (complex, np.complexfloating)):
        imaginary = float(np.imag(value))
        if not np.isfinite(imaginary) or imaginary != 0.0:
            return None
        value = np.real(value)
    return optional_float(value)


def _candidate_score(
    rows: pd.DataFrame,
    *,
    config: _IMPL.CandidateOracleTargetConfig,
) -> pd.Series:
    """Use the first finite real score in the configured fallback chain."""

    columns = (config.score_column, *config.fallback_score_columns)
    result = pd.Series(float("nan"), index=rows.index, dtype=float)
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.Series(
            [_optional_candidate_score(value) for value in rows[column]],
            index=rows.index,
            dtype=float,
        )
        result = result.where(result.notna(), values)
    return result.fillna(0.0).astype(float)


def _threshold_label(value: float) -> str:
    """Return a column-safe shortest round-trip floating-point label."""

    text = repr(float(value))
    if text.endswith(".0") and "e" not in text.lower():
        text = text[:-2]
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def build_candidate_oracle_targets(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    config: _IMPL.CandidateOracleTargetConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build targets after validating all label-generation controls."""

    return _ORIGINAL_BUILD_CANDIDATE_ORACLE_TARGETS(
        candidates,
        truth,
        config=_validated_config(config),
    )


_IMPL._candidate_score = _candidate_score
_IMPL._threshold_label = _threshold_label
_IMPL.build_candidate_oracle_targets = build_candidate_oracle_targets

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_numeric_tuple"] = _validated_numeric_tuple
globals()["_validated_score_column"] = _validated_score_column
globals()["_validated_score_columns"] = _validated_score_columns
globals()["_validated_config"] = _validated_config
globals()["_optional_candidate_score"] = _optional_candidate_score
globals()["_candidate_score"] = _candidate_score
globals()["_threshold_label"] = _threshold_label
globals()["build_candidate_oracle_targets"] = build_candidate_oracle_targets

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
