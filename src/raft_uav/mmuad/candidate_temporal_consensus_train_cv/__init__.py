"""Strict controls for temporal-consensus train-CV selection and apply.

The maintained implementation lives in the sibling
``candidate_temporal_consensus_train_cv.py`` module. This package preserves the
public import path while rejecting lossy grid, integer, schema, and identifier
coercions before they can change selected hyperparameters or frozen provenance.
"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_temporal_consensus import (
    TemporalConsensusConfig,
    _validated_config,
    add_temporal_candidate_consensus,
)
from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / (
    "candidate_temporal_consensus_train_cv.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_temporal_consensus_train_cv_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        "cannot load temporal-consensus train-CV implementation from "
        f"{_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SELECT_TEMPORAL_CONSENSUS_CONFIG_BY_SEQUENCE_CV = (
    _IMPL.select_temporal_consensus_config_by_sequence_cv
)
_ORIGINAL_LOAD_TRAIN_SELECTED_TEMPORAL_CONSENSUS_CONFIG = (
    _IMPL.load_train_selected_temporal_consensus_config
)
_CONFIG_SCHEMA_VERSION = _IMPL._CONFIG_SCHEMA_VERSION


def _sequence(values: object, *, name: str) -> tuple[object, ...]:
    """Return one finite control axis without splitting scalar text."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence")
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence") from exc
    if not normalized:
        raise ValueError(f"{name} must be a non-empty sequence")
    return normalized


def _finite_real(value: object, *, name: str) -> float:
    """Return one finite real scalar without Boolean or container coercion."""

    normalized = optional_float(value)
    if normalized is None:
        raise ValueError(f"{name} must be a finite real scalar")
    return normalized


def _positive_real(value: object, *, name: str) -> float:
    normalized = _finite_real(value, name=name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_real(value: object, *, name: str) -> float:
    normalized = _finite_real(value, name=name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _finite_grid(values: object, *, name: str) -> tuple[float, ...]:
    """Normalize a grid axis and reject malformed entries instead of dropping them."""

    normalized = tuple(
        _finite_real(value, name=name)
        for value in _sequence(values, name=name)
    )
    return tuple(sorted(set(normalized)))


def _positive_int_grid(values: object, *, name: str) -> tuple[int, ...]:
    """Normalize positive exact integers without truncating fractional values."""

    normalized: list[int] = []
    for value in _sequence(values, name=name):
        integer = optional_int(value)
        if integer is None or integer <= 0:
            raise ValueError(f"{name} must contain positive exact integers")
        normalized.append(integer)
    return tuple(sorted(set(normalized)))


def _nonblank_string(value: object, *, name: str) -> str:
    if not isinstance(value, (str, np.str_)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-blank string")
    return str(value)


def _validated_temporal_config_values(
    config_values: object,
) -> TemporalConsensusConfig:
    """Build one normalized frozen config at the public apply boundary."""

    if not isinstance(config_values, dict):
        raise ValueError(
            "temporal-consensus config missing temporal_consensus_config"
        )
    allowed = {field.name for field in fields(TemporalConsensusConfig)}
    unknown = sorted(set(config_values) - allowed)
    if unknown:
        raise ValueError(f"unknown temporal-consensus config keys: {unknown}")
    config = _validated_config(TemporalConsensusConfig(**config_values))
    return replace(
        config,
        score_column=_nonblank_string(
            config.score_column,
            name="score_column",
        ),
        fallback_score_column=_nonblank_string(
            config.fallback_score_column,
            name="fallback_score_column",
        ),
    )


def select_temporal_consensus_config_by_sequence_cv(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    base_score_weights: Sequence[float] = (0.1, 0.25, 0.5),
    support_weights: Sequence[float] = (0.5, 1.0, 1.5),
    bidirectional_bonuses: Sequence[float] = (0.0, 0.75),
    interpolation_weights: Sequence[float] = (0.0, 0.75),
    acceleration_weights: Sequence[float] = (0.0, 0.5),
    max_time_gap_s: float = 2.0,
    max_speed_mps: float = 60.0,
    distance_scale_m: float = 5.0,
    acceleration_scale_mps2: float = 20.0,
    score_column: str = "ranker_score",
    fallback_score_column: str = "confidence",
    source_diversity_bonus: float = 0.25,
    branch_diversity_bonus: float = 0.25,
    top_k_values: Sequence[int] = _IMPL._DEFAULT_TOP_K,
    max_truth_time_delta_s: float = 0.5,
    selection_metric: str = _IMPL._DEFAULT_SELECTION_METRIC,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select train-CV controls after lossless public-boundary validation."""

    return _ORIGINAL_SELECT_TEMPORAL_CONSENSUS_CONFIG_BY_SEQUENCE_CV(
        candidates,
        truth,
        base_score_weights=_finite_grid(
            base_score_weights,
            name="base_score_weights",
        ),
        support_weights=_finite_grid(
            support_weights,
            name="support_weights",
        ),
        bidirectional_bonuses=_finite_grid(
            bidirectional_bonuses,
            name="bidirectional_bonuses",
        ),
        interpolation_weights=_finite_grid(
            interpolation_weights,
            name="interpolation_weights",
        ),
        acceleration_weights=_finite_grid(
            acceleration_weights,
            name="acceleration_weights",
        ),
        max_time_gap_s=_positive_real(
            max_time_gap_s,
            name="max_time_gap_s",
        ),
        max_speed_mps=_positive_real(max_speed_mps, name="max_speed_mps"),
        distance_scale_m=_positive_real(
            distance_scale_m,
            name="distance_scale_m",
        ),
        acceleration_scale_mps2=_positive_real(
            acceleration_scale_mps2,
            name="acceleration_scale_mps2",
        ),
        score_column=_nonblank_string(score_column, name="score_column"),
        fallback_score_column=_nonblank_string(
            fallback_score_column,
            name="fallback_score_column",
        ),
        source_diversity_bonus=_finite_real(
            source_diversity_bonus,
            name="source_diversity_bonus",
        ),
        branch_diversity_bonus=_finite_real(
            branch_diversity_bonus,
            name="branch_diversity_bonus",
        ),
        top_k_values=_positive_int_grid(
            top_k_values,
            name="top_k_values",
        ),
        max_truth_time_delta_s=_nonnegative_real(
            max_truth_time_delta_s,
            name="max_truth_time_delta_s",
        ),
        selection_metric=_nonblank_string(
            selection_metric,
            name="selection_metric",
        ),
    )


def load_train_selected_temporal_consensus_config(path: Path) -> dict[str, Any]:
    """Load a frozen config without accepting lossy schema or control coercion."""

    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("temporal-consensus config JSON must contain an object")
    if "schema_version" in raw:
        schema_version = raw["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("schema_version must be an exact integer")
        if schema_version != _CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "unsupported temporal-consensus config schema: "
                f"{schema_version}"
            )

    normalized = _ORIGINAL_LOAD_TRAIN_SELECTED_TEMPORAL_CONSENSUS_CONFIG(
        config_path
    )
    config = _validated_temporal_config_values(
        normalized.get("temporal_consensus_config")
    )
    normalized["schema_version"] = _CONFIG_SCHEMA_VERSION
    normalized["temporal_consensus_config"] = asdict(config)
    return normalized


def apply_train_selected_temporal_consensus(
    candidates: CandidateFrame | pd.DataFrame,
    payload: dict[str, Any],
) -> CandidateFrame:
    """Apply a frozen config after validating its numeric and identifier controls."""

    if not isinstance(payload, dict):
        raise ValueError("temporal-consensus payload must be an object")
    config = _validated_temporal_config_values(
        payload.get("temporal_consensus_config")
    )
    return add_temporal_candidate_consensus(candidates, config=config)


_IMPL.select_temporal_consensus_config_by_sequence_cv = (
    select_temporal_consensus_config_by_sequence_cv
)
_IMPL.load_train_selected_temporal_consensus_config = (
    load_train_selected_temporal_consensus_config
)
_IMPL.apply_train_selected_temporal_consensus = (
    apply_train_selected_temporal_consensus
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_sequence"] = _sequence
globals()["_finite_real"] = _finite_real
globals()["_positive_real"] = _positive_real
globals()["_nonnegative_real"] = _nonnegative_real
globals()["_finite_grid"] = _finite_grid
globals()["_positive_int_grid"] = _positive_int_grid
globals()["_nonblank_string"] = _nonblank_string
globals()["_validated_temporal_config_values"] = (
    _validated_temporal_config_values
)
globals()["select_temporal_consensus_config_by_sequence_cv"] = (
    select_temporal_consensus_config_by_sequence_cv
)
globals()["load_train_selected_temporal_consensus_config"] = (
    load_train_selected_temporal_consensus_config
)
globals()["apply_train_selected_temporal_consensus"] = (
    apply_train_selected_temporal_consensus
)

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
