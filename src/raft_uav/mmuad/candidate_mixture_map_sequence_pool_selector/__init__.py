"""Sequence-aware wrapper for candidate-mixture pool selection.

The maintained implementation lives in the sibling
``candidate_mixture_map_sequence_pool_selector.py`` module. This package keeps
its public import path while making external initial estimates use the same
sequence alias and sequence-less expansion rules as the per-sequence
multi-start workflow, and while rejecting malformed selector settings before
lossy coercion or truthiness can alter the requested pool search.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad import candidate_mixture_map_sequence_multistart as sequence_multistart

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_mixture_map_sequence_pool_selector.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_map_sequence_pool_selector_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load candidate-mixture sequence pool selector implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RUN_SEQUENCE_POOL_SELECTOR = _IMPL.run_sequence_pool_selector
_ORIGINAL_BUILD_SEQUENCE_CANDIDATE_POOL_VARIANTS = (
    _IMPL.build_sequence_candidate_pool_variants
)
_BOOLEAN_FIELDS = (
    "include_full_pool",
    "include_leave_one_out",
    "restore_missing_frames",
    "normalize_component_count",
)

# Export the maintained implementation first; corrected functions below replace
# the affected callables while preserving all existing public/private helpers.
globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _normalize_sequence_pool_initialization(
    candidates: pd.DataFrame,
    initial_estimates: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Canonicalize sequence aliases or expand one shared trajectory per sequence."""

    if initial_estimates is None:
        return None

    rows = pd.DataFrame(initial_estimates).copy()
    rows.columns = [str(column).strip() for column in rows.columns]
    if not rows.empty:
        sequence_column = sequence_multistart._first_present_column(
            rows,
            sequence_multistart._SEQUENCE_ALIASES,
        )
        if sequence_column is not None:
            sequence_ids = sequence_multistart._sequence_id_text(rows[sequence_column])
            if not sequence_ids.ne("").any():
                rows = rows.drop(columns=[sequence_column])

    return sequence_multistart._expand_sequence_less_external_initialization(
        candidates,
        rows,
    )


def _finite_real_scalar(value: Any, *, name: str) -> float:
    """Return one finite real scalar without accepting Boolean pseudo-numbers."""

    message = f"{name} must be a finite real scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(message)
    item = scalar.item()
    if np.ma.is_masked(item) or isinstance(item, (bool, np.bool_)):
        raise ValueError(message)
    try:
        number = float(item)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def _nonnegative_integer(value: Any, *, name: str) -> int:
    """Return an exact non-negative integer without truncation."""

    number = _finite_real_scalar(value, name=name)
    integer = int(number)
    if number != float(integer) or integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


def _boolean(value: Any, *, name: str) -> bool:
    """Return a genuine Boolean configuration value."""

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a Boolean")
    return bool(value)


def _validated_selector_config(config: Any) -> Any:
    """Return a normalized selector config without lossy coercion."""

    config_class = _IMPL.CandidatePoolSequenceSelectorConfig
    if not isinstance(config, config_class):
        raise TypeError(
            "selector_config must be a CandidatePoolSequenceSelectorConfig "
            "instance or None"
        )
    if not isinstance(config.group_column, str) or not config.group_column.strip():
        raise ValueError("group_column must be a non-empty string")
    max_leave_one_out = _nonnegative_integer(
        config.max_leave_one_out,
        name="max_leave_one_out",
    )
    frame_fraction = _finite_real_scalar(
        config.min_group_frame_fraction,
        name="min_group_frame_fraction",
    )
    if not 0.0 <= frame_fraction <= 1.0:
        raise ValueError("min_group_frame_fraction must be within [0, 1]")
    flags = {
        field_name: _boolean(getattr(config, field_name), name=field_name)
        for field_name in _BOOLEAN_FIELDS
    }
    if not flags["include_full_pool"] and not flags["include_leave_one_out"]:
        raise ValueError("at least one candidate-pool variant must be enabled")
    return replace(
        config,
        group_column=config.group_column.strip(),
        max_leave_one_out=max_leave_one_out,
        min_group_frame_fraction=frame_fraction,
        **flags,
    )


def _validate_selector_config(config: Any) -> None:
    """Validate one selector config at the legacy internal boundary."""

    _validated_selector_config(config)


def _resolved_selector_config(config: Any) -> Any:
    """Resolve defaults only for ``None`` and validate explicit configurations."""

    resolved = (
        _IMPL.CandidatePoolSequenceSelectorConfig()
        if config is None
        else config
    )
    return _validated_selector_config(resolved)


def _resolved_mixture_config(config: Any) -> Any:
    """Resolve the mixture default without hiding malformed falsey objects."""

    config_class = _IMPL.core.CandidateMixtureMapConfig
    if config is None:
        return config_class()
    if not isinstance(config, config_class):
        raise TypeError(
            "mixture_config must be a CandidateMixtureMapConfig instance or None"
        )
    return config


def run_sequence_pool_selector(
    candidates: pd.DataFrame,
    *,
    mixture_config: Any | None = None,
    selector_config: Any | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> Any:
    """Run pool selection with normalized initialization and strict configs."""

    normalized_initial = _normalize_sequence_pool_initialization(
        candidates,
        initial_estimates,
    )
    return _ORIGINAL_RUN_SEQUENCE_POOL_SELECTOR(
        candidates,
        mixture_config=_resolved_mixture_config(mixture_config),
        selector_config=_resolved_selector_config(selector_config),
        initial_estimates=normalized_initial,
        truth=truth,
    )


def build_sequence_candidate_pool_variants(
    candidates: pd.DataFrame,
    *,
    config: Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Build candidate-pool variants after strict config validation."""

    return _ORIGINAL_BUILD_SEQUENCE_CANDIDATE_POOL_VARIANTS(
        candidates,
        config=_resolved_selector_config(config),
    )


# Make the legacy CLI and function globals resolve the corrected behavior.
_IMPL._validate_selector_config = _validate_selector_config
_IMPL.run_sequence_pool_selector = run_sequence_pool_selector
_IMPL.build_sequence_candidate_pool_variants = build_sequence_candidate_pool_variants
globals()["run_sequence_pool_selector"] = run_sequence_pool_selector
globals()["build_sequence_candidate_pool_variants"] = (
    build_sequence_candidate_pool_variants
)
globals()["_normalize_sequence_pool_initialization"] = (
    _normalize_sequence_pool_initialization
)
globals()["_finite_real_scalar"] = _finite_real_scalar
globals()["_nonnegative_integer"] = _nonnegative_integer
globals()["_boolean"] = _boolean
globals()["_validated_selector_config"] = _validated_selector_config
globals()["_validate_selector_config"] = _validate_selector_config
globals()["_resolved_selector_config"] = _resolved_selector_config
globals()["_resolved_mixture_config"] = _resolved_mixture_config

__all__ = sorted(
    {
        *[
            name
            for name in dir(_IMPL)
            if not (name.startswith("__") and name.endswith("__"))
        ],
        "run_sequence_pool_selector",
        "build_sequence_candidate_pool_variants",
    }
)
