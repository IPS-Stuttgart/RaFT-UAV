"""Harden candidate-reservoir config loading and summary frame counting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_COUNT_KEYS = (
    "global_top_n",
    "per_source_top_n",
    "per_branch_top_n",
    "max_candidates_per_frame",
)
_SCORE_COLUMN_KEYS = ("score_column", "fallback_score_column")
_SUMMARY_FRAME_COUNT_PATCH_MARKER = (
    "_raft_uav_summary_frame_count_without_candidate_schema_installed"
)


def _exact_integer_scalar(value: object, *, name: str) -> int:
    """Return an exact finite integer scalar without Boolean coercion."""

    error = f"{name} must be an exact integer scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        item = scalar.item()
        if np.ma.is_masked(item) or isinstance(
            item,
            (bool, np.bool_, complex, np.complexfloating),
        ):
            raise ValueError(error)
        if isinstance(item, (int, np.integer)):
            return int(item)
        if isinstance(item, (float, np.floating)):
            if not np.isfinite(item):
                raise ValueError(error)
            integer = int(item)
            if item != integer:
                raise ValueError(error)
            return integer
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(error)
    return int(number)


def _nonnegative_integer_scalar(value: object, *, name: str) -> int:
    """Return a non-negative exact integer scalar."""

    number = _exact_integer_scalar(value, name=name)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative exact integer scalar")
    return number


def _finite_real_scalar(value: object, *, name: str) -> float:
    """Return a finite scalar float without Boolean or complex coercion."""

    error = f"{name} must be a finite real scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        item = scalar.item()
        if np.ma.is_masked(item) or isinstance(
            item,
            (bool, np.bool_, complex, np.complexfloating),
        ):
            raise ValueError(error)
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number):
        raise ValueError(error)
    return number


def _unit_interval_scalar(value: object, *, name: str) -> float:
    """Return a finite scalar constrained to the closed unit interval."""

    number = _finite_real_scalar(value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _nonempty_string(value: object, *, name: str) -> str:
    """Return a non-empty string without silently stringifying other types."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_float_mapping(value: object, *, name: str) -> dict[str, float]:
    """Normalize a JSON object whose values must be finite real scalars."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {
        str(key): _finite_real_scalar(item, name=f"{name}[{key!r}]")
        for key, item in value.items()
    }


def _install_summary_frame_count_patch() -> None:
    """Keep summary frame counting independent of candidate geometry columns."""

    from raft_uav.mmuad import candidate_reservoir as reservoir_module

    if getattr(reservoir_module, _SUMMARY_FRAME_COUNT_PATCH_MARKER, False):
        return

    original_frame_counts = reservoir_module._ORIGINAL_FRAME_COUNTS

    def flight_aware_frame_counts(rows: pd.DataFrame) -> pd.Series:
        frame = pd.DataFrame(rows).copy()
        state, _ = reservoir_module._flight_metadata_state(frame)
        if frame.empty or state == "absent":
            return original_frame_counts(frame)
        keys = reservoir_module._joint_scope_keys(frame)
        key_to_token, _ = reservoir_module._scope_token_maps((keys,))
        scoped = reservoir_module._apply_scope_tokens(frame, keys, key_to_token)
        return original_frame_counts(scoped)

    reservoir_module._frame_counts = flight_aware_frame_counts
    reservoir_module._IMPL._frame_counts = flight_aware_frame_counts
    setattr(reservoir_module, _SUMMARY_FRAME_COUNT_PATCH_MARKER, True)


def install() -> None:
    """Install strict config validation and candidate-reservoir safety guards."""

    from raft_uav.mmuad._candidate_reservoir_numeric_stability_patch import (
        install as install_numeric_stability,
    )

    install_numeric_stability()
    _install_summary_frame_count_patch()

    from raft_uav.mmuad import candidate_reservoir_apply as apply_module

    if getattr(
        apply_module,
        "_raft_uav_exact_schema_validation_installed",
        False,
    ):
        return

    def validated_loader(path: Path) -> dict[str, object]:
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("candidate reservoir config JSON must contain an object")
        schema_version = _exact_integer_scalar(
            payload.get("schema_version", 1),
            name="candidate reservoir config schema_version",
        )
        if schema_version != 1:
            raise ValueError(
                f"unsupported candidate reservoir config schema: {schema_version}"
            )
        missing = [
            key for key in apply_module._REQUIRED_CONFIG_KEYS if key not in payload
        ]
        if missing:
            raise ValueError(
                f"candidate reservoir config missing required keys: {missing}"
            )

        validated = dict(payload)
        validated["schema_version"] = schema_version
        for key in _COUNT_KEYS:
            validated[key] = _nonnegative_integer_scalar(payload[key], name=key)
        for key in _SCORE_COLUMN_KEYS:
            validated[key] = _nonempty_string(payload[key], name=key)

        score_floor_quantile = payload.get("score_floor_quantile")
        if score_floor_quantile is not None:
            validated["score_floor_quantile"] = _unit_interval_scalar(
                score_floor_quantile,
                name="score_floor_quantile",
            )

        validated["branch_score_offsets"] = _finite_float_mapping(
            payload.get("branch_score_offsets", {}),
            name="branch_score_offsets",
        )
        validated["source_score_offsets"] = _finite_float_mapping(
            payload.get("source_score_offsets", {}),
            name="source_score_offsets",
        )
        return validated

    apply_module.load_train_selected_reservoir_config = validated_loader
    apply_module._raft_uav_exact_schema_validation_installed = True
