"""Compatibility package for safe source-calibration transforms and lookup.

The maintained implementation lives in the sibling ``source_calibration.py`` module.
This package preserves the public import path while validating every loaded or fitted
source transform before it can contaminate calibrated candidate coordinates, rejecting
ambiguous case-insensitive transform keys, preventing source-specific transforms from
leaking onto unrelated or broader sources, retaining the authoritative final truth
sample at duplicate timestamps, and scoping pooled calibration by physical flight.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "source_calibration.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._source_calibration_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load source-calibration implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SOURCE_TRANSFORM_POST_INIT = _IMPL.SourceTransform.__post_init__
_ORIGINAL_NORMALIZE_TRUTH_ROWS = _IMPL._normalize_truth_rows
_ORIGINAL_BUILD_SOURCE_CALIBRATION_PAIRS = _IMPL.build_source_calibration_pairs
_ORIGINAL_FIT_SOURCE_TRANSLATION_ALPHA_CV = _IMPL._fit_source_translation_alpha_cv


def _validated_source_transform_post_init(self: object) -> None:
    """Normalize a source transform, then reject non-finite coefficients."""

    _ORIGINAL_SOURCE_TRANSFORM_POST_INIT(self)
    if not np.isfinite(self.linear).all():
        raise ValueError("linear transform must contain only finite values")
    if not np.isfinite(self.translation_m).all():
        raise ValueError("translation_m must contain only finite values")


def _normalize_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    """Retain the final finite truth row for each normalized timestamp."""

    rows = _ORIGINAL_NORMALIZE_TRUTH_ROWS(truth)
    return rows.drop_duplicates(["sequence_id", "time_s"], keep="last").reset_index(
        drop=True
    )


def _normalized_flight_scope_value(value: object) -> str | None:
    """Normalize one optional physical-flight identifier for internal scoping."""

    if value is None or value is pd.NA or np.ma.is_masked(value):
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        return None
    return text


def _sequence_column(frame: pd.DataFrame) -> str | None:
    """Return the source-calibration sequence column before legacy normalization."""

    if "sequence_id" in frame.columns:
        return "sequence_id"
    if "Sequence" in frame.columns:
        return "Sequence"
    return None


def _physical_scope_keys(frame: pd.DataFrame) -> list[tuple[str, str | None]]:
    """Return joint sequence/flight identities for rows carrying ``flight_id``."""

    sequence_column = _sequence_column(frame)
    if sequence_column is None:
        raise ValueError("source calibration requires sequence_id metadata for flight scoping")
    return [
        (str(sequence_id), _normalized_flight_scope_value(flight_id))
        for sequence_id, flight_id in zip(
            frame[sequence_column],
            frame["flight_id"],
            strict=True,
        )
    ]


def _scope_token_map(
    *scope_lists: list[tuple[str, str | None]],
) -> dict[tuple[str, str | None], str]:
    """Create collision-safe temporary sequence ids for physical scopes."""

    tokens: dict[tuple[str, str | None], str] = {}
    for scopes in scope_lists:
        for scope in scopes:
            if scope not in tokens:
                tokens[scope] = f"__raft_source_calibration_scope_{len(tokens)}__"
    return tokens


def _has_multiple_flights_per_sequence(frame: pd.DataFrame) -> bool:
    """Return whether explicit flight metadata disambiguates one sequence."""

    if "flight_id" not in frame.columns or frame.empty:
        return False
    by_sequence: dict[str, set[str | None]] = {}
    for sequence_id, flight_id in _physical_scope_keys(frame):
        by_sequence.setdefault(sequence_id, set()).add(flight_id)
    return any(len(flight_ids) > 1 for flight_ids in by_sequence.values())


def build_source_calibration_pairs(
    candidates: object,
    truth: pd.DataFrame,
    *,
    max_truth_time_delta_s: float,
    max_pair_distance_m: float,
) -> pd.DataFrame:
    """Pair calibration rows without crossing physical ``flight_id`` boundaries."""

    candidate_rows = candidates.rows.copy()
    truth_rows = pd.DataFrame(truth).copy()
    candidate_has_flight = "flight_id" in candidate_rows.columns
    truth_has_flight = "flight_id" in truth_rows.columns

    if candidate_has_flight != truth_has_flight:
        scoped_side = candidate_rows if candidate_has_flight else truth_rows
        if _has_multiple_flights_per_sequence(scoped_side):
            raise ValueError(
                "source calibration cannot disambiguate multiple flight_id values under "
                "one sequence_id unless candidates and truth both provide flight_id"
            )
        return _ORIGINAL_BUILD_SOURCE_CALIBRATION_PAIRS(
            candidates,
            truth,
            max_truth_time_delta_s=max_truth_time_delta_s,
            max_pair_distance_m=max_pair_distance_m,
        )

    if not candidate_has_flight:
        return _ORIGINAL_BUILD_SOURCE_CALIBRATION_PAIRS(
            candidates,
            truth,
            max_truth_time_delta_s=max_truth_time_delta_s,
            max_pair_distance_m=max_pair_distance_m,
        )

    candidate_scopes = _physical_scope_keys(candidate_rows)
    truth_scopes = _physical_scope_keys(truth_rows)
    token_by_scope = _scope_token_map(candidate_scopes, truth_scopes)
    sequence_by_token = {
        token: scope[0]
        for scope, token in token_by_scope.items()
    }

    scoped_candidates = candidate_rows.copy()
    scoped_candidates["sequence_id"] = [token_by_scope[scope] for scope in candidate_scopes]
    scoped_truth = truth_rows.copy()
    scoped_truth["sequence_id"] = [token_by_scope[scope] for scope in truth_scopes]

    pairs = _ORIGINAL_BUILD_SOURCE_CALIBRATION_PAIRS(
        _IMPL.CandidateFrame(scoped_candidates),
        scoped_truth,
        max_truth_time_delta_s=max_truth_time_delta_s,
        max_pair_distance_m=max_pair_distance_m,
    )
    if pairs.empty:
        return pairs

    pairs = pairs.copy()
    pairs["sequence_id"] = pairs["sequence_id"].map(sequence_by_token)
    sort_columns = ["sequence_id", "flight_id", "source", "time_s"]
    return pairs.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _fit_source_translation_alpha_cv(
    group: pd.DataFrame,
    alpha_grid: tuple[float, ...],
) -> dict[str, object]:
    """Cross-validate source translation over physical flights when available."""

    if "flight_id" not in group.columns or group.empty:
        return _ORIGINAL_FIT_SOURCE_TRANSLATION_ALPHA_CV(group, alpha_grid)

    scopes = _physical_scope_keys(group)
    if len(set(scopes)) <= 1:
        return _ORIGINAL_FIT_SOURCE_TRANSLATION_ALPHA_CV(group, alpha_grid)

    token_by_scope = _scope_token_map(scopes)
    scoped_group = group.copy()
    scoped_group["sequence_id"] = [token_by_scope[scope] for scope in scopes]
    result = dict(_ORIGINAL_FIT_SOURCE_TRANSLATION_ALPHA_CV(scoped_group, alpha_grid))
    result["source_translation_alpha_cv_scope"] = "sequence_id+flight_id"
    return result


def _source_lookup_key(value: object) -> str:
    """Return the case-insensitive key used for source-transform lookup."""

    return str(value).casefold()


def _require_unambiguous_source_transform_keys(
    transforms: dict[str, object],
) -> None:
    """Reject blank keys and keys that collapse under case-insensitive lookup."""

    keys_by_normalized_value: dict[str, list[str]] = {}
    for key in transforms:
        rendered = str(key)
        if not rendered.strip():
            raise ValueError("source-calibration transforms must use non-blank source keys")
        keys_by_normalized_value.setdefault(_source_lookup_key(rendered), []).append(rendered)

    collisions = [
        sorted(keys)
        for keys in keys_by_normalized_value.values()
        if len(keys) > 1
    ]
    if not collisions:
        return

    rendered_collisions = "; ".join(
        ", ".join(repr(key) for key in keys)
        for keys in sorted(collisions)
    )
    raise ValueError(
        "source-calibration transforms contain ambiguous case-insensitive keys: "
        f"{rendered_collisions}"
    )


def _is_forward_source_prefix(source_key: str, transform_key: str) -> bool:
    """Return whether ``transform_key`` is a token-boundary prefix of ``source_key``."""

    if not transform_key or source_key == transform_key:
        return False
    if not source_key.startswith(transform_key):
        return False
    boundary_index = len(transform_key)
    return (not transform_key[-1].isalnum()) or (
        not source_key[boundary_index].isalnum()
    )


def _match_source_transform(source: str, transforms: dict[str, object]) -> object | None:
    """Return an exact or longest safe forward-prefix transform for one source.

    A calibration key may match a more specific exported source name at a token
    boundary, for example ``sensor_detail`` matching ``sensor_detail_clusters``.
    It must not match an unrelated alphanumeric continuation such as ``radar2``,
    and the reverse direction remains unsafe: a transform fitted specifically for
    ``sensor_detail`` must not be applied to the broader ``sensor`` source.
    """

    _require_unambiguous_source_transform_keys(transforms)
    source_key = _source_lookup_key(source)
    normalized_transforms = [
        (_source_lookup_key(key), transform)
        for key, transform in transforms.items()
    ]
    for transform_key, transform in normalized_transforms:
        if source_key == transform_key:
            return transform
    matches = [
        (len(transform_key), transform)
        for transform_key, transform in normalized_transforms
        if _is_forward_source_prefix(source_key, transform_key)
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


_IMPL.SourceTransform.__post_init__ = _validated_source_transform_post_init
_IMPL._normalize_truth_rows = _normalize_truth_rows
_IMPL.build_source_calibration_pairs = build_source_calibration_pairs
_IMPL._fit_source_translation_alpha_cv = _fit_source_translation_alpha_cv
_IMPL._match_source_transform = _match_source_transform

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_truth_rows"] = _normalize_truth_rows
globals()["_normalized_flight_scope_value"] = _normalized_flight_scope_value
globals()["_sequence_column"] = _sequence_column
globals()["_physical_scope_keys"] = _physical_scope_keys
globals()["_scope_token_map"] = _scope_token_map
globals()["_has_multiple_flights_per_sequence"] = _has_multiple_flights_per_sequence
globals()["build_source_calibration_pairs"] = build_source_calibration_pairs
globals()["_fit_source_translation_alpha_cv"] = _fit_source_translation_alpha_cv
globals()["_source_lookup_key"] = _source_lookup_key
globals()["_require_unambiguous_source_transform_keys"] = (
    _require_unambiguous_source_transform_keys
)
globals()["_is_forward_source_prefix"] = _is_forward_source_prefix
globals()["_match_source_transform"] = _match_source_transform

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
