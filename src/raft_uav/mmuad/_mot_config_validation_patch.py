"""Reject malformed explicit multi-object tracker configurations.

The startup hook also canonicalizes equal-confidence detections before the MOT
tracker assigns persistent output identities. This makes exact optimal-assignment
ties independent of DataFrame row order. Pooled tracker and metric calls are
scoped by every available physical-flight alias so reused local identifiers do
not leak state or identity bookkeeping across independent flights.
"""

from __future__ import annotations

from functools import wraps
import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

_PATCH_MARKER = "_raft_uav_validates_explicit_mot_config"
_CONFIG_POST_INIT_MARKER = "_raft_uav_validates_nested_mot_config_scalars"
_METRICS_SCOPE_MARKER = "_raft_uav_scopes_mot_metrics_by_flight"
_ORDER_HELPER_PREFIX = "_raft_uav_mot_order_"
_SCOPE_ALIASES = ("sequence_id", "flight_id")
_CONFIG_SCALAR_FIELDS = (
    "acceleration_std_mps2",
    "max_association_distance_m",
    "max_track_age_s",
    "min_new_track_confidence",
    "covariance_scale",
)


def _normalize_config_scalar_inputs(config: Any) -> None:
    """Normalize finite real controls without accepting recursively boxed Booleans."""

    for field_name in _CONFIG_SCALAR_FIELDS:
        number = optional_float(getattr(config, field_name))
        if number is None:
            raise ValueError(f"{field_name} must be finite")
        object.__setattr__(config, field_name, number)


def _install_config_scalar_guard(mot: Any) -> None:
    """Validate raw dataclass inputs before the legacy post-init coerces them."""

    config_type = mot.MultiObjectTrackerConfig
    original_post_init: Callable[..., Any] = config_type.__post_init__
    if getattr(original_post_init, _CONFIG_POST_INIT_MARKER, False):
        return

    @wraps(original_post_init)
    def validated_post_init(config: Any) -> None:
        _normalize_config_scalar_inputs(config)
        original_post_init(config)

    setattr(validated_post_init, _CONFIG_POST_INIT_MARKER, True)
    setattr(config_type, "__post_init__", validated_post_init)


def _finite_real_order_values(values: pd.Series) -> pd.Series:
    """Return finite real sort keys without changing the candidate payload."""

    parsed: list[float] = []
    for value in values.tolist():
        try:
            if np.ma.is_masked(value):
                raise TypeError
            array = np.asarray(value)
            if array.ndim != 0 or np.iscomplexobj(array):
                raise TypeError
            number = float(array.item())
        except (TypeError, ValueError, OverflowError):
            number = float("inf")
        parsed.append(number if np.isfinite(number) else float("inf"))
    return pd.Series(parsed, index=values.index, dtype=float)


def _text_order_values(values: pd.Series) -> pd.Series:
    """Return stable text fallback keys for optional candidate metadata."""

    return values.where(values.notna(), "").astype(str).str.strip()


def _ordered_candidate_frame(candidates: Any, mot: Any) -> Any:
    """Canonicalize within-frame detection order for deterministic MOT ties."""

    rows = getattr(candidates, "rows", None)
    if not isinstance(rows, pd.DataFrame) or rows.empty:
        return candidates

    ordered = rows.copy().reset_index(drop=True)
    sort_keys = pd.DataFrame(index=ordered.index)
    sort_columns: list[str] = []
    ascending: list[bool] = []

    for name in _SCOPE_ALIASES:
        key = f"{_ORDER_HELPER_PREFIX}{name}"
        values = (
            ordered[name]
            if name in ordered.columns
            else pd.Series("", index=ordered.index)
        )
        sort_keys[key] = _text_order_values(values)
        sort_columns.append(key)
        ascending.append(True)

    for name in ("time_s",):
        key = f"{_ORDER_HELPER_PREFIX}{name}"
        values = (
            ordered[name]
            if name in ordered.columns
            else pd.Series(np.nan, index=ordered.index)
        )
        sort_keys[key] = _finite_real_order_values(values)
        sort_columns.append(key)
        ascending.append(True)

    confidence_key = f"{_ORDER_HELPER_PREFIX}confidence"
    sort_keys[confidence_key] = mot._mot_confidence_values(ordered)
    sort_columns.append(confidence_key)
    ascending.append(False)

    for name in ("x_m", "y_m", "z_m"):
        key = f"{_ORDER_HELPER_PREFIX}{name}"
        values = (
            ordered[name]
            if name in ordered.columns
            else pd.Series(np.nan, index=ordered.index)
        )
        sort_keys[key] = _finite_real_order_values(values)
        sort_columns.append(key)
        ascending.append(True)

    for name in ("source", "track_id", "class_name"):
        key = f"{_ORDER_HELPER_PREFIX}{name}"
        values = (
            ordered[name]
            if name in ordered.columns
            else pd.Series("", index=ordered.index)
        )
        sort_keys[key] = _text_order_values(values)
        sort_columns.append(key)
        ascending.append(True)

    row_order = sort_keys.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).index.to_numpy()
    return mot.CandidateFrame(ordered.iloc[row_order].reset_index(drop=True))


def _scope_scalar(value: Any) -> str | None:
    """Normalize one scope identifier without conflating missing and text values."""

    if value is None or np.ma.is_masked(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "<na>", "nat"}:
        return None
    return text


def _flight_scope_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the physical-scope aliases carried by a frame."""

    return tuple(alias for alias in _SCOPE_ALIASES if alias in frame.columns)


def _scope_tokens(
    frame: pd.DataFrame,
    *,
    scope_columns: tuple[str, ...],
) -> pd.Series:
    """Return collision-safe text tokens for joint physical-flight scope."""

    values = [
        [
            _scope_scalar(frame.iloc[row_index][column])
            for column in scope_columns
        ]
        for row_index in range(len(frame))
    ]
    tokens = [
        "__raft_uav_mot_scope__"
        + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        for value in values
    ]
    return pd.Series(tokens, index=frame.index, dtype=object)


def _scope_display_key(
    *,
    scope_columns: tuple[str, ...],
    values: tuple[str | None, ...],
) -> str:
    """Return a deterministic human-readable key for per-scope metrics."""

    if scope_columns == ("sequence_id",):
        return "" if values[0] is None else str(values[0])
    payload = {column: value for column, value in zip(scope_columns, values, strict=True)}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _scoped_frame(
    frame: pd.DataFrame,
    *,
    scope_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, tuple[str | None, ...]]]:
    """Replace sequence_id by a collision-safe joint scope token."""

    scoped = frame.copy()
    tokens = _scope_tokens(scoped, scope_columns=scope_columns)
    mapping: dict[str, tuple[str | None, ...]] = {}
    for row_index, token in enumerate(tokens.tolist()):
        values = tuple(
            _scope_scalar(scoped.iloc[row_index][column])
            for column in scope_columns
        )
        existing = mapping.get(token)
        if existing is not None and existing != values:
            raise ValueError("MOT physical-flight scope token collision")
        mapping[token] = values
    scoped["sequence_id"] = tokens
    return scoped, mapping


def _shared_scope_columns(
    estimates: pd.DataFrame,
    truth: pd.DataFrame | None,
) -> tuple[str, ...]:
    """Return matching scope aliases or fail closed on one-sided flight metadata."""

    estimate_columns = _flight_scope_columns(estimates)
    if truth is None or truth.empty:
        return estimate_columns
    truth_columns = _flight_scope_columns(truth)
    if estimate_columns != truth_columns:
        raise ValueError(
            "MOT estimates and truth must carry the same sequence/flight scope aliases"
        )
    return estimate_columns


def _scoped_metrics_inputs(
    estimates: pd.DataFrame,
    truth: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Scope public MOT metric inputs by every available flight alias."""

    scope_columns = _shared_scope_columns(estimates, truth)
    if "flight_id" not in scope_columns:
        return estimates, truth
    scoped_estimates, _ = _scoped_frame(estimates, scope_columns=scope_columns)
    if truth is None:
        return scoped_estimates, None
    scoped_truth, _ = _scoped_frame(truth, scope_columns=scope_columns)
    return scoped_estimates, scoped_truth


def _restore_tracker_frame(
    frame: pd.DataFrame,
    *,
    scope_columns: tuple[str, ...],
    mapping: dict[str, tuple[str | None, ...]],
) -> pd.DataFrame:
    """Restore original scope metadata on a tracker output frame."""

    if frame.empty or "sequence_id" not in frame.columns:
        return frame
    restored = frame.copy()
    rows = [mapping[str(token)] for token in restored["sequence_id"].tolist()]
    for column_index, column in enumerate(scope_columns):
        restored[column] = [values[column_index] for values in rows]
    return restored


def _remap_sequence_metrics(
    metrics: dict[str, Any],
    *,
    scope_columns: tuple[str, ...],
    mapping: dict[str, tuple[str | None, ...]],
) -> dict[str, Any]:
    """Restore readable joint-scope labels in the sequence metric dictionary."""

    sequences = metrics.get("sequences")
    if not isinstance(sequences, dict):
        return metrics
    remapped = dict(metrics)
    remapped["sequences"] = {
        _scope_display_key(scope_columns=scope_columns, values=mapping[str(token)]): value
        for token, value in sequences.items()
    }
    return remapped


def _install_metric_scope_guard(mot: Any) -> None:
    """Scope direct public MOT metric calls by sequence and flight aliases."""

    original: Callable[..., Any] = mot.compute_multi_object_metrics
    if getattr(original, _METRICS_SCOPE_MARKER, False):
        return

    @wraps(original)
    def scoped_metrics(
        estimates: pd.DataFrame,
        truth: pd.DataFrame | None,
        *,
        match_distance_m: float = 25.0,
    ) -> dict[str, Any]:
        scoped_estimates, scoped_truth = _scoped_metrics_inputs(estimates, truth)
        return original(
            scoped_estimates,
            scoped_truth,
            match_distance_m=match_distance_m,
        )

    setattr(scoped_metrics, _METRICS_SCOPE_MARKER, True)
    setattr(mot, "compute_multi_object_metrics", scoped_metrics)


def install() -> None:
    """Install strict config validation, deterministic ordering, and flight scoping."""

    import raft_uav.mmuad as mmuad
    from raft_uav.mmuad import mot

    _install_config_scalar_guard(mot)
    _install_metric_scope_guard(mot)

    original: Callable[..., Any] = mot.run_mmuad_multi_object_tracker
    if getattr(original, _PATCH_MARKER, False):
        setattr(mmuad, "run_mmuad_multi_object_tracker", original)
        return

    @wraps(original)
    def validated(
        candidates: Any,
        truth: Any = None,
        *,
        config: Any = None,
    ) -> Any:
        if config is not None and not isinstance(config, mot.MultiObjectTrackerConfig):
            raise TypeError("config must be a MultiObjectTrackerConfig or None")
        ordered_candidates = _ordered_candidate_frame(candidates, mot)
        candidate_rows = getattr(ordered_candidates, "rows", None)
        if not isinstance(candidate_rows, pd.DataFrame):
            return original(ordered_candidates, truth, config=config)

        truth_rows = getattr(truth, "rows", None) if truth is not None else None
        scope_columns = _shared_scope_columns(candidate_rows, truth_rows)
        if "flight_id" not in scope_columns:
            return original(ordered_candidates, truth, config=config)

        scoped_candidate_rows, mapping = _scoped_frame(
            candidate_rows,
            scope_columns=scope_columns,
        )
        scoped_candidates = mot.CandidateFrame(scoped_candidate_rows)
        scoped_truth = None
        if truth is not None:
            assert isinstance(truth_rows, pd.DataFrame)
            scoped_truth_rows, truth_mapping = _scoped_frame(
                truth_rows,
                scope_columns=scope_columns,
            )
            if set(truth_mapping) - set(mapping):
                mapping = {**mapping, **truth_mapping}
            scoped_truth = mot.TruthFrame(scoped_truth_rows)

        result = original(scoped_candidates, scoped_truth, config=config)
        estimates = _restore_tracker_frame(
            result.estimates,
            scope_columns=scope_columns,
            mapping=mapping,
        )
        selected = _restore_tracker_frame(
            result.selected_tracklets,
            scope_columns=scope_columns,
            mapping=mapping,
        )
        metrics = _remap_sequence_metrics(
            result.metrics,
            scope_columns=scope_columns,
            mapping=mapping,
        )
        return mot.TrackerOutput(estimates, metrics, selected)

    setattr(validated, _PATCH_MARKER, True)
    setattr(mot, "run_mmuad_multi_object_tracker", validated)
    setattr(mmuad, "run_mmuad_multi_object_tracker", validated)
