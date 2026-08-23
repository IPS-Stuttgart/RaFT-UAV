"""Reject malformed explicit multi-object tracker configurations.

The startup hook also canonicalizes equal-confidence detections before the MOT
tracker assigns persistent output identities. This makes exact optimal-assignment
ties independent of DataFrame row order. Multi-object tracking and metrics are
also isolated by physical sequence/flight scope so local track identifiers do
not leak state or identity bookkeeping across independent recordings.
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
_METRICS_SCOPE_MARKER = "_raft_uav_scopes_mot_metrics_by_physical_recording"
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


def _scope_scalar(value: Any) -> Any:
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


def _nonempty_frame(frame: Any) -> bool:
    return isinstance(frame, pd.DataFrame) and not frame.empty


def _available_scope_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return physical identity aliases carried by this frame."""

    return tuple(alias for alias in _SCOPE_ALIASES if alias in frame.columns)


def _scope_tokens(
    frame: pd.DataFrame,
    *,
    scope_columns: tuple[str, ...],
) -> pd.Series:
    """Return collision-safe deterministic tokens for joint physical scope."""

    if frame.empty:
        return pd.Series(index=frame.index, dtype=object)
    values = [
        [_scope_scalar(value) for value in row]
        for row in frame.loc[:, list(scope_columns)].itertuples(index=False, name=None)
    ]
    tokens = [
        "__raft_uav_mot_scope__"
        + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        for value in values
    ]
    return pd.Series(tokens, index=frame.index, dtype=object)


def _scope_mapping(
    frame: pd.DataFrame,
    *,
    scope_columns: tuple[str, ...],
) -> dict[str, tuple[Any, ...]]:
    """Map generated scope tokens back to readable normalized metadata."""

    mapping: dict[str, tuple[Any, ...]] = {}
    if frame.empty:
        return mapping
    tokens = _scope_tokens(frame, scope_columns=scope_columns).tolist()
    rows = frame.loc[:, list(scope_columns)].itertuples(index=False, name=None)
    for token, row in zip(tokens, rows):
        values = tuple(_scope_scalar(value) for value in row)
        previous = mapping.get(str(token))
        if previous is not None and previous != values:
            raise ValueError("MOT physical scope token collision")
        mapping[str(token)] = values
    return mapping


def _validate_two_sided_flight_scope(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Reject ambiguous pooled calls only when both sides contain observations."""

    if not _nonempty_frame(left) or not _nonempty_frame(right):
        return
    assert right is not None
    left_has_flight = "flight_id" in left.columns
    right_has_flight = "flight_id" in right.columns
    if left_has_flight != right_has_flight:
        raise ValueError(
            f"{left_name} and {right_name} must both carry flight_id when both are nonempty"
        )


def _pairing_scope_columns(
    estimates: pd.DataFrame,
    truth: pd.DataFrame | None,
) -> tuple[str, ...]:
    """Return shared aliases needed to keep same-time frames physically separate."""

    if not _nonempty_frame(estimates) or not _nonempty_frame(truth):
        return ()
    assert truth is not None
    if "flight_id" not in estimates.columns or "flight_id" not in truth.columns:
        return ()
    if "sequence_id" in estimates.columns and "sequence_id" in truth.columns:
        return ("sequence_id", "flight_id")
    return ("flight_id",)


def _namespace_metric_identity(
    frame: pd.DataFrame,
    *,
    id_column: str,
    mot: Any,
) -> pd.DataFrame:
    """Namespace a metric identity by its recording scope without altering missing IDs."""

    scope_columns = _available_scope_columns(frame)
    if frame.empty or id_column not in frame.columns or not scope_columns:
        return frame
    namespaced = frame.copy()
    values = namespaced[id_column].astype(object).to_numpy(copy=True)
    present = mot._track_id_present_mask(namespaced[id_column]).to_numpy(dtype=bool)
    tokens = _scope_tokens(namespaced, scope_columns=scope_columns).to_numpy(dtype=object)
    for position in np.flatnonzero(present):
        values[position] = f"{tokens[position]}::track::{values[position]}"
    namespaced[id_column] = pd.Series(values, index=namespaced.index, dtype=object)
    return namespaced


def _scoped_metric_inputs(
    estimates: pd.DataFrame,
    truth: pd.DataFrame | None,
    mot: Any,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Isolate frame matching and track identities across independent recordings."""

    scoped_estimates = estimates.copy()
    scoped_truth = truth.copy() if truth is not None else None
    _validate_two_sided_flight_scope(
        scoped_estimates,
        scoped_truth,
        left_name="MOT estimates",
        right_name="truth",
    )

    if "output_track_id" in scoped_estimates.columns:
        scoped_estimates = _namespace_metric_identity(
            scoped_estimates,
            id_column="output_track_id",
            mot=mot,
        )
    elif "track_id" in scoped_estimates.columns:
        scoped_estimates = _namespace_metric_identity(
            scoped_estimates,
            id_column="track_id",
            mot=mot,
        )
    if scoped_truth is not None and "track_id" in scoped_truth.columns:
        scoped_truth = _namespace_metric_identity(
            scoped_truth,
            id_column="track_id",
            mot=mot,
        )

    pairing_columns = _pairing_scope_columns(scoped_estimates, scoped_truth)
    if pairing_columns:
        scoped_estimates["sequence_id"] = _scope_tokens(
            scoped_estimates,
            scope_columns=pairing_columns,
        )
        assert scoped_truth is not None
        scoped_truth["sequence_id"] = _scope_tokens(
            scoped_truth,
            scope_columns=pairing_columns,
        )
    return scoped_estimates, scoped_truth


def _install_metric_scope_guard(mot: Any) -> None:
    """Scope direct public MOT metric calls by recording identity."""

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
        scoped_estimates, scoped_truth = _scoped_metric_inputs(estimates, truth, mot)
        return original(
            scoped_estimates,
            scoped_truth,
            match_distance_m=match_distance_m,
        )

    setattr(scoped_metrics, _METRICS_SCOPE_MARKER, True)
    setattr(mot, "compute_multi_object_metrics", scoped_metrics)


def _tracker_scope_columns(
    candidate_rows: pd.DataFrame,
    truth_rows: pd.DataFrame | None,
) -> tuple[str, ...]:
    """Choose the joint scope used to partition persistent tracker state."""

    candidate_has_flight = _nonempty_frame(candidate_rows) and "flight_id" in candidate_rows.columns
    truth_has_flight = _nonempty_frame(truth_rows) and "flight_id" in truth_rows.columns
    if not candidate_has_flight and not truth_has_flight:
        return ()
    reference = candidate_rows if candidate_has_flight else truth_rows
    assert isinstance(reference, pd.DataFrame)
    columns = [alias for alias in _SCOPE_ALIASES if alias in reference.columns]
    if "flight_id" not in columns:
        return ()
    return tuple(columns)


def _tracker_scoped_frame(
    frame: pd.DataFrame,
    *,
    scope_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Encode joint physical scope into sequence_id for the legacy tracker core."""

    scoped = frame.copy()
    if scoped.empty:
        if "sequence_id" not in scoped.columns:
            scoped["sequence_id"] = pd.Series(index=scoped.index, dtype=object)
        return scoped.drop(columns=["flight_id"], errors="ignore")
    missing = [column for column in scope_columns if column not in scoped.columns]
    if missing:
        raise ValueError(
            "MOT candidates and truth must carry the same physical scope aliases"
        )
    scoped["sequence_id"] = _scope_tokens(scoped, scope_columns=scope_columns)
    return scoped.drop(columns=["flight_id"], errors="ignore")


def _restore_tracker_frame(
    frame: pd.DataFrame,
    *,
    scope_columns: tuple[str, ...],
    mapping: dict[str, tuple[Any, ...]],
) -> pd.DataFrame:
    """Restore readable physical scope metadata on tracker output rows."""

    if frame.empty or "sequence_id" not in frame.columns:
        return frame.copy()
    restored = frame.copy()
    rows: list[tuple[Any, ...]] = []
    for token in restored["sequence_id"].astype(str).tolist():
        if token not in mapping:
            raise ValueError("MOT tracker returned an unknown physical scope token")
        rows.append(mapping[token])
    for column_index, column in enumerate(scope_columns):
        restored[column] = [values[column_index] for values in rows]
    return restored


def _scope_display_key(
    *,
    scope_columns: tuple[str, ...],
    values: tuple[Any, ...],
) -> str:
    payload = {column: value for column, value in zip(scope_columns, values)}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _restore_sequence_metrics(
    metrics: dict[str, Any],
    *,
    scope_columns: tuple[str, ...],
    mapping: dict[str, tuple[Any, ...]],
) -> dict[str, Any]:
    """Replace opaque internal sequence tokens by readable joint-scope labels."""

    sequences = metrics.get("sequences")
    if not isinstance(sequences, dict):
        return metrics
    restored = dict(metrics)
    remapped: dict[str, Any] = {}
    for token, value in sequences.items():
        token_text = str(token)
        if token_text not in mapping:
            raise ValueError("MOT metrics returned an unknown physical scope token")
        key = _scope_display_key(
            scope_columns=scope_columns,
            values=mapping[token_text],
        )
        remapped[key] = value
    restored["sequences"] = remapped
    return restored


def install() -> None:
    """Install strict config validation, deterministic ordering, and scope isolation."""

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
        if truth_rows is not None and not isinstance(truth_rows, pd.DataFrame):
            return original(ordered_candidates, truth, config=config)
        _validate_two_sided_flight_scope(
            candidate_rows,
            truth_rows,
            left_name="MOT candidates",
            right_name="truth",
        )
        scope_columns = _tracker_scope_columns(candidate_rows, truth_rows)
        if not scope_columns:
            return original(ordered_candidates, truth, config=config)

        mapping = _scope_mapping(candidate_rows, scope_columns=scope_columns)
        if isinstance(truth_rows, pd.DataFrame) and not truth_rows.empty:
            mapping.update(_scope_mapping(truth_rows, scope_columns=scope_columns))

        scoped_candidate_rows = _tracker_scoped_frame(
            candidate_rows,
            scope_columns=scope_columns,
        )
        scoped_candidates = mot.CandidateFrame(scoped_candidate_rows)
        scoped_truth = None
        if truth is not None:
            assert isinstance(truth_rows, pd.DataFrame)
            scoped_truth_rows = _tracker_scoped_frame(
                truth_rows,
                scope_columns=scope_columns,
            )
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
        metrics = _restore_sequence_metrics(
            result.metrics,
            scope_columns=scope_columns,
            mapping=mapping,
        )
        return mot.TrackerOutput(estimates, metrics, selected)

    setattr(validated, _PATCH_MARKER, True)
    setattr(mot, "run_mmuad_multi_object_tracker", validated)
    setattr(mmuad, "run_mmuad_multi_object_tracker", validated)
