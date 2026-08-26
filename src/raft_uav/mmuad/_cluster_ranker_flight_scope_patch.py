"""Keep cluster-ranker truth supervision inside physical flight scopes."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad import _mot_config_validation_patch as scope_patch
from raft_uav.mmuad import _mot_scope_compat_patch as scope_compat

_LABEL_PATCH_MARKER = "_raft_uav_scopes_cluster_ranker_truth_by_flight"
_SELECTION_PATCH_MARKER = "_raft_uav_scopes_cluster_ranker_frames_by_flight"
_ROW_POSITION = "__raft_uav_cluster_ranker_row_position"


def _truth_frame(cluster_ranker: Any, truth: Any) -> pd.DataFrame:
    """Return an in-memory truth table without discarding physical-scope columns."""

    if isinstance(truth, Path):
        return cluster_ranker.load_evaluation_truth_file(truth).rows.copy()
    return pd.DataFrame(truth).copy()


def _scoped_copy(frame: pd.DataFrame, *, name: str) -> tuple[pd.DataFrame, bool]:
    """Normalize complete flight metadata and reject partially populated scopes."""

    rows = pd.DataFrame(frame).copy()
    if rows.empty or "flight_id" not in rows.columns:
        return rows, False

    normalized = pd.Series(
        [scope_patch._scope_scalar(value) for value in rows["flight_id"].tolist()],
        index=rows.index,
        dtype=object,
    )
    present = normalized.notna()
    if not bool(present.any()):
        return rows.drop(columns="flight_id"), False
    if not bool(present.all()):
        missing_positions = [
            position for position, is_present in enumerate(present.tolist()) if not is_present
        ]
        preview = ", ".join(str(position) for position in missing_positions[:8])
        if len(missing_positions) > 8:
            preview = f"{preview}, ..."
        raise ValueError(
            f"{name} contains partially missing flight_id values at row positions "
            f"[{preview}]; flight_id metadata must be complete or absent"
        )
    rows["flight_id"] = normalized
    return rows, True


def _helper_column(*frames: pd.DataFrame) -> str:
    """Return a temporary row-position column that cannot overwrite user data."""

    column = _ROW_POSITION
    while any(column in frame.columns for frame in frames):
        column += "_"
    return column


def _scoped_truth_labels(
    previous: Callable[..., pd.DataFrame],
    cluster_ranker: Any,
    features: pd.DataFrame,
    truth: Any,
    *,
    good_threshold_m: float,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    feature_rows = pd.DataFrame(features).copy()
    truth_rows = _truth_frame(cluster_ranker, truth)
    if feature_rows.empty or truth_rows.empty:
        return previous(
            feature_rows,
            truth_rows,
            good_threshold_m=good_threshold_m,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )

    scoped_features, features_have_flight = _scoped_copy(
        feature_rows,
        name="cluster features",
    )
    scoped_truth, truth_has_flight = _scoped_copy(
        truth_rows,
        name="cluster-ranker truth",
    )
    scope_compat._validate_two_sided_flight_scope(
        scoped_features,
        scoped_truth,
        left_name="cluster features",
        right_name="cluster-ranker truth",
    )
    if not (features_have_flight and truth_has_flight):
        return previous(
            feature_rows,
            truth_rows,
            good_threshold_m=good_threshold_m,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )

    scope_columns = ("sequence_id", "flight_id")
    feature_tokens = scope_patch._scope_tokens(
        scoped_features,
        scope_columns=scope_columns,
    )
    truth_tokens = scope_patch._scope_tokens(
        scoped_truth,
        scope_columns=scope_columns,
    )
    ordered_tokens = list(dict.fromkeys(str(token) for token in feature_tokens))
    row_position = _helper_column(feature_rows, truth_rows)
    original_index = feature_rows.index.copy()
    parts: list[pd.DataFrame] = []

    for token in ordered_tokens:
        feature_mask = (feature_tokens == token).to_numpy()
        truth_mask = (truth_tokens == token).to_numpy()
        feature_positions = feature_mask.nonzero()[0]
        truth_positions = truth_mask.nonzero()[0]
        scoped_input = feature_rows.iloc[feature_positions].copy()
        scoped_input[row_position] = feature_positions
        parts.append(
            previous(
                scoped_input,
                truth_rows.iloc[truth_positions].copy(),
                good_threshold_m=good_threshold_m,
                max_truth_time_delta_s=max_truth_time_delta_s,
            )
        )

    combined = (
        pd.concat(parts, ignore_index=True, sort=False)
        .sort_values(row_position, kind="stable")
        .drop(columns=row_position)
    )
    combined.index = original_index
    return combined


def _scoped_frame_selection_rows(
    previous: Callable[[pd.DataFrame], pd.DataFrame],
    rows: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return previous(frame)
    scoped, has_flight = _scoped_copy(frame, name="cluster-ranker diagnostics")
    if not has_flight:
        return previous(frame)

    scope_columns = ("sequence_id", "flight_id")
    tokens = scope_patch._scope_tokens(scoped, scope_columns=scope_columns)
    mappings = scope_patch._scope_mapping(scoped, scope_columns=scope_columns)
    parts: list[pd.DataFrame] = []
    for token in dict.fromkeys(str(value) for value in tokens):
        positions = (tokens == token).to_numpy().nonzero()[0]
        part = previous(frame.iloc[positions].copy())
        if part.empty:
            continue
        _, flight_id = mappings[token]
        part.insert(1, "flight_id", flight_id)
        parts.append(part)
    if not parts:
        empty = previous(frame.iloc[0:0].copy())
        empty.insert(1, "flight_id", pd.Series(dtype=object))
        return empty
    return (
        pd.concat(parts, ignore_index=True, sort=False)
        .sort_values(["sequence_id", "flight_id", "time_s"], kind="mergesort")
        .reset_index(drop=True)
    )


def install() -> None:
    """Install flight-scoped truth labeling and frame diagnostics."""

    from raft_uav.mmuad import cluster_ranker

    previous_label = cluster_ranker.label_cluster_features_against_truth
    if not getattr(previous_label, _LABEL_PATCH_MARKER, False):

        @wraps(previous_label)
        def scoped_label_cluster_features_against_truth(
            features: pd.DataFrame,
            truth: Any,
            *,
            good_threshold_m: float = 5.0,
            max_truth_time_delta_s: float = 0.5,
        ) -> pd.DataFrame:
            return _scoped_truth_labels(
                previous_label,
                cluster_ranker,
                features,
                truth,
                good_threshold_m=good_threshold_m,
                max_truth_time_delta_s=max_truth_time_delta_s,
            )

        setattr(scoped_label_cluster_features_against_truth, _LABEL_PATCH_MARKER, True)
        setattr(scoped_label_cluster_features_against_truth, "_raft_uav_previous", previous_label)
        cluster_ranker.label_cluster_features_against_truth = (
            scoped_label_cluster_features_against_truth
        )
        cluster_ranker._IMPL.label_cluster_features_against_truth = (
            scoped_label_cluster_features_against_truth
        )

    previous_selection = cluster_ranker._ranker_frame_selection_rows
    if not getattr(previous_selection, _SELECTION_PATCH_MARKER, False):

        @wraps(previous_selection)
        def scoped_ranker_frame_selection_rows(rows: pd.DataFrame) -> pd.DataFrame:
            return _scoped_frame_selection_rows(previous_selection, rows)

        setattr(scoped_ranker_frame_selection_rows, _SELECTION_PATCH_MARKER, True)
        setattr(scoped_ranker_frame_selection_rows, "_raft_uav_previous", previous_selection)
        cluster_ranker._ranker_frame_selection_rows = scoped_ranker_frame_selection_rows
        cluster_ranker._IMPL._ranker_frame_selection_rows = (
            scoped_ranker_frame_selection_rows
        )
