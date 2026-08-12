"""Restore compatibility contracts that overlapping runtime guards regressed.

The project intentionally layers narrow validation and compatibility patches over
maintained implementations.  A few of those guards were individually reasonable
but composed incorrectly.  This module is installed after the existing runtime
hooks and reconciles only the affected boundaries.
"""

from __future__ import annotations

from functools import wraps
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


def _patch_factor_graph_sequence_guard() -> None:
    guard = import_module("raft_uav.research._factor_graph_sequence_guard_patch")

    def _single_sequence_id(
        frame: pd.DataFrame | None,
        *,
        name: str,
    ) -> str | None:
        if frame is None or "sequence_id" not in frame.columns:
            return None
        identifiers = {
            identifier
            for identifier in (
                guard._normalized_sequence_id(value)
                for value in frame["sequence_id"]
            )
            if identifier is not None
        }
        if len(identifiers) > 1:
            raise ValueError(
                f"{name} contains multiple sequence_id values; "
                "factor-graph smoothing must be run separately for each sequence"
            )
        return next(iter(identifiers), None)

    guard._single_sequence_id = _single_sequence_id


def _patch_golden_metrics_schema_message() -> None:
    patch = import_module("raft_uav.evaluation._golden_metrics_schema_patch")
    golden_artifacts = import_module("raft_uav.evaluation.golden_artifacts")

    def _check_metrics(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            metrics = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            return [
                {
                    "check": "metrics_json_parse",
                    "file": str(path),
                    "passed": False,
                    "message": str(exc),
                }
            ]
        rows.append(
            {
                "check": "metrics_json_parse",
                "file": str(path),
                "passed": True,
                "message": "",
            }
        )
        is_object = isinstance(metrics, dict)
        rows.append(
            {
                "check": "metrics_json_object",
                "file": str(path),
                "passed": is_object,
                "message": "" if is_object else "metrics JSON must contain an object",
            }
        )
        if not is_object:
            return rows
        for key in ("posterior_records", "accepted_measurements", "position_error_3d"):
            rows.append(
                {
                    "check": "metrics_required_key",
                    "file": str(path),
                    "key": key,
                    "passed": key in metrics,
                    "message": "" if key in metrics else f"missing key {key}",
                }
            )
        return rows

    patch._check_metrics = _check_metrics
    golden_artifacts._check_metrics = _check_metrics


def _patch_split_manifest_overlap() -> None:
    splits = import_module("raft_uav.mmuad.splits")
    previous = splits.load_split_manifest

    def load_split_manifest(path: Path) -> dict[str, tuple[str, ...]]:
        path = Path(path)
        if path.suffix.lower() in {".json", ".yaml", ".yml"}:
            payload = splits._load_manifest_payload(path)
            return splits._manifest_from_payload(payload)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        manifest = splits._manifest_from_rows(frame.to_dict("records"))
        if not manifest:
            raise ValueError(
                "CSV split manifest must contain sequence id and split columns; "
                "accepted aliases include sequence_id/id/name and "
                "split/subset/partition"
            )
        return manifest

    splits.load_split_manifest = load_split_manifest
    implementation = getattr(splits, "_IMPL", None)
    if implementation is not None:
        implementation.load_split_manifest = load_split_manifest
    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", {}) if module is not None else {}
        if namespace.get("load_split_manifest") is previous:
            setattr(module, "load_split_manifest", load_split_manifest)


def _patch_candidate_protection_flags() -> None:
    diversity = import_module("raft_uav.mmuad.candidate_diversity")

    def _parse_protected_flag(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value is None or np.ma.is_masked(value):
            return False
        if isinstance(value, str):
            text = value.strip().lower()
            if text in diversity._TRUE_PROTECTED_TEXT:
                return True
            if text in diversity._FALSE_PROTECTED_TEXT:
                return False
            raise ValueError(
                "candidate_reservoir_protected values must be boolean-like; "
                f"got {value!r}"
            )
        try:
            scalar = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "candidate_reservoir_protected values must be boolean-like; "
                f"got {value!r}"
            ) from exc
        if scalar.ndim != 0:
            raise ValueError(
                "candidate_reservoir_protected values must be boolean-like; "
                f"got {value!r}"
            )
        item = scalar.item()
        if isinstance(item, (complex, np.complexfloating)):
            raise ValueError(
                "candidate_reservoir_protected values must be boolean-like; "
                f"got {value!r}"
            )
        try:
            missing = pd.isna(item)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return False
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "candidate_reservoir_protected values must be boolean-like; "
                f"got {value!r}"
            ) from exc
        return bool(number) if np.isfinite(number) else False

    diversity._parse_protected_flag = _parse_protected_flag
    implementation = getattr(diversity, "_IMPL", None)
    if implementation is not None:
        implementation._parse_protected_flag = _parse_protected_flag


def _patch_completion_default_sentinels() -> None:
    completion = import_module("raft_uav.mmuad.completion")

    def _validate_completion_sequence_metadata(value: object, *, name: str) -> None:
        values = completion._completion_sequence_values(value)
        if values is None or values.empty:
            return
        missing = values.isna()
        text = values.where(~missing, "").astype(str).str.strip().str.casefold()
        missing = missing | text.isin(completion._MISSING_SEQUENCE_ID_STRINGS)
        if not (bool(missing.any()) and bool((~missing).any())):
            return
        explicit = set(text.loc[~missing].tolist())
        if name == "completion template" and explicit == {"default"}:
            return
        raise ValueError(
            f"{name} sequence IDs are partially missing; provide sequence IDs "
            "for every row or omit them for the entire table"
        )

    completion._validate_completion_sequence_metadata = (
        _validate_completion_sequence_metadata
    )


def _finite_timestamp_scalar(value: Any) -> float | None:
    if value is None or np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        return None
    try:
        number = float(scalar.item())
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _patch_timestamp_alias_cycles() -> None:
    schema = import_module("raft_uav.mmuad.schema")

    def _stamp_dict_to_seconds(value: Any) -> float | None:
        def parse(candidate: Any, seen: set[int]) -> float | None:
            mapping = schema._coerce_stamp_mapping(candidate)
            if mapping is None:
                return None
            identity = id(mapping)
            if identity in seen:
                return None
            next_seen = {*seen, identity}

            nested = schema._mapping_get_case_insensitive(mapping, "stamp")
            if nested is not None:
                nested_time = parse(nested, next_seen)
                if nested_time is not None:
                    return nested_time

            seconds = schema._first_mapping_value_case_insensitive(
                mapping, ("sec", "secs", "seconds")
            )
            nanoseconds = schema._first_mapping_value_case_insensitive(
                mapping, ("nanosec", "nsec", "nsecs", "nanoseconds")
            )
            second_value = _finite_timestamp_scalar(seconds)
            if second_value is not None:
                if schema._is_json_missing_scalar(nanoseconds):
                    nanosecond_value = 0.0
                else:
                    nanosecond_value = _finite_timestamp_scalar(nanoseconds)
                    if nanosecond_value is None:
                        return None
                return second_value + nanosecond_value * 1.0e-9

            for alias, scale in schema._TIME_UNIT_ALIASES.items():
                scalar = schema._mapping_get_case_insensitive(mapping, alias)
                scalar_value = _finite_timestamp_scalar(scalar)
                if scalar_value is not None:
                    return scalar_value * scale

            scalar = schema._first_mapping_value_case_insensitive(
                mapping,
                ("time_s", "timestamp_s", "timestamp", "stamp", "time"),
            )
            return _finite_timestamp_scalar(scalar)

        return parse(value, set())

    def _combine_time_alias_series(
        candidates: Any,
    ) -> pd.Series | None:
        combined: pd.Series | None = None
        for candidate in candidates:
            series = pd.Series(candidate, copy=False)
            combined = series.copy() if combined is None else combined.fillna(series)
        return combined

    schema._stamp_dict_to_seconds = _stamp_dict_to_seconds
    schema._combine_time_alias_series = _combine_time_alias_series


def _patch_track5_probability_message() -> None:
    relabel = import_module("raft_uav.mmuad.track5_classification_relabel")
    original = relabel._validate_probability_column_schema

    def _validate_probability_column_schema(rows: pd.DataFrame) -> None:
        try:
            original(rows)
        except ValueError as exc:
            message = str(exc)
            if "probability columns outside official classes" in message:
                raise ValueError(
                    message.replace(
                        "probability columns outside official classes",
                        "official-class probability columns outside supported classes",
                    )
                ) from exc
            raise

    relabel._validate_probability_column_schema = (
        _validate_probability_column_schema
    )
    implementation = getattr(relabel, "_IMPL", None)
    if implementation is not None:
        implementation._validate_probability_column_schema = (
            _validate_probability_column_schema
        )


def _patch_paper_timestamp_cadence() -> None:
    paper = import_module("raft_uav.paper_selection")
    numeric_patch = import_module("raft_uav._paper_selection_numeric_time_patch")
    original_segmenter = paper._continuous_track_segments

    def _continuous_track_segments(radar: pd.DataFrame) -> list[pd.DataFrame]:
        normalized = numeric_patch._normalize_chronology(radar)
        sequence_ids = paper._explicit_sequence_ids(normalized)
        if len(sequence_ids) > 1:
            raise ValueError(
                "paper radar track selection requires one sequence_id; "
                "split pooled radar data by sequence"
            )
        if normalized.empty or "track_id" not in normalized.columns:
            return []

        segments: list[pd.DataFrame] = []
        for _, track_rows in normalized.groupby("track_id", sort=True):
            frame_index = (
                pd.to_numeric(track_rows["frame_index"], errors="coerce")
                if "frame_index" in track_rows.columns
                else None
            )
            use_frame_index = frame_index is not None and bool(
                np.isfinite(frame_index).all()
            )
            times = (
                pd.to_numeric(track_rows["time_s"], errors="coerce")
                if "time_s" in track_rows.columns
                else None
            )
            if (
                not use_frame_index
                and times is not None
                and bool(np.isfinite(times).all())
            ):
                segments.extend(
                    numeric_patch._continuous_track_segments_on_axis(track_rows)
                )
            else:
                segments.extend(original_segmenter(track_rows))
        return segments

    paper._continuous_track_segments = _continuous_track_segments
    legacy = getattr(paper, "_LEGACY", None)
    if legacy is not None:
        legacy._continuous_track_segments = _continuous_track_segments


def _patch_radar_std_message() -> None:
    validation = import_module(
        "raft_uav.baselines._radar_association_std_validation_patch"
    )
    optional_float = import_module("raft_uav.numeric").optional_float

    def _positive_finite_real(value: Any, *, name: str) -> float:
        parsed = optional_float(value)
        if parsed is None or parsed <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return parsed

    validation._positive_finite_real = _positive_finite_real


def _patch_explicit_radar_velocity_mode() -> None:
    aerpaw = import_module("raft_uav.io.aerpaw")
    original = aerpaw.radar_measurements_to_enu

    @wraps(original)
    def radar_measurements_to_enu(
        radar: pd.DataFrame,
        projector: Any = None,
        truth_origin_time: pd.Timestamp | None = None,
        default_xy_std_m: float = 25.0,
        default_z_std_m: float = 35.0,
        default_velocity_std_mps: float = 12.0,
        include_velocity: bool = False,
        clock_offset_s: float = aerpaw.DEFAULT_RADAR_CLOCK_OFFSET_S,
    ) -> list[Any]:
        if isinstance(include_velocity, (bool, np.bool_)) and bool(include_velocity):
            required = (
                "velocity_east_mps",
                "velocity_north_mps",
                "velocity_down_mps",
            )
            missing = [column for column in required if column not in radar.columns]
            if missing and not radar.empty:
                raise ValueError(
                    "include_velocity=True requires complete finite radar velocity "
                    f"components; missing columns: {missing}"
                )
            if not radar.empty and not missing:
                velocity = radar.loc[:, required].apply(
                    pd.to_numeric, errors="coerce"
                )
                finite = np.isfinite(velocity.to_numpy(dtype=float)).all(axis=1)
                if not bool(finite.all()):
                    bad_rows = radar.index[~finite].tolist()[:8]
                    raise ValueError(
                        "include_velocity=True requires complete finite radar "
                        f"velocity components at every row; invalid rows: {bad_rows}"
                    )
        return original(
            radar,
            projector=projector,
            truth_origin_time=truth_origin_time,
            default_xy_std_m=default_xy_std_m,
            default_z_std_m=default_z_std_m,
            default_velocity_std_mps=default_velocity_std_mps,
            include_velocity=include_velocity,
            clock_offset_s=clock_offset_s,
        )

    previous = aerpaw.radar_measurements_to_enu
    aerpaw.radar_measurements_to_enu = radar_measurements_to_enu
    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", {}) if module is not None else {}
        if namespace.get("radar_measurements_to_enu") is previous:
            setattr(module, "radar_measurements_to_enu", radar_measurements_to_enu)


def install() -> None:
    """Install the post-hook compatibility reconciliation once."""

    marker_module = sys.modules[__name__]
    if getattr(marker_module, "_installed", False):
        return
    _patch_factor_graph_sequence_guard()
    _patch_golden_metrics_schema_message()
    _patch_split_manifest_overlap()
    _patch_candidate_protection_flags()
    _patch_completion_default_sentinels()
    _patch_timestamp_alias_cycles()
    _patch_track5_probability_message()
    _patch_paper_timestamp_cadence()
    _patch_radar_std_message()
    _patch_explicit_radar_velocity_mode()
    marker_module._installed = True
