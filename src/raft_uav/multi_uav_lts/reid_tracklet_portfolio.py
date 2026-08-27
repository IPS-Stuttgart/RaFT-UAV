"""ReID-aware tracklet portfolio fusion and bridge-only LTS gap repair.

The module chooses complete observed trajectory blocks from pre-generated
prediction sources, never averages boxes from different sources, and only
creates new boxes inside short internal gaps. Appearance is optional in the
core API and is supplied by a small provider protocol, which keeps the
selection logic deterministic and independently testable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Protocol

import numpy as np

from ._records import (
    Detection,
    box_iou,
    format_detection,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
)
from .trajectory_box_calibration import BoxCalibrationParameters, _smooth_track
from .trajectory_gap_completion import _bridge_gap

_SCHEMA = "raft-uav-multi-uav-lts-reid-tracklet-portfolio-v1"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
_EPS = 1.0e-12


class AppearanceProvider(Protocol):
    """Return one normalized feature row per requested detection."""

    def embed(self, sequence: str, rows: Sequence[Detection]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class PortfolioParameters:
    """Truth-free controls for source selection and internal-gap completion."""

    raw_candidate: str = "raw"
    window_frames: int = 60
    min_segment_rows: int = 1
    sample_count_per_window: int = 3
    coverage_weight: float = 1.0
    confidence_weight: float = 0.05
    appearance_weight: float = 0.75
    support_weight: float = 0.20
    smoothness_weight: float = 0.05
    source_switch_penalty: float = 0.04
    transition_speed_weight: float = 0.01
    transition_size_weight: float = 0.02
    max_transition_normalized_speed: float = 8.0
    bridge_max_gap_frames: int = 0
    bridge_max_normalized_speed: float = 5.0
    bridge_max_log_size_change: float = 1.0
    bridge_min_endpoint_confidence: float = 0.003
    bridge_confidence_decay: float = 0.85
    bridge_require_appearance: bool = True
    bridge_endpoint_appearance_threshold: float = 0.30
    bridge_endpoint_appearance_threshold_late: float | None = None
    bridge_anchor_appearance_threshold: float = 0.40
    bridge_anchor_appearance_threshold_late: float | None = None
    bridge_use_smoothed_endpoints: bool = True

    def validate(self) -> None:
        _candidate_name(self.raw_candidate)
        _positive_int(self.window_frames, name="window_frames")
        _positive_int(self.min_segment_rows, name="min_segment_rows")
        _positive_int(self.sample_count_per_window, name="sample_count_per_window")
        for name, value in (
            ("coverage_weight", self.coverage_weight),
            ("confidence_weight", self.confidence_weight),
            ("appearance_weight", self.appearance_weight),
            ("support_weight", self.support_weight),
            ("smoothness_weight", self.smoothness_weight),
            ("source_switch_penalty", self.source_switch_penalty),
            ("transition_speed_weight", self.transition_speed_weight),
            ("transition_size_weight", self.transition_size_weight),
        ):
            _nonnegative_finite(value, name=name)
        _positive_finite(
            self.max_transition_normalized_speed,
            name="max_transition_normalized_speed",
        )
        _nonnegative_int(self.bridge_max_gap_frames, name="bridge_max_gap_frames")
        _positive_finite(
            self.bridge_max_normalized_speed,
            name="bridge_max_normalized_speed",
        )
        _nonnegative_finite(
            self.bridge_max_log_size_change,
            name="bridge_max_log_size_change",
        )
        _unit(
            self.bridge_min_endpoint_confidence,
            name="bridge_min_endpoint_confidence",
        )
        _unit(self.bridge_confidence_decay, name="bridge_confidence_decay")
        _unit(
            self.bridge_endpoint_appearance_threshold,
            name="bridge_endpoint_appearance_threshold",
        )
        _optional_unit(
            self.bridge_endpoint_appearance_threshold_late,
            name="bridge_endpoint_appearance_threshold_late",
        )
        _unit(
            self.bridge_anchor_appearance_threshold,
            name="bridge_anchor_appearance_threshold",
        )
        _optional_unit(
            self.bridge_anchor_appearance_threshold_late,
            name="bridge_anchor_appearance_threshold_late",
        )
        if not isinstance(self.bridge_require_appearance, bool):
            raise ValueError("bridge_require_appearance must be a Boolean")
        if not isinstance(self.bridge_use_smoothed_endpoints, bool):
            raise ValueError("bridge_use_smoothed_endpoints must be a Boolean")


@dataclass(frozen=True)
class BridgeCounts:
    eligible: int = 0
    completed: int = 0
    inserted_rows: int = 0
    rejected_motion: int = 0
    rejected_size: int = 0
    rejected_confidence: int = 0
    rejected_endpoint_appearance: int = 0
    rejected_anchor_appearance: int = 0
    rejected_missing_appearance: int = 0


@dataclass(frozen=True)
class SequencePortfolioSummary:
    sequence: str
    seed_id_count: int
    input_rows_by_candidate: Mapping[str, int]
    output_rows: int
    selected_observed_rows: int
    seeded_track_count: int
    raw_birth_track_count: int
    occupied_windows: int
    source_window_counts: Mapping[str, int]
    source_switches: int
    bridge: BridgeCounts


@dataclass(frozen=True)
class PortfolioSummary:
    schema: str
    output_dir: str
    first_frame_label_dir: str
    parameters: PortfolioParameters
    candidate_paths: Mapping[str, str]
    sequence_count: int
    output_rows: int
    selected_observed_rows: int
    inserted_rows: int
    source_window_counts: Mapping[str, int]
    sequences: tuple[SequencePortfolioSummary, ...]


@dataclass(frozen=True)
class _Segment:
    source: str
    window_start: int
    rows: tuple[Detection, ...]
    emission: float


@dataclass(frozen=True)
class _TrackSelection:
    rows: tuple[Detection, ...]
    sources: tuple[str, ...]
    occupied_windows: int


class FastReidAppearanceProvider:
    """Lazy, cached adapter around the pinned upstream FastReID interface."""

    def __init__(
        self,
        sequence_root: Path,
        botsort_root: Path,
        *,
        config_path: Path | str = Path("logs/sbs_S50/config.yaml"),
        weights_path: Path | str = Path("logs/sbs_S50/model_0016.pth"),
        device: str = "0",
        crop_scales: Sequence[float] = (1.0,),
        batch_size: int = 16,
    ) -> None:
        self.sequence_root = Path(sequence_root).expanduser().resolve()
        self.botsort_root = Path(botsort_root).expanduser().resolve()
        if not self.sequence_root.is_dir():
            raise NotADirectoryError(self.sequence_root)
        if not self.botsort_root.is_dir():
            raise NotADirectoryError(self.botsort_root)
        self.config_path = _under_root(self.botsort_root, config_path)
        self.weights_path = _under_root(self.botsort_root, weights_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)
        self.device = str(device)
        self.crop_scales = _validated_scales(crop_scales)
        self.batch_size = _positive_int(batch_size, name="batch_size")
        self._encoder = None
        self._frame_paths: dict[str, tuple[Path, ...]] = {}
        self._cache: dict[tuple[object, ...], np.ndarray] = {}

    def frame_count(self, sequence: str) -> int:
        return len(self._sequence_frames(sequence))

    def embed(self, sequence: str, rows: Sequence[Detection]) -> np.ndarray:
        materialized = tuple(rows)
        if not materialized:
            return np.empty((0, 0), dtype=np.float32)
        missing: dict[int, list[tuple[int, Detection, tuple[object, ...]]]] = defaultdict(list)
        output: list[np.ndarray | None] = [None] * len(materialized)
        for index, row in enumerate(materialized):
            key = self._cache_key(sequence, row)
            cached = self._cache.get(key)
            if cached is None:
                missing[row.frame_id].append((index, row, key))
            else:
                output[index] = cached

        frame_paths = self._sequence_frames(sequence)
        cv2 = importlib.import_module("cv2")
        encoder = self._get_encoder()
        for frame_id, requests in sorted(missing.items()):
            if frame_id <= 0 or frame_id > len(frame_paths):
                raise ValueError(
                    f"{sequence}: frame {frame_id} is outside 1..{len(frame_paths)}"
                )
            image = cv2.imread(str(frame_paths[frame_id - 1]))
            if image is None:
                raise ValueError(f"failed to read image: {frame_paths[frame_id - 1]}")
            boxes = np.asarray([_tlbr(request[1]) for request in requests], dtype=float)
            feature_sets: list[np.ndarray] = []
            valid_sets: list[np.ndarray] = []
            for scale in self.crop_scales:
                scaled = _expanded_tlbr(boxes, scale, image.shape)
                features = np.asarray(encoder.inference(image, scaled), dtype=float)
                if features.ndim != 2 or features.shape[0] != len(requests):
                    raise ValueError(
                        "FastReID returned a feature matrix with unexpected shape"
                    )
                normalized, valid = _normalize_rows(features)
                feature_sets.append(normalized)
                valid_sets.append(valid)
            stacked = np.stack(feature_sets, axis=0)
            valid_stack = np.stack(valid_sets, axis=0)
            counts = np.sum(valid_stack, axis=0)
            combined = np.sum(stacked, axis=0)
            available = counts > 0
            combined[available] /= counts[available, None]
            combined, normalized_valid = _normalize_rows(combined)
            available &= normalized_valid
            combined[~available] = np.nan
            combined = combined.astype(np.float32, copy=False)
            for row_index, (output_index, _row, key) in enumerate(requests):
                feature = combined[row_index].copy()
                self._cache[key] = feature
                output[output_index] = feature

        if any(feature is None for feature in output):
            raise RuntimeError("appearance cache failed to populate every request")
        return np.stack([feature for feature in output if feature is not None], axis=0)

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        additions = (self.botsort_root, self.botsort_root / "yolov12")
        for path in reversed(additions):
            rendered = str(path)
            if rendered not in sys.path:
                sys.path.insert(0, rendered)
        module = importlib.import_module("fast_reid.fast_reid_interfece")
        interface = getattr(module, "FastReIDInterface")
        self._encoder = interface(
            str(self.config_path),
            str(self.weights_path),
            self.device,
            batch_size=self.batch_size,
        )
        return self._encoder

    def _sequence_frames(self, sequence: str) -> tuple[Path, ...]:
        cached = self._frame_paths.get(sequence)
        if cached is not None:
            return cached
        directory = self.sequence_root / sequence
        if not directory.is_dir():
            raise FileNotFoundError(f"sequence image directory is missing: {directory}")
        paths = tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                ),
                key=lambda path: _natural_key(path.name),
            )
        )
        if not paths:
            raise ValueError(f"sequence contains no supported images: {directory}")
        self._frame_paths[sequence] = paths
        return paths

    def _cache_key(self, sequence: str, row: Detection) -> tuple[object, ...]:
        return (
            sequence,
            row.frame_id,
            round(row.x1, 5),
            round(row.y1, 5),
            round(row.width, 5),
            round(row.height, 5),
            self.crop_scales,
        )


def fuse_prediction_portfolio(
    candidates: Mapping[str, Path],
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    parameters: PortfolioParameters | None = None,
    appearance_provider: AppearanceProvider | None = None,
    smoother_parameters: BoxCalibrationParameters | None = None,
    sequences: Sequence[str] | None = None,
    sequence_frame_counts: Mapping[str, int] | None = None,
) -> PortfolioSummary:
    """Select source blocks and optionally complete ReID-consistent gaps."""

    controls = parameters or PortfolioParameters()
    controls.validate()
    smoother = smoother_parameters or BoxCalibrationParameters()
    smoother.validate()
    normalized = _normalize_candidates(candidates)
    if controls.raw_candidate not in normalized:
        raise ValueError(f"raw candidate is missing: {controls.raw_candidate}")
    _validate_output_paths(normalized, first_frame_label_dir, output_dir)

    source_texts = {
        name: prediction_texts(path) for name, path in normalized.items()
    }
    manifests = {
        name: {Path(file_name).stem for file_name in texts}
        for name, texts in source_texts.items()
    }
    expected_manifest = manifests[controls.raw_candidate]
    for name, manifest in manifests.items():
        if manifest != expected_manifest:
            missing = sorted(expected_manifest - manifest)
            extra = sorted(manifest - expected_manifest)
            raise ValueError(
                f"{name}: sequence coverage mismatch; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    labels = _label_map(first_frame_label_dir)
    requested = tuple(dict.fromkeys(str(value) for value in (sequences or ())))
    selected = requested or tuple(sorted(expected_manifest))
    missing = sorted(set(selected) - expected_manifest)
    if missing:
        raise ValueError(f"unknown prediction sequences: {', '.join(missing)}")
    missing_labels = sorted(set(selected) - set(labels))
    if missing_labels:
        raise ValueError(f"missing first-frame labels: {', '.join(missing_labels)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.txt"):
        stale.unlink()

    summaries: list[SequencePortfolioSummary] = []
    for sequence in selected:
        candidate_rows: dict[str, tuple[Detection, ...]] = {}
        for name in sorted(normalized):
            file_name = f"{sequence}.txt"
            rows = tuple(
                parse_detection_text(
                    source_texts[name][file_name],
                    source=f"{normalized[name]}:{file_name}",
                )
            )
            reject_duplicate_keys(list(rows), label=f"{name} prediction")
            candidate_rows[name] = rows
        seed_rows = _seed_rows(labels[sequence])
        frame_count = None
        if sequence_frame_counts is not None:
            frame_count = sequence_frame_counts.get(sequence)
        if frame_count is None and appearance_provider is not None:
            provider_frame_count = getattr(appearance_provider, "frame_count", None)
            if callable(provider_frame_count):
                frame_count = int(provider_frame_count(sequence))
        if frame_count is None:
            frame_count = max(
                (row.frame_id for rows in candidate_rows.values() for row in rows),
                default=1,
            )
        fused, summary = fuse_sequence(
            sequence,
            candidate_rows,
            seed_rows=seed_rows,
            parameters=controls,
            appearance_provider=appearance_provider,
            smoother_parameters=smoother,
            sequence_frame_count=frame_count,
        )
        (output_dir / f"{sequence}.txt").write_text(
            "".join(format_detection(row) + "\n" for row in fused),
            encoding="utf-8",
        )
        summaries.append(summary)

    aggregate_windows: Counter[str] = Counter()
    for summary in summaries:
        aggregate_windows.update(summary.source_window_counts)
    return PortfolioSummary(
        schema=_SCHEMA,
        output_dir=str(output_dir),
        first_frame_label_dir=str(first_frame_label_dir),
        parameters=controls,
        candidate_paths={name: str(path) for name, path in normalized.items()},
        sequence_count=len(summaries),
        output_rows=sum(summary.output_rows for summary in summaries),
        selected_observed_rows=sum(
            summary.selected_observed_rows for summary in summaries
        ),
        inserted_rows=sum(summary.bridge.inserted_rows for summary in summaries),
        source_window_counts=dict(sorted(aggregate_windows.items())),
        sequences=tuple(summaries),
    )


def fuse_sequence(
    sequence: str,
    candidates: Mapping[str, Sequence[Detection]],
    *,
    seed_rows: Sequence[Detection],
    parameters: PortfolioParameters,
    appearance_provider: AppearanceProvider | None,
    smoother_parameters: BoxCalibrationParameters,
    sequence_frame_count: int,
) -> tuple[tuple[Detection, ...], SequencePortfolioSummary]:
    """Fuse one sequence while retaining raw late-birth identities unchanged."""

    parameters.validate()
    smoother_parameters.validate()
    _positive_int(sequence_frame_count, name="sequence_frame_count")
    if parameters.raw_candidate not in candidates:
        raise ValueError(f"raw candidate is missing: {parameters.raw_candidate}")

    grouped: dict[str, dict[int, tuple[Detection, ...]]] = {}
    for name, rows in candidates.items():
        _candidate_name(name)
        materialized = tuple(rows)
        reject_duplicate_keys(list(materialized), label=f"{name} prediction")
        by_id: dict[int, list[Detection]] = defaultdict(list)
        for row in materialized:
            by_id[row.object_id].append(row)
        grouped[name] = {
            object_id: tuple(sorted(values, key=lambda row: row.frame_id))
            for object_id, values in by_id.items()
        }

    seed_materialized = tuple(seed_rows)
    reject_duplicate_keys(list(seed_materialized), label="seed")
    if any(row.frame_id != 1 for row in seed_materialized):
        raise ValueError(f"{sequence}: seed rows must all be at frame 1")
    seed_by_id = {row.object_id: row for row in seed_materialized}
    raw_tracks = grouped[parameters.raw_candidate]

    output: list[Detection] = []
    source_windows: Counter[str] = Counter()
    source_switches = 0
    occupied_windows = 0
    anchor_features: dict[int, np.ndarray | None] = {}
    for object_id in sorted(seed_by_id):
        anchor = _one_feature(
            appearance_provider,
            sequence,
            seed_by_id[object_id],
        )
        anchor_features[object_id] = anchor
        selection = _select_seeded_track(
            sequence,
            object_id,
            grouped,
            anchor_feature=anchor,
            appearance_provider=appearance_provider,
            parameters=parameters,
        )
        selected_by_frame = {row.frame_id: row for row in selection.rows}
        selected_by_frame[1] = seed_by_id[object_id]
        output.extend(
            selected_by_frame[frame_id]
            for frame_id in sorted(selected_by_frame)
        )
        source_windows.update(selection.sources)
        occupied_windows += selection.occupied_windows
        source_switches += sum(
            left != right
            for left, right in zip(selection.sources, selection.sources[1:])
        )

    raw_birth_ids = sorted(set(raw_tracks) - set(seed_by_id))
    for object_id in raw_birth_ids:
        output.extend(raw_tracks[object_id])

    output.sort(key=lambda row: (row.frame_id, row.object_id))
    reject_duplicate_keys(output, label="portfolio output")
    observed_count = len(output)
    bridged, bridge_counts = _complete_selected_gaps(
        sequence,
        output,
        seed_ids=set(seed_by_id),
        anchor_features=anchor_features,
        appearance_provider=appearance_provider,
        parameters=parameters,
        smoother_parameters=smoother_parameters,
        sequence_frame_count=sequence_frame_count,
    )
    return bridged, SequencePortfolioSummary(
        sequence=sequence,
        seed_id_count=len(seed_by_id),
        input_rows_by_candidate={
            name: len(tuple(rows)) for name, rows in sorted(candidates.items())
        },
        output_rows=len(bridged),
        selected_observed_rows=observed_count,
        seeded_track_count=len(seed_by_id),
        raw_birth_track_count=len(raw_birth_ids),
        occupied_windows=occupied_windows,
        source_window_counts=dict(sorted(source_windows.items())),
        source_switches=source_switches,
        bridge=bridge_counts,
    )


def _select_seeded_track(
    sequence: str,
    object_id: int,
    grouped: Mapping[str, Mapping[int, tuple[Detection, ...]]],
    *,
    anchor_feature: np.ndarray | None,
    appearance_provider: AppearanceProvider | None,
    parameters: PortfolioParameters,
) -> _TrackSelection:
    occupied = sorted(
        {
            _window_start(row.frame_id, parameters.window_frames)
            for tracks in grouped.values()
            for row in tracks.get(object_id, ())
        }
    )
    if not occupied:
        return _TrackSelection((), (), 0)

    segments_by_window: list[list[_Segment]] = []
    for window_start in occupied:
        window_end = window_start + parameters.window_frames - 1
        rows_by_source = {
            source: tuple(
                row
                for row in tracks.get(object_id, ())
                if window_start <= row.frame_id <= window_end
            )
            for source, tracks in grouped.items()
        }
        segments: list[_Segment] = []
        for source in sorted(rows_by_source):
            rows = rows_by_source[source]
            if len(rows) < parameters.min_segment_rows:
                continue
            emission = _segment_emission(
                sequence,
                source,
                rows,
                rows_by_source,
                anchor_feature=anchor_feature,
                appearance_provider=appearance_provider,
                parameters=parameters,
            )
            segments.append(_Segment(source, window_start, rows, emission))
        if not segments:
            continue
        segments_by_window.append(segments)

    if not segments_by_window:
        return _TrackSelection((), (), 0)

    scores: list[list[float]] = []
    parents: list[list[int]] = []
    scores.append([segment.emission for segment in segments_by_window[0]])
    parents.append([-1] * len(segments_by_window[0]))
    for window_index in range(1, len(segments_by_window)):
        current_scores: list[float] = []
        current_parents: list[int] = []
        for current in segments_by_window[window_index]:
            best_score = -math.inf
            best_parent = -1
            for previous_index, previous in enumerate(
                segments_by_window[window_index - 1]
            ):
                transition = _transition_score(previous, current, parameters)
                candidate_score = (
                    scores[window_index - 1][previous_index]
                    + current.emission
                    + transition
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_parent = previous_index
            current_scores.append(best_score)
            current_parents.append(best_parent)
        scores.append(current_scores)
        parents.append(current_parents)

    final_index = int(np.argmax(scores[-1]))
    selected_segments: list[_Segment] = []
    for window_index in range(len(segments_by_window) - 1, -1, -1):
        selected_segments.append(segments_by_window[window_index][final_index])
        final_index = parents[window_index][final_index]
    selected_segments.reverse()
    rows = tuple(
        sorted(
            (row for segment in selected_segments for row in segment.rows),
            key=lambda row: row.frame_id,
        )
    )
    if len({row.frame_id for row in rows}) != len(rows):
        raise ValueError(
            f"{sequence}: source windows produced duplicate frames for identity {object_id}"
        )
    return _TrackSelection(
        rows=rows,
        sources=tuple(segment.source for segment in selected_segments),
        occupied_windows=len(selected_segments),
    )


def _segment_emission(
    sequence: str,
    source: str,
    rows: tuple[Detection, ...],
    rows_by_source: Mapping[str, tuple[Detection, ...]],
    *,
    anchor_feature: np.ndarray | None,
    appearance_provider: AppearanceProvider | None,
    parameters: PortfolioParameters,
) -> float:
    coverage = len({row.frame_id for row in rows}) / parameters.window_frames
    confidence = float(np.mean([row.confidence for row in rows]))
    support = _cross_source_support(
        rows,
        rows_by_source,
        current_source=source,
    )
    smoothness = _trajectory_jitter(rows)
    score = (
        parameters.coverage_weight * coverage
        + parameters.confidence_weight * confidence
        + parameters.support_weight * support
        - parameters.smoothness_weight * smoothness
    )
    if (
        parameters.appearance_weight > 0.0
        and appearance_provider is not None
        and anchor_feature is not None
    ):
        sampled = _sample_rows(rows, parameters.sample_count_per_window)
        features = appearance_provider.embed(sequence, sampled)
        distances = [
            _cosine_distance(anchor_feature, feature)
            for feature in features
            if _valid_feature(feature)
        ]
        if distances:
            similarity = 1.0 - float(np.mean(distances))
            score += parameters.appearance_weight * similarity
    return score


def _transition_score(
    previous: _Segment,
    current: _Segment,
    parameters: PortfolioParameters,
) -> float:
    left = previous.rows[-1]
    right = current.rows[0]
    duration = right.frame_id - left.frame_id
    if duration <= 0:
        return -1.0e6
    scale = _pair_scale(left, right)
    speed = math.hypot(
        right.center_x - left.center_x,
        right.center_y - left.center_y,
    ) / (scale * duration)
    if speed > parameters.max_transition_normalized_speed:
        return -1.0e6 - speed
    size_change = abs(math.log(right.width / left.width)) + abs(
        math.log(right.height / left.height)
    )
    penalty = (
        parameters.transition_speed_weight * speed
        + parameters.transition_size_weight * size_change
    )
    if previous.source != current.source:
        penalty += parameters.source_switch_penalty
    return -penalty


def _complete_selected_gaps(
    sequence: str,
    rows: Sequence[Detection],
    *,
    seed_ids: set[int],
    anchor_features: Mapping[int, np.ndarray | None],
    appearance_provider: AppearanceProvider | None,
    parameters: PortfolioParameters,
    smoother_parameters: BoxCalibrationParameters,
    sequence_frame_count: int,
) -> tuple[tuple[Detection, ...], BridgeCounts]:
    if parameters.bridge_max_gap_frames <= 0:
        return tuple(rows), BridgeCounts()
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for row in rows:
        grouped[row.object_id].append(row)
    output = list(rows)
    counts = Counter()
    for object_id in sorted(seed_ids):
        track = tuple(sorted(grouped.get(object_id, ()), key=lambda row: row.frame_id))
        if len(track) < 2:
            continue
        smoothed = (
            _smooth_track(track, smoother_parameters)
            if parameters.bridge_use_smoothed_endpoints
            else ()
        )
        anchor = anchor_features.get(object_id)
        for index, (left, right) in enumerate(zip(track, track[1:])):
            gap = right.frame_id - left.frame_id - 1
            if gap <= 0 or gap > parameters.bridge_max_gap_frames:
                continue
            counts["eligible"] += 1
            if (
                min(left.confidence, right.confidence)
                < parameters.bridge_min_endpoint_confidence
            ):
                counts["rejected_confidence"] += 1
                continue
            duration = right.frame_id - left.frame_id
            speed = math.hypot(
                right.center_x - left.center_x,
                right.center_y - left.center_y,
            ) / (_pair_scale(left, right) * duration)
            if speed > parameters.bridge_max_normalized_speed:
                counts["rejected_motion"] += 1
                continue
            size_change = abs(math.log(right.width / left.width)) + abs(
                math.log(right.height / left.height)
            )
            if (
                size_change > parameters.bridge_max_log_size_change
                or left.class_id != right.class_id
            ):
                counts["rejected_size"] += 1
                continue

            if parameters.bridge_require_appearance:
                if appearance_provider is None or anchor is None:
                    counts["rejected_missing_appearance"] += 1
                    continue
                endpoint_features = appearance_provider.embed(
                    sequence,
                    (left, right),
                )
                if (
                    endpoint_features.shape[0] != 2
                    or not all(_valid_feature(feature) for feature in endpoint_features)
                ):
                    counts["rejected_missing_appearance"] += 1
                    continue
                phase = min(
                    1.0,
                    max(
                        0.0,
                        (0.5 * (left.frame_id + right.frame_id) - 1.0)
                        / max(1.0, sequence_frame_count - 1.0),
                    ),
                )
                endpoint_threshold = _phase_value(
                    parameters.bridge_endpoint_appearance_threshold,
                    parameters.bridge_endpoint_appearance_threshold_late,
                    phase,
                )
                anchor_threshold = _phase_value(
                    parameters.bridge_anchor_appearance_threshold,
                    parameters.bridge_anchor_appearance_threshold_late,
                    phase,
                )
                if (
                    _cosine_distance(endpoint_features[0], endpoint_features[1])
                    > endpoint_threshold
                ):
                    counts["rejected_endpoint_appearance"] += 1
                    continue
                if max(
                    _cosine_distance(anchor, endpoint_features[0]),
                    _cosine_distance(anchor, endpoint_features[1]),
                ) > anchor_threshold:
                    counts["rejected_anchor_appearance"] += 1
                    continue

            left_state = smoothed[index].state if smoothed else None
            right_state = smoothed[index + 1].state if smoothed else None
            inserted = _bridge_gap(
                left,
                right,
                gap,
                left_state=left_state,
                right_state=right_state,
                confidence_decay=parameters.bridge_confidence_decay,
            )
            output.extend(inserted)
            counts["completed"] += 1
            counts["inserted_rows"] += len(inserted)

    output.sort(key=lambda row: (row.frame_id, row.object_id))
    reject_duplicate_keys(output, label="bridged portfolio output")
    return tuple(output), BridgeCounts(
        eligible=counts["eligible"],
        completed=counts["completed"],
        inserted_rows=counts["inserted_rows"],
        rejected_motion=counts["rejected_motion"],
        rejected_size=counts["rejected_size"],
        rejected_confidence=counts["rejected_confidence"],
        rejected_endpoint_appearance=counts["rejected_endpoint_appearance"],
        rejected_anchor_appearance=counts["rejected_anchor_appearance"],
        rejected_missing_appearance=counts["rejected_missing_appearance"],
    )


def _cross_source_support(
    rows: Sequence[Detection],
    rows_by_source: Mapping[str, tuple[Detection, ...]],
    *,
    current_source: str,
) -> float:
    if len(rows_by_source) <= 1:
        return 0.0
    maps = {
        source: {row.frame_id: row for row in values}
        for source, values in rows_by_source.items()
        if source != current_source
    }
    values: list[float] = []
    for row in rows:
        alternatives = [
            box_iou(row, frame_map[row.frame_id])
            for frame_map in maps.values()
            if row.frame_id in frame_map
        ]
        if alternatives:
            values.append(max(alternatives))
    return 0.0 if not values else float(np.mean(values))


def _trajectory_jitter(rows: Sequence[Detection]) -> float:
    ordered = tuple(sorted(rows, key=lambda row: row.frame_id))
    if len(ordered) < 3:
        return 0.0
    velocities: list[tuple[float, float]] = []
    for left, right in zip(ordered, ordered[1:]):
        duration = right.frame_id - left.frame_id
        if duration <= 0:
            continue
        scale = _pair_scale(left, right)
        velocities.append(
            (
                (right.center_x - left.center_x) / (scale * duration),
                (right.center_y - left.center_y) / (scale * duration),
            )
        )
    if len(velocities) < 2:
        return 0.0
    changes = [
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(velocities, velocities[1:])
    ]
    return min(10.0, float(np.mean(changes)))


def _sample_rows(rows: Sequence[Detection], count: int) -> tuple[Detection, ...]:
    materialized = tuple(rows)
    if len(materialized) <= count:
        return materialized
    indices = np.linspace(0, len(materialized) - 1, count)
    return tuple(materialized[int(round(index))] for index in indices)


def _one_feature(
    provider: AppearanceProvider | None,
    sequence: str,
    row: Detection,
) -> np.ndarray | None:
    if provider is None:
        return None
    features = np.asarray(provider.embed(sequence, (row,)), dtype=float)
    if features.ndim != 2 or features.shape[0] != 1:
        raise ValueError("appearance provider returned an unexpected feature shape")
    feature = features[0]
    return feature if _valid_feature(feature) else None


def _valid_feature(feature: np.ndarray) -> bool:
    array = np.asarray(feature, dtype=float).reshape(-1)
    return bool(
        array.size > 0
        and np.all(np.isfinite(array))
        and np.linalg.norm(array) > _EPS
    )


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = np.asarray(left, dtype=float).reshape(-1)
    right_vector = np.asarray(right, dtype=float).reshape(-1)
    if left_vector.size != right_vector.size:
        raise ValueError("appearance feature dimensions differ")
    left_norm = np.linalg.norm(left_vector)
    right_norm = np.linalg.norm(right_vector)
    if (
        not np.isfinite(left_norm)
        or not np.isfinite(right_norm)
        or left_norm <= _EPS
        or right_norm <= _EPS
    ):
        return math.inf
    cosine = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
    return 0.5 * (1.0 - max(-1.0, min(1.0, cosine)))


def _phase_value(early: float, late: float | None, phase: float) -> float:
    late_value = early if late is None else late
    fraction = min(1.0, max(0.0, float(phase)))
    return (1.0 - fraction) * early + fraction * late_value


def _window_start(frame_id: int, window_frames: int) -> int:
    return ((frame_id - 1) // window_frames) * window_frames + 1


def _pair_scale(left: Detection, right: Detection) -> float:
    return max(
        1.0,
        0.5
        * (
            math.sqrt(max(_EPS, left.width * left.height))
            + math.sqrt(max(_EPS, right.width * right.height))
        ),
    )


def _tlbr(row: Detection) -> tuple[float, float, float, float]:
    return (
        row.x1,
        row.y1,
        row.x1 + row.width,
        row.y1 + row.height,
    )


def _expanded_tlbr(
    boxes: np.ndarray,
    scale: float,
    image_shape: Sequence[int],
) -> np.ndarray:
    array = np.asarray(boxes, dtype=float).copy()
    if array.size == 0:
        return array.reshape(0, 4)
    height, width = int(image_shape[0]), int(image_shape[1])
    centers = 0.5 * (array[:, :2] + array[:, 2:])
    half_sizes = 0.5 * np.maximum(array[:, 2:] - array[:, :2], 1.0) * scale
    lower = np.floor(centers - half_sizes)
    upper = np.ceil(centers + half_sizes)
    lower[:, 0] = np.clip(lower[:, 0], 0, width - 2)
    lower[:, 1] = np.clip(lower[:, 1], 0, height - 2)
    upper[:, 0] = np.clip(upper[:, 0], lower[:, 0] + 1, width - 1)
    upper[:, 1] = np.clip(upper[:, 1], lower[:, 1] + 1, height - 1)
    return np.concatenate((lower, upper), axis=1)


def _normalize_rows(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2:
        raise ValueError("appearance features must be a matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > _EPS)
    output = np.zeros_like(array, dtype=float)
    output[valid] = array[valid] / norms[valid]
    return output, valid


def _normalize_candidates(candidates: Mapping[str, Path]) -> dict[str, Path]:
    if not candidates:
        raise ValueError("at least one prediction candidate is required")
    normalized: dict[str, Path] = {}
    for raw_name, raw_path in candidates.items():
        name = _candidate_name(raw_name)
        if name in normalized:
            raise ValueError(f"duplicate candidate: {name}")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        normalized[name] = path
    return normalized


def _label_map(label_dir: Path) -> dict[str, Path]:
    root = Path(label_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(root)
    labels = {path.stem: path for path in sorted(root.glob("*.txt"))}
    if not labels:
        raise ValueError(f"first-frame label directory contains no .txt files: {root}")
    return labels


def _seed_rows(path: Path) -> tuple[Detection, ...]:
    rows = tuple(
        parse_detection_text(path.read_text(encoding="utf-8"), source=str(path))
    )
    reject_duplicate_keys(list(rows), label="seed")
    if any(row.frame_id != 1 for row in rows):
        raise ValueError(f"{path}: expected first-frame-only labels")
    return rows


def _validate_output_paths(
    candidates: Mapping[str, Path],
    label_dir: Path,
    output_dir: Path,
) -> None:
    output = Path(output_dir).expanduser().resolve()
    labels = Path(label_dir).expanduser().resolve()
    if output == labels or labels in output.parents or output in labels.parents:
        raise ValueError("output directory must be disjoint from first-frame labels")
    for name, source_path in candidates.items():
        if not source_path.is_dir():
            continue
        source = source_path.expanduser().resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError(
                f"output directory must be disjoint from candidate {name}"
            )


def _candidate_name(value: object) -> str:
    name = str(value).strip()
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid candidate name: {value!r}")
    return name


def _parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("candidates must use NAME=PATH")
    try:
        normalized = _candidate_name(name)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return normalized, Path(raw_path).expanduser()


def _parse_scales(value: str) -> tuple[float, ...]:
    try:
        return _validated_scales(item.strip() for item in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _validated_scales(values: Sequence[float] | object) -> tuple[float, ...]:
    materialized = tuple(float(value) for value in values)  # type: ignore[arg-type]
    if not materialized:
        raise ValueError("at least one crop scale is required")
    if len(set(materialized)) != len(materialized):
        raise ValueError("crop scales must not contain duplicates")
    for value in materialized:
        if not math.isfinite(value) or not 0.5 <= value <= 3.0:
            raise ValueError("crop scales must be finite values in [0.5, 3.0]")
    return materialized


def _under_root(root: Path, path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return parsed


def _positive_finite(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _unit(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def _optional_unit(value: object, *, name: str) -> float | None:
    return None if value is None else _unit(value, name=name)


def write_summary(summary: PortfolioSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--raw-candidate", default="raw")
    parser.add_argument("--first-frame-label-dir", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--botsort-root", type=Path)
    parser.add_argument("--fast-reid-config", type=Path, default=Path("logs/sbs_S50/config.yaml"))
    parser.add_argument("--fast-reid-weights", type=Path, default=Path("logs/sbs_S50/model_0016.pth"))
    parser.add_argument("--device", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--crop-scales", type=_parse_scales, default=(1.0,))
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--no-appearance", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--window-frames", type=int, default=60)
    parser.add_argument("--min-segment-rows", type=int, default=1)
    parser.add_argument("--sample-count-per-window", type=int, default=3)
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--confidence-weight", type=float, default=0.05)
    parser.add_argument("--appearance-weight", type=float, default=0.75)
    parser.add_argument("--support-weight", type=float, default=0.20)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--source-switch-penalty", type=float, default=0.04)
    parser.add_argument("--transition-speed-weight", type=float, default=0.01)
    parser.add_argument("--transition-size-weight", type=float, default=0.02)
    parser.add_argument("--max-transition-normalized-speed", type=float, default=8.0)
    parser.add_argument("--bridge-max-gap-frames", type=int, default=0)
    parser.add_argument("--bridge-max-normalized-speed", type=float, default=5.0)
    parser.add_argument("--bridge-max-log-size-change", type=float, default=1.0)
    parser.add_argument("--bridge-min-endpoint-confidence", type=float, default=0.003)
    parser.add_argument("--bridge-confidence-decay", type=float, default=0.85)
    parser.add_argument(
        "--bridge-require-appearance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bridge-endpoint-appearance-threshold", type=float, default=0.30)
    parser.add_argument("--bridge-endpoint-appearance-threshold-late", type=float)
    parser.add_argument("--bridge-anchor-appearance-threshold", type=float, default=0.40)
    parser.add_argument("--bridge-anchor-appearance-threshold-late", type=float)
    parser.add_argument(
        "--bridge-use-smoothed-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--process-accel-sigma", type=float, default=0.50)
    parser.add_argument("--process-log-size-sigma", type=float, default=0.035)
    parser.add_argument("--center-measurement-sigma", type=float, default=0.30)
    parser.add_argument("--log-size-measurement-sigma", type=float, default=0.12)
    args = parser.parse_args(argv)

    candidates: dict[str, Path] = {}
    for name, path in args.candidate:
        if name in candidates:
            parser.error(f"duplicate candidate: {name}")
        candidates[name] = path
    appearance_provider: AppearanceProvider | None = None
    if not args.no_appearance:
        if args.sequence_root is None or args.botsort_root is None:
            parser.error("appearance requires --sequence-root and --botsort-root")
        appearance_provider = FastReidAppearanceProvider(
            args.sequence_root,
            args.botsort_root,
            config_path=args.fast_reid_config,
            weights_path=args.fast_reid_weights,
            device=str(args.device),
            crop_scales=args.crop_scales,
            batch_size=args.embedding_batch_size,
        )
    parameters = PortfolioParameters(
        raw_candidate=args.raw_candidate,
        window_frames=args.window_frames,
        min_segment_rows=args.min_segment_rows,
        sample_count_per_window=args.sample_count_per_window,
        coverage_weight=args.coverage_weight,
        confidence_weight=args.confidence_weight,
        appearance_weight=args.appearance_weight,
        support_weight=args.support_weight,
        smoothness_weight=args.smoothness_weight,
        source_switch_penalty=args.source_switch_penalty,
        transition_speed_weight=args.transition_speed_weight,
        transition_size_weight=args.transition_size_weight,
        max_transition_normalized_speed=args.max_transition_normalized_speed,
        bridge_max_gap_frames=args.bridge_max_gap_frames,
        bridge_max_normalized_speed=args.bridge_max_normalized_speed,
        bridge_max_log_size_change=args.bridge_max_log_size_change,
        bridge_min_endpoint_confidence=args.bridge_min_endpoint_confidence,
        bridge_confidence_decay=args.bridge_confidence_decay,
        bridge_require_appearance=args.bridge_require_appearance,
        bridge_endpoint_appearance_threshold=args.bridge_endpoint_appearance_threshold,
        bridge_endpoint_appearance_threshold_late=args.bridge_endpoint_appearance_threshold_late,
        bridge_anchor_appearance_threshold=args.bridge_anchor_appearance_threshold,
        bridge_anchor_appearance_threshold_late=args.bridge_anchor_appearance_threshold_late,
        bridge_use_smoothed_endpoints=args.bridge_use_smoothed_endpoints,
    )
    smoother = BoxCalibrationParameters(
        process_accel_sigma=args.process_accel_sigma,
        process_log_size_sigma=args.process_log_size_sigma,
        center_measurement_sigma=args.center_measurement_sigma,
        log_size_measurement_sigma=args.log_size_measurement_sigma,
    )
    summary = fuse_prediction_portfolio(
        candidates,
        args.first_frame_label_dir,
        args.output_dir,
        parameters=parameters,
        appearance_provider=appearance_provider,
        smoother_parameters=smoother,
        sequences=args.sequences,
    )
    if args.output_json is not None:
        write_summary(summary, args.output_json)
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
