"""Shared parsing and geometry helpers for Multi-UAV LTS tools."""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from raft_uav.numeric import optional_float, optional_int


@dataclass(frozen=True)
class Detection:
    frame_id: int
    object_id: int
    x1: float
    y1: float
    width: float
    height: float
    confidence: float
    class_id: int
    visibility: float

    @property
    def center_x(self) -> float:
        return self.x1 + 0.5 * self.width

    @property
    def center_y(self) -> float:
        return self.y1 + 0.5 * self.height


@dataclass(frozen=True)
class PreparedSequence:
    frame_count: int
    gt_ids: tuple[np.ndarray, ...]
    tracker_ids: tuple[np.ndarray, ...]
    similarity_scores: tuple[np.ndarray, ...]
    num_gt_ids: int
    num_tracker_ids: int
    num_gt_detections: int
    num_tracker_detections: int


def prediction_texts(path: Path) -> dict[str, str]:
    if path.is_dir():
        return {
            file_path.name: file_path.read_text(encoding="utf-8")
            for file_path in sorted(path.glob("*.txt"))
        }
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != len(set(names)):
            raise ValueError("prediction ZIP contains duplicate member names")
        nested = [name for name in names if "/" in name.rstrip("/") or "\\" in name]
        if nested:
            raise ValueError(f"prediction ZIP contains nested members: {nested[0]}")
        return {
            name: archive.read(name).decode("utf-8")
            for name in names
            if name.endswith(".txt")
        }


def parse_detection_text(text: str, *, source: str) -> list[Detection]:
    rows: list[Detection] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9:
            raise ValueError(f"{source}:{line_number}: expected 9 columns, got {len(parts)}")
        try:
            frame_id = parse_int(parts[0])
            object_id = parse_int(parts[1])
            x1, y1, width, height, confidence = (float(value) for value in parts[2:7])
            class_id = parse_int(parts[7])
            visibility = float(parts[8])
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: malformed row") from exc
        values = (x1, y1, width, height, confidence, visibility)
        if frame_id <= 0 or object_id <= 0 or class_id <= 0:
            raise ValueError(f"{source}:{line_number}: IDs and class must be positive")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{source}:{line_number}: non-finite value")
        if width <= 0 or height <= 0:
            raise ValueError(f"{source}:{line_number}: box dimensions must be positive")
        if not 0.0 <= visibility <= 1.0:
            raise ValueError(f"{source}:{line_number}: visibility must be in [0, 1]")
        rows.append(
            Detection(
                frame_id,
                object_id,
                x1,
                y1,
                width,
                height,
                confidence,
                class_id,
                visibility,
            )
        )
    return rows


def prepare_sequence(
    truth_rows: list[Detection], prediction_rows: list[Detection]
) -> PreparedSequence:
    reject_duplicate_keys(truth_rows, label="truth")
    reject_duplicate_keys(prediction_rows, label="predictions")
    gt_map = {
        value: index
        for index, value in enumerate(sorted({row.object_id for row in truth_rows}))
    }
    tracker_map = {
        value: index
        for index, value in enumerate(sorted({row.object_id for row in prediction_rows}))
    }
    truth_by_frame = rows_by_frame(truth_rows)
    predictions_by_frame = rows_by_frame(prediction_rows)
    frame_count = max([0, *truth_by_frame, *predictions_by_frame])
    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    similarities: list[np.ndarray] = []
    for frame_id in range(1, frame_count + 1):
        gt_frame = truth_by_frame.get(frame_id, ())
        tracker_frame = predictions_by_frame.get(frame_id, ())
        gt_ids.append(np.asarray([gt_map[row.object_id] for row in gt_frame], dtype=int))
        tracker_ids.append(
            np.asarray([tracker_map[row.object_id] for row in tracker_frame], dtype=int)
        )
        similarities.append(iou_matrix(gt_frame, tracker_frame))
    return PreparedSequence(
        frame_count,
        tuple(gt_ids),
        tuple(tracker_ids),
        tuple(similarities),
        len(gt_map),
        len(tracker_map),
        len(truth_rows),
        len(prediction_rows),
    )


def parse_int(value: str) -> int:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(value) from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(value)
    return int(parsed)


def reject_duplicate_keys(rows: list[Detection], *, label: str) -> None:
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (row.frame_id, row.object_id)
        if key in seen:
            raise ValueError(f"duplicate {label} frame/object key: {key}")
        seen.add(key)


def rows_by_frame(rows: list[Detection]) -> dict[int, tuple[Detection, ...]]:
    grouped: dict[int, list[Detection]] = {}
    for row in rows:
        grouped.setdefault(row.frame_id, []).append(row)
    return {
        frame_id: tuple(sorted(frame_rows, key=lambda row: row.object_id))
        for frame_id, frame_rows in grouped.items()
    }


def iou_matrix(
    left: tuple[Detection, ...], right: tuple[Detection, ...]
) -> np.ndarray:
    matrix = np.zeros((len(left), len(right)), dtype=float)
    for left_index, left_row in enumerate(left):
        for right_index, right_row in enumerate(right):
            matrix[left_index, right_index] = box_iou(left_row, right_row)
    return matrix


def box_iou(left: Detection, right: Detection) -> float:
    left_x2 = left.x1 + left.width
    left_y2 = left.y1 + left.height
    right_x2 = right.x1 + right.width
    right_y2 = right.y1 + right.height
    inter_width = max(0.0, min(left_x2, right_x2) - max(left.x1, right.x1))
    inter_height = max(0.0, min(left_y2, right_y2) - max(left.y1, right.y1))
    intersection = inter_width * inter_height
    union = left.width * left.height + right.width * right.height - intersection
    return 0.0 if union <= 0.0 else intersection / union


def format_detection(row: Detection) -> str:
    values: tuple[int | float, ...] = (
        row.frame_id,
        row.object_id,
        row.x1,
        row.y1,
        row.width,
        row.height,
        row.confidence,
        row.class_id,
        row.visibility,
    )
    return ",".join(format_value(value) for value in values)


def format_value(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".15g")


def validate_unit_interval(value: object, *, name: str) -> float:
    parsed = validate_nonnegative_finite(value, name=name)
    if parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def validate_nonnegative_finite(value: object, *, name: str) -> float:
    parsed = optional_float(value)
    if parsed is None or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return parsed


def validate_nonnegative_int(value: object, *, name: str) -> int:
    parsed = optional_int(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed
