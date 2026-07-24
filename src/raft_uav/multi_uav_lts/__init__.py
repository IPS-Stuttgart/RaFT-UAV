"""Utilities for the Beyond Strong Baseline Multi-UAV Tracking LTS benchmark."""

from __future__ import annotations

import math


def _install_zero_frame_coverage_guard() -> None:
    try:
        from raft_uav.multi_uav_lts import coverage_audit as _coverage_audit
        from raft_uav.multi_uav_lts.cli import _parse_int_like as _parse_int_like
    except Exception:
        return

    def _count_out_of_range_frame_rows(text: str, *, expected_frame_count: int | None) -> int:
        if expected_frame_count is None:
            return 0
        count = 0
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 1:
                continue
            try:
                frame_id = _parse_int_like(parts[0])
            except ValueError:
                continue
            if frame_id < 1 or frame_id > expected_frame_count:
                count += 1
        return count

    _coverage_audit._count_out_of_range_frame_rows = _count_out_of_range_frame_rows


def _install_lts_submission_domain_guard() -> None:
    try:
        from raft_uav.multi_uav_lts import cli as _cli
        from raft_uav.multi_uav_lts.cli import _parse_int_like as _parse_int_like
    except Exception:
        return

    installed_attr = "_raft_uav_submission_domain_guard_installed"
    if getattr(_cli, installed_attr, False):
        return

    original_attr = "_raft_uav_original_summarize_prediction_text"
    if not hasattr(_cli, original_attr):
        setattr(_cli, original_attr, _cli._summarize_prediction_text)
    original = getattr(_cli, original_attr)

    def _summarize_prediction_text(name: str, text: str):
        summary = original(name, text)
        invalid_domain_rows = _count_invalid_class_visibility_rows(
            text,
            parse_int_like=_parse_int_like,
        )
        if invalid_domain_rows == 0:
            return summary
        return _cli.SubmissionFileSummary(
            name=summary.name,
            row_count=summary.row_count,
            first_frame=summary.first_frame,
            last_frame=summary.last_frame,
            unique_object_ids=summary.unique_object_ids,
            parse_errors=summary.parse_errors + invalid_domain_rows,
            invalid_geometry_rows=summary.invalid_geometry_rows,
            invalid_confidence_rows=summary.invalid_confidence_rows,
            unsorted_rows=summary.unsorted_rows,
        )

    _cli._summarize_prediction_text = _summarize_prediction_text
    setattr(_cli, installed_attr, True)


def _install_lts_duplicate_key_validation_guard() -> None:
    try:
        from raft_uav.multi_uav_lts import cli as _cli
        from raft_uav.numeric import optional_int as _optional_int
    except Exception:
        return

    installed_attr = "_raft_uav_duplicate_key_validation_guard_installed"
    if getattr(_cli, installed_attr, False):
        return
    original = _cli._summarize_prediction_text

    def _summarize_prediction_text(name: str, text: str):
        summary = original(name, text)
        duplicate_rows = _count_duplicate_frame_object_rows(
            text,
            parse_int_like=_optional_int,
        )
        if duplicate_rows == 0:
            return summary
        return _cli.SubmissionFileSummary(
            name=summary.name,
            row_count=summary.row_count,
            first_frame=summary.first_frame,
            last_frame=summary.last_frame,
            unique_object_ids=summary.unique_object_ids,
            parse_errors=summary.parse_errors + duplicate_rows,
            invalid_geometry_rows=summary.invalid_geometry_rows,
            invalid_confidence_rows=summary.invalid_confidence_rows,
            unsorted_rows=summary.unsorted_rows,
        )

    _cli._summarize_prediction_text = _summarize_prediction_text
    setattr(_cli, installed_attr, True)


def _count_invalid_class_visibility_rows(text: str, *, parse_int_like) -> int:
    count = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9:
            continue
        try:
            frame_id = parse_int_like(parts[0])
            object_id = parse_int_like(parts[1])
            x1, y1, w, h, confidence = (float(part) for part in parts[2:7])
            class_id = parse_int_like(parts[7])
            visibility = float(parts[8])
        except ValueError:
            continue

        if frame_id <= 0 or object_id <= 0:
            continue
        if not all(math.isfinite(value) for value in (x1, y1, w, h, confidence, visibility)):
            continue
        if class_id <= 0 or not 0.0 <= visibility <= 1.0:
            count += 1
    return count


def _count_duplicate_frame_object_rows(text: str, *, parse_int_like) -> int:
    duplicate_rows = 0
    seen: set[tuple[int, int]] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9:
            continue
        try:
            frame_id = parse_int_like(parts[0])
            object_id = parse_int_like(parts[1])
        except ValueError:
            continue
        if frame_id is None or object_id is None:
            continue
        if frame_id <= 0 or object_id <= 0:
            continue
        key = (int(frame_id), int(object_id))
        if key in seen:
            duplicate_rows += 1
        else:
            seen.add(key)
    return duplicate_rows


_install_zero_frame_coverage_guard()
_install_lts_submission_domain_guard()
_install_lts_duplicate_key_validation_guard()
