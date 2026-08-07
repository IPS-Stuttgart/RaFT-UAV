"""Utilities for the Beyond Strong Baseline Multi-UAV Tracking LTS benchmark."""

from __future__ import annotations

import math
from functools import wraps
from pathlib import Path


def _install_lts_input_output_alias_guard() -> None:
    try:
        from raft_uav.multi_uav_lts import cli as _cli
    except Exception:
        return

    installed_attr = "_raft_uav_input_output_alias_guard_installed"
    if getattr(_cli, installed_attr, False):
        return

    original_package_submission = _cli.package_submission
    original_write_constant_first_frame_predictions = (
        _cli.write_constant_first_frame_predictions
    )
    original_write_first_frame_labels = _cli.write_first_frame_labels

    def _paths_alias(left: Path, right: Path) -> bool:
        return Path(left).resolve() == Path(right).resolve()

    @wraps(original_package_submission)
    def _package_submission(
        prediction_dir: Path,
        output_zip: Path,
        *,
        template_zip: Path | None = None,
        normalize: bool = False,
        sort_rows: bool = False,
    ):
        prediction_sources = list(Path(prediction_dir).glob("*.txt"))
        if template_zip is not None:
            if _paths_alias(output_zip, template_zip):
                raise ValueError(
                    f"output ZIP must differ from template ZIP: {output_zip}"
                )
            expected_names = _cli._validated_template_member_names(
                Path(template_zip)
            )
            prediction_sources.extend(
                Path(prediction_dir) / name for name in expected_names or ()
            )
        for source in prediction_sources:
            if _paths_alias(output_zip, source):
                raise ValueError(
                    f"output ZIP must differ from prediction input: {source}"
                )
        return original_package_submission(
            prediction_dir,
            output_zip,
            template_zip=template_zip,
            normalize=normalize,
            sort_rows=sort_rows,
        )

    @wraps(original_write_constant_first_frame_predictions)
    def _write_constant_first_frame_predictions(
        sequence_root: Path,
        first_frame_label_dir: Path,
        output_dir: Path,
    ):
        if _paths_alias(output_dir, first_frame_label_dir):
            raise ValueError(
                "output directory must differ from first-frame label directory: "
                f"{output_dir}"
            )
        return original_write_constant_first_frame_predictions(
            sequence_root,
            first_frame_label_dir,
            output_dir,
        )

    @wraps(original_write_first_frame_labels)
    def _write_first_frame_labels(
        truth_dir: Path,
        output_dir: Path,
        *,
        frame_id: int = 1,
    ):
        if _paths_alias(output_dir, truth_dir):
            raise ValueError(
                f"output directory must differ from truth directory: {output_dir}"
            )
        return original_write_first_frame_labels(
            truth_dir,
            output_dir,
            frame_id=frame_id,
        )

    _cli.package_submission = _package_submission
    _cli.write_constant_first_frame_predictions = _write_constant_first_frame_predictions
    _cli.write_first_frame_labels = _write_first_frame_labels
    setattr(_cli, installed_attr, True)


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


_install_lts_input_output_alias_guard()
_install_zero_frame_coverage_guard()
_install_lts_submission_domain_guard()
_install_lts_duplicate_key_validation_guard()
