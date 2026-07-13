"""Compatibility wrapper for strict Multi-UAV LTS ZIP validation.

The maintained implementation lives in the sibling ``cli.py`` module. This
package preserves the public import path while rejecting repeated ZIP member
names, non-root member paths, non-positive class ids, and visibility values
outside ``[0, 1]``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
import sys
import zipfile

_IMPL_PATH = Path(__file__).resolve().parent.parent / "cli.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.multi_uav_lts._cli_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Multi-UAV LTS CLI implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SUBMISSION_VALIDATION = _IMPL.SubmissionValidation
_summarize_prediction_text = _IMPL._summarize_prediction_text


@dataclass(frozen=True)
class SubmissionValidation(_ORIGINAL_SUBMISSION_VALIDATION):
    """Submission validation with strict archive and row-domain diagnostics."""

    duplicate_entries: list[str] = field(default_factory=list)
    invalid_class_rows: int = 0
    invalid_visibility_rows: int = 0


def _count_invalid_class_visibility_rows(text: str) -> tuple[int, int]:
    invalid_class_rows = 0
    invalid_visibility_rows = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(_IMPL.SUBMISSION_COLUMNS):
            continue
        try:
            frame_id = _IMPL._parse_int_like(parts[0])
            object_id = _IMPL._parse_int_like(parts[1])
            x1, y1, width, height, confidence, _class_value, visibility = (
                float(part) for part in parts[2:]
            )
            class_id = _IMPL._parse_int_like(parts[7])
        except ValueError:
            continue
        if frame_id <= 0 or object_id <= 0:
            continue
        if not all(
            math.isfinite(value)
            for value in (x1, y1, width, height, confidence, visibility)
        ):
            continue
        if class_id <= 0:
            invalid_class_rows += 1
        if not 0.0 <= visibility <= 1.0:
            invalid_visibility_rows += 1
    return invalid_class_rows, invalid_visibility_rows


def validate_submission_zip(
    zip_path: Path,
    *,
    template_zip: Path | None = None,
    expected_file_count: int | None = 98,
) -> SubmissionValidation:
    """Validate one submission ZIP, rejecting invalid archive members and rows."""

    expected_names = _IMPL.expected_names_from_template(template_zip)
    if expected_names is not None:
        expected_file_count = len(expected_names)

    invalid_class_rows = 0
    invalid_visibility_rows = 0
    with zipfile.ZipFile(zip_path) as archive:
        physical_names = sorted(
            name for name in archive.namelist() if not name.endswith("/")
        )
        counts = Counter(physical_names)
        duplicate_entries = sorted(name for name, count in counts.items() if count > 1)
        names = sorted(counts)
        nested_entries = [
            name
            for name in names
            if "/" in name.rstrip("/") or "\\" in name
        ]
        non_txt_entries = [name for name in names if not name.endswith(".txt")]
        expected_set = set(expected_names or [])
        name_set = set(names)
        missing_files = sorted(expected_set - name_set)
        extra_files = sorted(name_set - expected_set) if expected_names else []

        file_summaries = []
        for name in names:
            if name in nested_entries or not name.endswith(".txt"):
                continue
            text = archive.read(name).decode("utf-8", errors="replace")
            file_summaries.append(_summarize_prediction_text(name, text))
            class_rows, visibility_rows = _count_invalid_class_visibility_rows(text)
            invalid_class_rows += class_rows
            invalid_visibility_rows += visibility_rows

    parse_errors = sum(summary.parse_errors for summary in file_summaries)
    invalid_geometry_rows = sum(summary.invalid_geometry_rows for summary in file_summaries)
    invalid_confidence_rows = sum(
        summary.invalid_confidence_rows for summary in file_summaries
    )
    unsorted_rows = sum(summary.unsorted_rows for summary in file_summaries)
    total_rows = sum(summary.row_count for summary in file_summaries)

    valid = (
        not duplicate_entries
        and not nested_entries
        and not non_txt_entries
        and not missing_files
        and not extra_files
        and (
            expected_file_count is None
            or len(physical_names) == expected_file_count
        )
        and parse_errors == 0
        and invalid_geometry_rows == 0
        and invalid_confidence_rows == 0
        and invalid_class_rows == 0
        and invalid_visibility_rows == 0
        and unsorted_rows == 0
    )
    return SubmissionValidation(
        zip_path=str(zip_path),
        valid=valid,
        file_count=len(physical_names),
        expected_file_count=expected_file_count,
        missing_files=missing_files,
        extra_files=extra_files,
        nested_entries=nested_entries,
        non_txt_entries=non_txt_entries,
        total_rows=total_rows,
        parse_errors=parse_errors,
        invalid_geometry_rows=invalid_geometry_rows,
        invalid_confidence_rows=invalid_confidence_rows,
        unsorted_rows=unsorted_rows,
        files=file_summaries,
        duplicate_entries=duplicate_entries,
        invalid_class_rows=invalid_class_rows,
        invalid_visibility_rows=invalid_visibility_rows,
    )


_IMPL.SubmissionValidation = SubmissionValidation
_IMPL.validate_submission_zip = validate_submission_zip

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["SubmissionValidation"] = SubmissionValidation
globals()["validate_submission_zip"] = validate_submission_zip
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
