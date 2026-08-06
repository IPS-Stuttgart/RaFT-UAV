"""Compatibility wrapper for strict Multi-UAV LTS validation and scoring.

The maintained implementation lives in the sibling ``cli.py`` module. This
package preserves the public import path while rejecting repeated ZIP member
names, non-root member paths, non-positive class ids, visibility values outside
``[0, 1]``, unsafe template members used during packaging, and invalid IoU
thresholds used by the diagnostic scorer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import importlib.util
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

from raft_uav.numeric import optional_float

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
_ORIGINAL_SCORE_LTS_PREDICTIONS = _IMPL.score_lts_predictions
_summarize_prediction_text = _IMPL._summarize_prediction_text


def _parse_int_like_exact(value: str) -> int:
    """Parse an integer-like submission ID without binary-float rounding."""

    text = str(value).strip()
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"expected integer-like value, got {value!r}") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"expected integer-like value, got {value!r}")
    return int(parsed)


def _normalize_iou_threshold(value: object) -> float:
    """Return a finite scalar IoU threshold in the closed unit interval."""

    threshold = optional_float(value)
    if threshold is None or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be a finite real scalar in [0, 1]")
    return threshold


def _validate_truth_selection(
    truth_dir: Path,
    sequences: list[str] | None,
) -> None:
    """Reject missing truth data and requested sequence names that cannot be scored."""

    truth_root = Path(truth_dir)
    if not truth_root.is_dir():
        raise FileNotFoundError(f"truth directory does not exist: {truth_root}")

    available = {
        path.stem
        for path in truth_root.glob("*.txt")
        if path.is_file()
    }
    if not available:
        raise ValueError(f"truth directory contains no .txt sequence files: {truth_root}")

    if sequences:
        unknown = sorted(set(sequences) - available)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"requested truth sequences are unavailable: {names}")


def _match_rows_by_iou(truth, predictions, *, iou_threshold):
    """Maximize valid match count first, then total IoU."""

    threshold = _normalize_iou_threshold(iou_threshold)
    if not truth or not predictions:
        return []
    iou = np.asarray(
        [
            [_IMPL._box_iou(gt_row, pred_row) for pred_row in predictions]
            for gt_row in truth
        ],
        dtype=float,
    )
    valid = iou >= threshold
    if not np.any(valid):
        return []

    cardinality_bonus = float(min(iou.shape) + 1)
    benefit = np.where(valid, cardinality_bonus + iou, 0.0)
    gt_indices, pred_indices = linear_sum_assignment(benefit, maximize=True)
    matches = [
        (int(gt_index), int(pred_index), float(iou[gt_index, pred_index]))
        for gt_index, pred_index in zip(gt_indices, pred_indices, strict=True)
        if valid[gt_index, pred_index]
    ]
    matches.sort(key=lambda match: (match[0], match[1]))
    return matches


def score_lts_predictions(
    prediction_path: Path,
    truth_dir: Path,
    *,
    iou_threshold: object = 0.5,
    sequences: list[str] | None = None,
):
    """Build an LTS scorecard with validated truth inputs and IoU threshold."""

    threshold = _normalize_iou_threshold(iou_threshold)
    _validate_truth_selection(truth_dir, sequences)
    return _ORIGINAL_SCORE_LTS_PREDICTIONS(
        prediction_path,
        truth_dir,
        iou_threshold=threshold,
        sequences=sequences,
    )


_IMPL._parse_int_like = _parse_int_like_exact
_IMPL._match_rows_by_iou = _match_rows_by_iou
_IMPL.score_lts_predictions = score_lts_predictions


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


def _validated_template_member_names(template_zip: Path | None) -> list[str] | None:
    """Return safe root-level prediction members before touching output files."""

    names = _IMPL.expected_names_from_template(template_zip)
    if names is None:
        return None

    counts = Counter(names)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    invalid = sorted(
        {
            name
            for name in names
            if (
                not name.endswith(".txt")
                or PurePosixPath(name).name != name
                or PureWindowsPath(name).name != name
            )
        }
    )
    if duplicates or invalid:
        details: list[str] = []
        if duplicates:
            details.append(f"duplicate members={duplicates[:5]}")
        if invalid:
            details.append(f"unsafe or non-text members={invalid[:5]}")
        message = "template ZIP contains unsupported prediction members: " + "; ".join(details)
        raise ValueError(message)
    return names


def package_submission(
    prediction_dir: Path,
    output_zip: Path,
    *,
    template_zip: Path | None = None,
    normalize: bool = False,
    sort_rows: bool = False,
) -> SubmissionValidation:
    """Package predictions without allowing template-controlled file traversal."""

    expected_names = _validated_template_member_names(template_zip)
    names = expected_names or sorted(path.name for path in prediction_dir.glob("*.txt"))
    if output_zip.exists():
        output_zip.unlink()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            source = prediction_dir / name
            if not source.exists():
                archive.writestr(name, "")
            elif normalize or sort_rows:
                archive.writestr(
                    name,
                    _IMPL.normalize_prediction_text(
                        source.read_text(encoding="utf-8"),
                        sort_rows=sort_rows,
                    ),
                )
            else:
                archive.write(source, arcname=name)
    expected_file_count = None if template_zip is not None else len(names)
    return validate_submission_zip(
        output_zip,
        template_zip=template_zip,
        expected_file_count=expected_file_count,
    )


_IMPL.SubmissionValidation = SubmissionValidation
_IMPL.validate_submission_zip = validate_submission_zip
_IMPL.package_submission = package_submission


globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["SubmissionValidation"] = SubmissionValidation
globals()["validate_submission_zip"] = validate_submission_zip
globals()["package_submission"] = package_submission
globals()["_validated_template_member_names"] = _validated_template_member_names
globals()["_normalize_iou_threshold"] = _normalize_iou_threshold
globals()["_validate_truth_selection"] = _validate_truth_selection
globals()["_match_rows_by_iou"] = _match_rows_by_iou
globals()["score_lts_predictions"] = score_lts_predictions
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
