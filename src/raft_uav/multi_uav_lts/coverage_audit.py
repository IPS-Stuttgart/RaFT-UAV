"""Coverage audit for Multi-UAV LTS prediction files.

The LTS submission packager can create template-shaped ZIPs by writing empty
files for missing predictions. That behavior is useful for upload safety, but it
can hide runner failures before scoring. This module audits a prediction
folder/ZIP against a template ZIP and/or image sequence root before packaging.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from raft_uav.multi_uav_lts.cli import (
    _parse_int_like,
    _prediction_texts,
    _summarize_prediction_text,
    expected_names_from_template,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PredictionCoverageRow:
    """One expected or observed LTS prediction file."""

    name: str
    expected: bool
    present: bool
    status: str
    row_count: int
    first_frame: int | None
    last_frame: int | None
    expected_frame_count: int | None
    unique_object_ids: int
    parse_errors: int
    invalid_geometry_rows: int
    invalid_confidence_rows: int
    unsorted_rows: int
    out_of_range_frame_rows: int


@dataclass(frozen=True)
class PredictionCoverageAudit:
    """Prediction coverage and formatting audit for one prediction source."""

    prediction_path: str
    template_zip: str | None
    sequence_root: str | None
    ready: bool
    expected_file_count: int
    present_file_count: int
    missing_file_count: int
    extra_file_count: int
    empty_expected_file_count: int
    parse_errors: int
    invalid_geometry_rows: int
    invalid_confidence_rows: int
    unsorted_rows: int
    out_of_range_frame_rows: int
    missing_files: list[str]
    extra_files: list[str]
    empty_expected_files: list[str]
    out_of_range_frame_files: list[str]
    rows: list[PredictionCoverageRow]


def audit_prediction_coverage(
    prediction_path: Path,
    *,
    template_zip: Path | None = None,
    sequence_root: Path | None = None,
) -> PredictionCoverageAudit:
    """Audit prediction files against expected template/sequence names."""

    predictions = _prediction_texts(prediction_path)
    expected_names = _expected_prediction_names(template_zip=template_zip, sequence_root=sequence_root)
    expected_frame_counts = _expected_frame_counts(sequence_root)
    present_names = sorted(predictions)
    expected_set = set(expected_names)
    present_set = set(present_names)
    all_names = sorted(expected_set | present_set)

    rows: list[PredictionCoverageRow] = []
    missing_files: list[str] = []
    extra_files: list[str] = []
    empty_expected_files: list[str] = []
    out_of_range_frame_files: list[str] = []
    for name in all_names:
        expected = name in expected_set
        present = name in present_set
        expected_frame_count = expected_frame_counts.get(name)
        if not present:
            status = "missing"
            missing_files.append(name)
            row = PredictionCoverageRow(
                name=name,
                expected=expected,
                present=False,
                status=status,
                row_count=0,
                first_frame=None,
                last_frame=None,
                expected_frame_count=expected_frame_count,
                unique_object_ids=0,
                parse_errors=0,
                invalid_geometry_rows=0,
                invalid_confidence_rows=0,
                unsorted_rows=0,
                out_of_range_frame_rows=0,
            )
            rows.append(row)
            continue
        summary = _summarize_prediction_text(name, predictions[name])
        out_of_range_frame_rows = _count_out_of_range_frame_rows(
            predictions[name],
            expected_frame_count=expected_frame_count,
        )
        if out_of_range_frame_rows:
            out_of_range_frame_files.append(name)
        if not expected:
            status = "extra"
            extra_files.append(name)
        elif summary.row_count == 0:
            status = "empty_expected"
            empty_expected_files.append(name)
        elif (
            summary.parse_errors
            or summary.invalid_geometry_rows
            or summary.invalid_confidence_rows
            or out_of_range_frame_rows
        ):
            status = "invalid"
        elif summary.unsorted_rows:
            status = "unsorted"
        else:
            status = "ok"
        rows.append(
            PredictionCoverageRow(
                name=name,
                expected=expected,
                present=True,
                status=status,
                row_count=summary.row_count,
                first_frame=summary.first_frame,
                last_frame=summary.last_frame,
                expected_frame_count=expected_frame_count,
                unique_object_ids=summary.unique_object_ids,
                parse_errors=summary.parse_errors,
                invalid_geometry_rows=summary.invalid_geometry_rows,
                invalid_confidence_rows=summary.invalid_confidence_rows,
                unsorted_rows=summary.unsorted_rows,
                out_of_range_frame_rows=out_of_range_frame_rows,
            )
        )

    parse_errors = sum(row.parse_errors for row in rows)
    invalid_geometry_rows = sum(row.invalid_geometry_rows for row in rows)
    invalid_confidence_rows = sum(row.invalid_confidence_rows for row in rows)
    unsorted_rows = sum(row.unsorted_rows for row in rows)
    out_of_range_frame_rows = sum(row.out_of_range_frame_rows for row in rows)
    ready = (
        not missing_files
        and not extra_files
        and not empty_expected_files
        and parse_errors == 0
        and invalid_geometry_rows == 0
        and invalid_confidence_rows == 0
        and unsorted_rows == 0
        and out_of_range_frame_rows == 0
    )
    return PredictionCoverageAudit(
        prediction_path=str(prediction_path),
        template_zip=str(template_zip) if template_zip is not None else None,
        sequence_root=str(sequence_root) if sequence_root is not None else None,
        ready=ready,
        expected_file_count=len(expected_names),
        present_file_count=len(present_names),
        missing_file_count=len(missing_files),
        extra_file_count=len(extra_files),
        empty_expected_file_count=len(empty_expected_files),
        parse_errors=parse_errors,
        invalid_geometry_rows=invalid_geometry_rows,
        invalid_confidence_rows=invalid_confidence_rows,
        unsorted_rows=unsorted_rows,
        out_of_range_frame_rows=out_of_range_frame_rows,
        missing_files=missing_files,
        extra_files=extra_files,
        empty_expected_files=empty_expected_files,
        out_of_range_frame_files=sorted(set(out_of_range_frame_files)),
        rows=rows,
    )


def write_prediction_coverage_artifacts(
    audit: PredictionCoverageAudit,
    *,
    output_json: Path | None = None,
    row_csv: Path | None = None,
) -> None:
    """Write coverage audit artifacts."""

    payload = asdict(audit)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if row_csv is not None:
        row_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(PredictionCoverageRow.__dataclass_fields__)
        with row_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in audit.rows:
                writer.writerow(asdict(row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-multi-uav-lts-coverage-audit",
        description="audit Multi-UAV LTS prediction coverage before packaging",
    )
    parser.add_argument("prediction_path", type=Path, help="prediction directory or ZIP")
    parser.add_argument("--template-zip", type=Path)
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--row-csv", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)

    audit = audit_prediction_coverage(
        args.prediction_path,
        template_zip=args.template_zip,
        sequence_root=args.sequence_root,
    )
    write_prediction_coverage_artifacts(audit, output_json=args.output_json, row_csv=args.row_csv)
    payload: dict[str, Any] = asdict(audit)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_ready and not audit.ready:
        raise SystemExit(1)
    return 0


def _expected_prediction_names(
    *,
    template_zip: Path | None,
    sequence_root: Path | None,
) -> list[str]:
    names: set[str] = set()
    names.update(_template_prediction_names(template_zip))
    if sequence_root is not None:
        names.update(f"{path.name}.txt" for path in sorted(sequence_root.iterdir()) if path.is_dir())
    if not names:
        raise ValueError("provide --template-zip and/or --sequence-root for coverage auditing")
    return sorted(names)


def _template_prediction_names(template_zip: Path | None) -> list[str]:
    names = expected_names_from_template(template_zip) or []
    return sorted(name for name in names if name.endswith(".txt") and "/" not in name.rstrip("/"))


def _expected_frame_counts(sequence_root: Path | None) -> dict[str, int]:
    if sequence_root is None:
        return {}
    counts: dict[str, int] = {}
    for sequence_dir in sorted(sequence_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        counts[f"{sequence_dir.name}.txt"] = sum(
            1
            for path in sequence_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return counts


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
