"""Duplicate row audit for Multi-UAV LTS prediction files.

Codabench-style MOT submissions can be format-valid while containing repeated
``(frame_id, object_id)`` rows.  Those duplicates are easy to miss in large
submission folders and can distort local diagnostics or official uploads.  This
module provides a lightweight pre-upload audit for directories or ZIP files.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any
import zipfile

from raft_uav.multi_uav_lts.cli import SUBMISSION_COLUMNS
from raft_uav.numeric import optional_int


@dataclass(frozen=True)
class DuplicatePredictionKeyRow:
    """Duplicate count for one frame/object key in one prediction file."""

    name: str
    frame_id: int
    object_id: int
    occurrence_count: int
    duplicate_rows: int


@dataclass(frozen=True)
class DuplicatePredictionFileSummary:
    """Duplicate summary for one prediction file."""

    name: str
    row_count: int
    parse_errors: int
    duplicate_key_count: int
    duplicate_rows: int


@dataclass(frozen=True)
class DuplicatePredictionAudit:
    """Duplicate-key audit for one prediction directory or ZIP."""

    prediction_path: str
    clean: bool
    file_count: int
    total_rows: int
    parse_errors: int
    duplicate_key_count: int
    duplicate_rows: int
    duplicate_files: list[str]
    files: list[DuplicatePredictionFileSummary]
    duplicate_keys: list[DuplicatePredictionKeyRow]


def audit_duplicate_predictions(prediction_path: Path) -> DuplicatePredictionAudit:
    """Audit duplicate ``(frame_id, object_id)`` keys in every prediction file."""

    prediction_texts = _prediction_texts_for_duplicate_audit(prediction_path)
    file_summaries: list[DuplicatePredictionFileSummary] = []
    duplicate_keys: list[DuplicatePredictionKeyRow] = []
    for name, text in sorted(prediction_texts.items()):
        summary, keys = _audit_prediction_text(name, text)
        file_summaries.append(summary)
        duplicate_keys.extend(keys)
    parse_errors = sum(summary.parse_errors for summary in file_summaries)
    duplicate_rows = sum(summary.duplicate_rows for summary in file_summaries)
    duplicate_key_count = sum(summary.duplicate_key_count for summary in file_summaries)
    total_rows = sum(summary.row_count for summary in file_summaries)
    duplicate_files = [summary.name for summary in file_summaries if summary.duplicate_rows > 0]
    return DuplicatePredictionAudit(
        prediction_path=str(prediction_path),
        clean=parse_errors == 0 and duplicate_rows == 0,
        file_count=len(file_summaries),
        total_rows=total_rows,
        parse_errors=parse_errors,
        duplicate_key_count=duplicate_key_count,
        duplicate_rows=duplicate_rows,
        duplicate_files=duplicate_files,
        files=file_summaries,
        duplicate_keys=duplicate_keys,
    )


def write_duplicate_audit_artifacts(
    audit: DuplicatePredictionAudit,
    *,
    output_json: Path | None = None,
    file_summary_csv: Path | None = None,
    duplicate_keys_csv: Path | None = None,
) -> None:
    """Write duplicate-audit artifacts."""

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(asdict(audit), indent=2, sort_keys=True), encoding="utf-8")
    if file_summary_csv is not None:
        _write_dataclass_csv(
            file_summary_csv,
            DuplicatePredictionFileSummary,
            audit.files,
        )
    if duplicate_keys_csv is not None:
        _write_dataclass_csv(
            duplicate_keys_csv,
            DuplicatePredictionKeyRow,
            audit.duplicate_keys,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.multi_uav_lts.duplicate_audit",
        description="audit duplicate Multi-UAV LTS frame/object prediction keys",
    )
    parser.add_argument("prediction_path", type=Path, help="prediction directory or ZIP")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--file-summary-csv", type=Path)
    parser.add_argument("--duplicate-keys-csv", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args(argv)

    audit = audit_duplicate_predictions(args.prediction_path)
    write_duplicate_audit_artifacts(
        audit,
        output_json=args.output_json,
        file_summary_csv=args.file_summary_csv,
        duplicate_keys_csv=args.duplicate_keys_csv,
    )
    print(json.dumps(asdict(audit), indent=2, sort_keys=True))
    if args.require_clean and not audit.clean:
        raise SystemExit(1)
    return 0


def _prediction_texts_for_duplicate_audit(prediction_path: Path) -> dict[str, str]:
    """Return all .txt prediction payloads that duplicate auditing should inspect."""

    if prediction_path.is_dir():
        base = prediction_path.resolve()
        return {
            _relative_posix_path(path, base): path.read_text(encoding="utf-8")
            for path in sorted(prediction_path.rglob("*.txt"))
            if path.is_file()
        }
    with zipfile.ZipFile(prediction_path) as archive:
        chunks_by_name: dict[str, list[str]] = {}
        for info in archive.infolist():
            if not _is_zip_text_prediction_member(info):
                continue
            name = _normalized_zip_member_name(info.filename)
            chunks_by_name.setdefault(name, []).append(
                archive.read(info).decode("utf-8", errors="replace")
            )
    return {
        name: _join_duplicate_zip_member_payloads(chunks)
        for name, chunks in chunks_by_name.items()
    }


def _join_duplicate_zip_member_payloads(chunks: list[str]) -> str:
    """Combine repeated ZIP members with the same logical prediction filename."""

    if len(chunks) == 1:
        return chunks[0]
    normalized_chunks = [
        chunk if not chunk or chunk.endswith("\n") else f"{chunk}\n"
        for chunk in chunks
    ]
    return "".join(normalized_chunks)


def _is_zip_text_prediction_member(info: zipfile.ZipInfo) -> bool:
    if info.is_dir():
        return False
    member = PurePosixPath(_normalized_zip_member_name(info.filename))
    return member.suffix.lower() == ".txt"


def _relative_posix_path(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base).as_posix()


def _normalized_zip_member_name(name: str) -> str:
    return str(name).replace("\\", "/")


def _audit_prediction_text(
    name: str,
    text: str,
) -> tuple[DuplicatePredictionFileSummary, list[DuplicatePredictionKeyRow]]:
    row_count = 0
    parse_errors = 0
    keys: Counter[tuple[int, int]] = Counter()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        row_count += 1
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(SUBMISSION_COLUMNS):
            parse_errors += 1
            continue
        try:
            frame_id = _parse_lts_key_int(parts[0])
            object_id = _parse_lts_key_int(parts[1])
        except ValueError:
            parse_errors += 1
            continue
        if frame_id <= 0 or object_id <= 0:
            parse_errors += 1
            continue
        keys[(frame_id, object_id)] += 1
    duplicate_rows = sum(count - 1 for count in keys.values() if count > 1)
    duplicate_key_rows = [
        DuplicatePredictionKeyRow(
            name=name,
            frame_id=frame_id,
            object_id=object_id,
            occurrence_count=count,
            duplicate_rows=count - 1,
        )
        for (frame_id, object_id), count in sorted(keys.items())
        if count > 1
    ]
    summary = DuplicatePredictionFileSummary(
        name=name,
        row_count=row_count,
        parse_errors=parse_errors,
        duplicate_key_count=len(duplicate_key_rows),
        duplicate_rows=duplicate_rows,
    )
    return summary, duplicate_key_rows


def _parse_lts_key_int(value: str) -> int:
    parsed = optional_int(value)
    if parsed is None:
        raise ValueError(f"expected integer-like value, got {value!r}")
    return parsed


def _write_dataclass_csv(path: Path, row_type: type[Any], rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_type.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
