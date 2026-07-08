"""Non-maximum suppression repair for Multi-UAV LTS submissions.

Codabench Multi-UAV LTS predictions can be structurally valid while containing
near-duplicate boxes in the same frame.  Duplicate high-overlap detections are
usually counted as false positives by MOT-style evaluators.  This module applies
an inference-safe per-frame NMS pass to prediction directories or ZIP files,
writes normalized root-level prediction files, and optionally packages and
validates the repaired submission against an official template ZIP.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

from raft_uav.multi_uav_lts.cli import LtsDetection
from raft_uav.multi_uav_lts.cli import _box_iou
from raft_uav.multi_uav_lts.cli import _format_lts_detection
from raft_uav.multi_uav_lts.cli import _load_lts_detection_rows
from raft_uav.multi_uav_lts.cli import expected_names_from_template
from raft_uav.multi_uav_lts.cli import package_submission
from raft_uav.multi_uav_lts.cli import validate_submission_zip


@dataclass(frozen=True)
class NmsFileSummary:
    """NMS summary for one prediction file."""

    name: str
    input_rows: int
    output_rows: int
    suppressed_rows: int
    parse_errors: int


@dataclass(frozen=True)
class NmsRepairSummary:
    """NMS summary for one prediction directory or ZIP."""

    prediction_path: str
    output_dir: str
    output_zip: str | None
    iou_threshold: float
    class_aware: bool
    file_count: int
    input_rows: int
    output_rows: int
    suppressed_rows: int
    parse_errors: int
    valid_zip: bool | None
    files: list[NmsFileSummary]


SUMMARY_JSON = "multi_uav_lts_nms_summary.json"
FILE_SUMMARY_CSV = "multi_uav_lts_nms_file_summary.csv"


def apply_lts_nms_to_text(
    text: str,
    *,
    iou_threshold: float = 0.9,
    class_aware: bool = True,
) -> tuple[str, NmsFileSummary]:
    """Apply per-frame NMS to one LTS prediction text payload."""

    try:
        rows = _load_lts_detection_rows(text)
    except ValueError:
        return "", NmsFileSummary(
            name="<memory>",
            input_rows=0,
            output_rows=0,
            suppressed_rows=0,
            parse_errors=1,
        )
    kept = apply_lts_nms_to_rows(
        rows,
        iou_threshold=iou_threshold,
        class_aware=class_aware,
    )
    kept = sorted(kept, key=lambda row: (row.frame_id, row.object_id, -row.confidence))
    output = "\n".join(_format_lts_detection(row) for row in kept)
    if output:
        output += "\n"
    return output, NmsFileSummary(
        name="<memory>",
        input_rows=len(rows),
        output_rows=len(kept),
        suppressed_rows=len(rows) - len(kept),
        parse_errors=0,
    )


def apply_lts_nms_to_rows(
    rows: list[LtsDetection],
    *,
    iou_threshold: float = 0.9,
    class_aware: bool = True,
) -> list[LtsDetection]:
    """Return rows after per-frame, optionally class-aware NMS."""

    iou_threshold = float(iou_threshold)
    if not np.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and in [0, 1]")
    by_group: dict[tuple[int, int | None], list[LtsDetection]] = {}
    for row in rows:
        key = (int(row.frame_id), int(row.class_id) if class_aware else None)
        by_group.setdefault(key, []).append(row)
    kept: list[LtsDetection] = []
    for group_rows in by_group.values():
        kept.extend(_nms_group(group_rows, iou_threshold=iou_threshold))
    return kept


def repair_lts_submission_with_nms(
    prediction_path: Path,
    output_dir: Path,
    *,
    output_zip: Path | None = None,
    template_zip: Path | None = None,
    iou_threshold: float = 0.9,
    class_aware: bool = True,
    require_valid: bool = False,
) -> NmsRepairSummary:
    """Write an NMS-repaired prediction directory and optional upload ZIP."""

    payloads = _prediction_texts(prediction_path)
    expected_names = expected_names_from_template(template_zip)
    if expected_names is not None:
        payloads = {name: payloads.get(name, "") for name in expected_names}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    file_summaries: list[NmsFileSummary] = []
    for name, text in sorted(payloads.items()):
        repaired_text, summary = apply_lts_nms_to_text(
            text,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
        )
        output_path = output / PurePosixPath(name).name
        output_path.write_text(repaired_text, encoding="utf-8")
        file_summaries.append(
            NmsFileSummary(
                name=PurePosixPath(name).name,
                input_rows=summary.input_rows,
                output_rows=summary.output_rows,
                suppressed_rows=summary.suppressed_rows,
                parse_errors=summary.parse_errors,
            )
        )

    validation_valid: bool | None = None
    if output_zip is not None:
        validation = package_submission(
            output,
            output_zip,
            template_zip=template_zip,
            normalize=True,
            sort_rows=True,
        )
        validation_valid = bool(validation.valid)
        if require_valid and not validation.valid:
            raise SystemExit("NMS-repaired Multi-UAV LTS ZIP is not valid")
    parse_errors = sum(item.parse_errors for item in file_summaries)
    summary = NmsRepairSummary(
        prediction_path=str(prediction_path),
        output_dir=str(output),
        output_zip=str(output_zip) if output_zip is not None else None,
        iou_threshold=float(iou_threshold),
        class_aware=bool(class_aware),
        file_count=len(file_summaries),
        input_rows=sum(item.input_rows for item in file_summaries),
        output_rows=sum(item.output_rows for item in file_summaries),
        suppressed_rows=sum(item.suppressed_rows for item in file_summaries),
        parse_errors=parse_errors,
        valid_zip=validation_valid,
        files=file_summaries,
    )
    _write_summary_artifacts(summary, output)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-multi-uav-lts-nms",
        description="apply per-frame NMS to a Multi-UAV LTS prediction directory or ZIP",
    )
    parser.add_argument("prediction_path", type=Path, help="prediction directory or ZIP")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path)
    parser.add_argument("--template-zip", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.9)
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument("--require-valid", action="store_true")
    args = parser.parse_args(argv)

    summary = repair_lts_submission_with_nms(
        args.prediction_path,
        args.output_dir,
        output_zip=args.output_zip,
        template_zip=args.template_zip,
        iou_threshold=float(args.iou_threshold),
        class_aware=not bool(args.class_agnostic),
        require_valid=bool(args.require_valid),
    )
    print(json.dumps(_summary_to_jsonable(summary), indent=2, sort_keys=True))
    return 0


def _nms_group(rows: list[LtsDetection], *, iou_threshold: float) -> list[LtsDetection]:
    ordered = sorted(rows, key=lambda row: (-float(row.confidence), row.object_id, row.x1, row.y1))
    kept: list[LtsDetection] = []
    for row in ordered:
        if any(_box_iou(row, kept_row) >= iou_threshold for kept_row in kept):
            continue
        kept.append(row)
    return kept


def _prediction_texts(prediction_path: Path) -> dict[str, str]:
    path = Path(prediction_path)
    if path.is_dir():
        return {
            item.name: item.read_text(encoding="utf-8")
            for item in sorted(path.glob("*.txt"))
            if item.is_file()
        }
    with zipfile.ZipFile(path) as archive:
        return {
            PurePosixPath(name).name: archive.read(name).decode("utf-8", errors="replace")
            for name in sorted(archive.namelist())
            if name.endswith(".txt") and not name.endswith("/")
        }


def _write_summary_artifacts(summary: NmsRepairSummary, output_dir: Path) -> None:
    output = Path(output_dir)
    (output / SUMMARY_JSON).write_text(
        json.dumps(_summary_to_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output / FILE_SUMMARY_CSV).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(NmsFileSummary.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.files:
            writer.writerow(asdict(row))


def _summary_to_jsonable(summary: NmsRepairSummary) -> dict[str, object]:
    payload = asdict(summary)
    payload["files"] = [asdict(item) for item in summary.files]
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
