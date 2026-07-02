from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


SUBMISSION_COLUMNS = (
    "frame_id",
    "object_id",
    "x1",
    "y1",
    "w",
    "h",
    "confidence",
    "class_id",
    "visibility",
)


@dataclass(frozen=True)
class SubmissionFileSummary:
    name: str
    row_count: int
    first_frame: int | None
    last_frame: int | None
    unique_object_ids: int
    parse_errors: int
    invalid_geometry_rows: int
    invalid_confidence_rows: int
    unsorted_rows: int


@dataclass(frozen=True)
class SubmissionValidation:
    zip_path: str
    valid: bool
    file_count: int
    expected_file_count: int | None
    missing_files: list[str]
    extra_files: list[str]
    nested_entries: list[str]
    non_txt_entries: list[str]
    total_rows: int
    parse_errors: int
    invalid_geometry_rows: int
    invalid_confidence_rows: int
    unsorted_rows: int
    files: list[SubmissionFileSummary]


@dataclass(frozen=True)
class LtsDetection:
    frame_id: int
    object_id: int
    x1: float
    y1: float
    w: float
    h: float
    confidence: float
    class_id: int
    visibility: float


@dataclass(frozen=True)
class LtsSequenceScore:
    sequence: str
    frame_count: int
    gt_detections: int
    predicted_detections: int
    matches: int
    false_positives: int
    false_negatives: int
    id_switches: int
    mota_like: float | None
    precision: float | None
    recall: float | None
    mean_matched_iou: float | None
    id_switches_per_match: float | None


@dataclass(frozen=True)
class LtsScorecard:
    prediction_path: str
    truth_dir: str
    iou_threshold: float
    sequence_count: int
    gt_detections: int
    predicted_detections: int
    matches: int
    false_positives: int
    false_negatives: int
    id_switches: int
    mota_like: float | None
    precision: float | None
    recall: float | None
    mean_matched_iou: float | None
    id_switches_per_match: float | None
    sequences: list[LtsSequenceScore]


def expected_names_from_template(template_zip: Path | None) -> list[str] | None:
    if template_zip is None:
        return None
    with zipfile.ZipFile(template_zip) as archive:
        return sorted(name for name in archive.namelist() if not name.endswith("/"))


def validate_submission_zip(
    zip_path: Path,
    *,
    template_zip: Path | None = None,
    expected_file_count: int | None = 98,
) -> SubmissionValidation:
    expected_names = expected_names_from_template(template_zip)
    if expected_names is not None:
        expected_file_count = len(expected_names)

    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        nested_entries = [name for name in names if "/" in name.rstrip("/")]
        non_txt_entries = [name for name in names if not name.endswith(".txt")]
        missing_files = sorted(set(expected_names or []) - set(names))
        extra_files = sorted(set(names) - set(expected_names or [])) if expected_names else []

        file_summaries = []
        for name in names:
            if name in nested_entries or not name.endswith(".txt"):
                continue
            text = archive.read(name).decode("utf-8", errors="replace")
            file_summaries.append(_summarize_prediction_text(name, text))

    parse_errors = sum(summary.parse_errors for summary in file_summaries)
    invalid_geometry_rows = sum(summary.invalid_geometry_rows for summary in file_summaries)
    invalid_confidence_rows = sum(summary.invalid_confidence_rows for summary in file_summaries)
    unsorted_rows = sum(summary.unsorted_rows for summary in file_summaries)
    total_rows = sum(summary.row_count for summary in file_summaries)

    valid = (
        not nested_entries
        and not non_txt_entries
        and not missing_files
        and not extra_files
        and (expected_file_count is None or len(names) == expected_file_count)
        and parse_errors == 0
        and invalid_geometry_rows == 0
        and invalid_confidence_rows == 0
        and unsorted_rows == 0
    )
    return SubmissionValidation(
        zip_path=str(zip_path),
        valid=valid,
        file_count=len(names),
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
    )


def _summarize_prediction_text(name: str, text: str) -> SubmissionFileSummary:
    row_count = 0
    parse_errors = 0
    invalid_geometry_rows = 0
    invalid_confidence_rows = 0
    unsorted_rows = 0
    first_frame: int | None = None
    last_frame: int | None = None
    previous_key: tuple[int, int] | None = None
    object_ids: set[int] = set()

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
            frame_id = _parse_int_like(parts[0])
            object_id = _parse_int_like(parts[1])
            x1, y1, w, h, confidence, _class_id, visibility = (float(part) for part in parts[2:])
            _parse_int_like(parts[7])
        except ValueError:
            parse_errors += 1
            continue

        if frame_id <= 0 or object_id <= 0:
            parse_errors += 1
            continue
        if not all(math.isfinite(value) for value in (x1, y1, w, h, confidence, visibility)):
            parse_errors += 1
            continue
        if w <= 0.0 or h <= 0.0:
            invalid_geometry_rows += 1
        if not -1.0 <= confidence <= 1.0:
            invalid_confidence_rows += 1

        first_frame = frame_id if first_frame is None else min(first_frame, frame_id)
        last_frame = frame_id if last_frame is None else max(last_frame, frame_id)
        object_ids.add(object_id)
        key = (frame_id, object_id)
        if previous_key is not None and key < previous_key:
            unsorted_rows += 1
        previous_key = key

    return SubmissionFileSummary(
        name=name,
        row_count=row_count,
        first_frame=first_frame,
        last_frame=last_frame,
        unique_object_ids=len(object_ids),
        parse_errors=parse_errors,
        invalid_geometry_rows=invalid_geometry_rows,
        invalid_confidence_rows=invalid_confidence_rows,
        unsorted_rows=unsorted_rows,
    )


def write_first_frame_labels(
    truth_dir: Path,
    output_dir: Path,
    *,
    frame_id: int = 1,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total_rows = 0
    for truth_path in sorted(truth_dir.glob("*.txt")):
        rows = [
            row
            for row in _load_lts_detection_rows(truth_path.read_text(encoding="utf-8"))
            if row.frame_id == frame_id
        ]
        output_path = output_dir / truth_path.name
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_format_lts_detection(row) + "\n")
        total_rows += len(rows)
        records.append(
            {
                "sequence": truth_path.stem,
                "frame_id": frame_id,
                "row_count": len(rows),
                "output_file": str(output_path),
            }
        )
    return {
        "truth_dir": str(truth_dir),
        "output_dir": str(output_dir),
        "frame_id": frame_id,
        "sequence_count": len(records),
        "total_rows": total_rows,
        "sequences": records,
    }


def package_submission(
    prediction_dir: Path,
    output_zip: Path,
    *,
    template_zip: Path | None = None,
    normalize: bool = False,
    sort_rows: bool = False,
) -> SubmissionValidation:
    expected_names = expected_names_from_template(template_zip)
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
                    normalize_prediction_text(
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


def score_lts_predictions(
    prediction_path: Path,
    truth_dir: Path,
    *,
    iou_threshold: float = 0.5,
    sequences: list[str] | None = None,
) -> LtsScorecard:
    prediction_files = _prediction_texts(prediction_path)
    requested = set(sequences or [])
    truth_paths = sorted(truth_dir.glob("*.txt"))
    sequence_scores = []
    for truth_path in truth_paths:
        sequence = truth_path.stem
        if requested and sequence not in requested:
            continue
        prediction_text = prediction_files.get(f"{sequence}.txt", "")
        truth_text = truth_path.read_text(encoding="utf-8")
        sequence_scores.append(
            _score_lts_sequence(
                sequence,
                prediction_text,
                truth_text,
                iou_threshold=iou_threshold,
            )
        )
    gt_count = sum(score.gt_detections for score in sequence_scores)
    pred_count = sum(score.predicted_detections for score in sequence_scores)
    matches = sum(score.matches for score in sequence_scores)
    false_positives = sum(score.false_positives for score in sequence_scores)
    false_negatives = sum(score.false_negatives for score in sequence_scores)
    id_switches = sum(score.id_switches for score in sequence_scores)
    weighted_iou_sum = sum(
        score.matches * (score.mean_matched_iou or 0.0) for score in sequence_scores
    )
    return LtsScorecard(
        prediction_path=str(prediction_path),
        truth_dir=str(truth_dir),
        iou_threshold=iou_threshold,
        sequence_count=len(sequence_scores),
        gt_detections=gt_count,
        predicted_detections=pred_count,
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        id_switches=id_switches,
        mota_like=_safe_ratio(gt_count - false_positives - false_negatives - id_switches, gt_count),
        precision=_safe_ratio(matches, pred_count),
        recall=_safe_ratio(matches, gt_count),
        mean_matched_iou=_safe_ratio(weighted_iou_sum, matches),
        id_switches_per_match=_safe_ratio(id_switches, matches),
        sequences=sequence_scores,
    )


def _prediction_texts(prediction_path: Path) -> dict[str, str]:
    if prediction_path.is_dir():
        return {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted(prediction_path.glob("*.txt"))
        }
    with zipfile.ZipFile(prediction_path) as archive:
        return {
            name: archive.read(name).decode("utf-8", errors="replace")
            for name in archive.namelist()
            if name.endswith(".txt") and "/" not in name.rstrip("/")
        }


def _score_lts_sequence(
    sequence: str,
    prediction_text: str,
    truth_text: str,
    *,
    iou_threshold: float,
) -> LtsSequenceScore:
    predictions = _rows_by_frame(_load_lts_detection_rows(prediction_text))
    truth = _rows_by_frame(_load_lts_detection_rows(truth_text))
    frames = sorted(set(predictions) | set(truth))
    matches = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    iou_sum = 0.0
    previous_match_by_gt_id: dict[int, int] = {}

    for frame in frames:
        gt_rows = truth.get(frame, [])
        pred_rows = predictions.get(frame, [])
        frame_matches = _match_rows_by_iou(gt_rows, pred_rows, iou_threshold=iou_threshold)
        matches += len(frame_matches)
        false_negatives += len(gt_rows) - len(frame_matches)
        false_positives += len(pred_rows) - len(frame_matches)
        for gt_index, pred_index, iou in frame_matches:
            gt_id = gt_rows[gt_index].object_id
            pred_id = pred_rows[pred_index].object_id
            previous_pred_id = previous_match_by_gt_id.get(gt_id)
            if previous_pred_id is not None and previous_pred_id != pred_id:
                id_switches += 1
            previous_match_by_gt_id[gt_id] = pred_id
            iou_sum += iou

    gt_count = sum(len(rows) for rows in truth.values())
    pred_count = sum(len(rows) for rows in predictions.values())
    return LtsSequenceScore(
        sequence=sequence,
        frame_count=len(frames),
        gt_detections=gt_count,
        predicted_detections=pred_count,
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        id_switches=id_switches,
        mota_like=_safe_ratio(gt_count - false_positives - false_negatives - id_switches, gt_count),
        precision=_safe_ratio(matches, pred_count),
        recall=_safe_ratio(matches, gt_count),
        mean_matched_iou=_safe_ratio(iou_sum, matches),
        id_switches_per_match=_safe_ratio(id_switches, matches),
    )


def _load_lts_detection_rows(text: str) -> list[LtsDetection]:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(SUBMISSION_COLUMNS):
            raise ValueError(f"expected {len(SUBMISSION_COLUMNS)} columns, got {len(parts)}")
        rows.append(
            LtsDetection(
                frame_id=_parse_int_like(parts[0]),
                object_id=_parse_int_like(parts[1]),
                x1=float(parts[2]),
                y1=float(parts[3]),
                w=float(parts[4]),
                h=float(parts[5]),
                confidence=float(parts[6]),
                class_id=_parse_int_like(parts[7]),
                visibility=float(parts[8]),
            )
        )
    return rows


def _rows_by_frame(rows: list[LtsDetection]) -> dict[int, list[LtsDetection]]:
    by_frame: dict[int, list[LtsDetection]] = {}
    for row in rows:
        by_frame.setdefault(row.frame_id, []).append(row)
    return by_frame


def _match_rows_by_iou(
    truth: list[LtsDetection],
    predictions: list[LtsDetection],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    pairs = [
        (gt_index, pred_index, _box_iou(gt_row, pred_row))
        for gt_index, gt_row in enumerate(truth)
        for pred_index, pred_row in enumerate(predictions)
    ]
    pairs = [pair for pair in pairs if pair[2] >= iou_threshold]
    pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches = []
    for gt_index, pred_index, iou in pairs:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def _box_iou(left: LtsDetection, right: LtsDetection) -> float:
    left_x2 = left.x1 + left.w
    left_y2 = left.y1 + left.h
    right_x2 = right.x1 + right.w
    right_y2 = right.y1 + right.h
    inter_w = max(0.0, min(left_x2, right_x2) - max(left.x1, right.x1))
    inter_h = max(0.0, min(left_y2, right_y2) - max(left.y1, right.y1))
    intersection = inter_w * inter_h
    union = left.w * left.h + right.w * right.h - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_lts_detection(row: LtsDetection) -> str:
    values: list[float | int] = [
        row.frame_id,
        row.object_id,
        row.x1,
        row.y1,
        row.w,
        row.h,
        row.confidence,
        row.class_id,
        row.visibility,
    ]
    return ",".join(_format_submission_value(value) for value in values)


def normalize_prediction_text(text: str, *, sort_rows: bool = False) -> str:
    rows: list[tuple[int, int, list[float | int]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(SUBMISSION_COLUMNS):
            raise ValueError(f"expected {len(SUBMISSION_COLUMNS)} columns, got {len(parts)}")
        frame_id = _parse_int_like(parts[0])
        object_id = _parse_int_like(parts[1])
        class_id = _parse_int_like(parts[7])
        values: list[float | int] = [
            frame_id,
            object_id,
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
            class_id,
            float(parts[8]),
        ]
        rows.append((frame_id, object_id, values))
    if sort_rows:
        rows.sort(key=lambda row: (row[0], row[1]))
    lines = [",".join(_format_submission_value(value) for value in values) for _, _, values in rows]
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_int_like(value: str) -> int:
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"expected integer-like value, got {value!r}")
    return int(parsed)


def write_constant_first_frame_predictions(
    sequence_root: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dirs = sorted(path for path in sequence_root.iterdir() if path.is_dir())
    records = []
    total_rows = 0
    for sequence_dir in sequence_dirs:
        sequence = sequence_dir.name
        frames = sorted(
            path
            for path in sequence_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        template_rows = _load_first_frame_rows(first_frame_label_dir / f"{sequence}.txt")
        prediction_path = output_dir / f"{sequence}.txt"
        row_count = 0
        with prediction_path.open("w", encoding="utf-8") as handle:
            for frame_number in range(1, len(frames) + 1):
                for row in template_rows:
                    values = [frame_number, *row[1:]]
                    handle.write(
                        ",".join(_format_submission_value(value) for value in values) + "\n"
                    )
                    row_count += 1
        total_rows += row_count
        records.append(
            {
                "sequence": sequence,
                "frame_count": len(frames),
                "first_frame_objects": len(template_rows),
                "row_count": row_count,
                "prediction_file": str(prediction_path),
            }
        )
    return {
        "sequence_root": str(sequence_root),
        "first_frame_label_dir": str(first_frame_label_dir),
        "output_dir": str(output_dir),
        "sequence_count": len(records),
        "total_rows": total_rows,
        "sequences": records,
    }


def _load_first_frame_rows(path: Path) -> list[list[float | int]]:
    if not path.exists():
        return []
    rows: list[list[float | int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(SUBMISSION_COLUMNS):
            raise ValueError(f"{path} has {len(parts)} columns, expected {len(SUBMISSION_COLUMNS)}")
        frame_id = _parse_int_like(parts[0])
        object_id = _parse_int_like(parts[1])
        values: list[float | int] = [
            frame_id,
            object_id,
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
            _parse_int_like(parts[7]),
            float(parts[8]),
        ]
        rows.append(values)
    return rows


def _format_submission_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def inventory_path(path: Path) -> dict[str, object]:
    if path.is_dir():
        entries = [item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()]
        source_type = "directory"
        size_bytes = sum((path / entry).stat().st_size for entry in entries)
    else:
        with zipfile.ZipFile(path) as archive:
            entries = [name for name in archive.namelist() if not name.endswith("/")]
            size_bytes = path.stat().st_size
        source_type = "zip"

    suffix_counts = Counter(Path(entry).suffix.lower() or "<none>" for entry in entries)
    top_level_counts = Counter(entry.split("/", 1)[0] for entry in entries if entry)
    scenario_counts = Counter(Path(entry).stem.split("_", 1)[0] for entry in entries if entry)
    return {
        "path": str(path),
        "source_type": source_type,
        "size_bytes": size_bytes,
        "file_count": len(entries),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "top_level_counts": dict(sorted(top_level_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "sample_entries": sorted(entries)[:200],
    }


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_file_summary_csv(validation: SubmissionValidation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(SubmissionFileSummary.__dataclass_fields__)
    lines = [",".join(header)]
    for summary in validation.files:
        values = [getattr(summary, field) for field in header]
        lines.append(",".join("" if value is None else str(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sequence_score_csv(scorecard: LtsScorecard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(LtsSequenceScore.__dataclass_fields__)
    lines = [",".join(header)]
    for score in scorecard.sequences:
        values = [getattr(score, field) for field in header]
        lines.append(",".join("" if value is None else str(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raft-uav-multi-uav-lts",
        description=(
            "Support utilities for the Beyond Strong Baseline Multi-UAV Tracking "
            "LTS benchmark."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="inspect a benchmark ZIP or directory")
    inventory.add_argument("path", type=Path)
    inventory.add_argument("--output-json", type=Path)

    validate = subparsers.add_parser("validate-submission", help="validate an LTS submission ZIP")
    validate.add_argument("submission_zip", type=Path)
    validate.add_argument("--template-zip", type=Path)
    validate.add_argument("--expected-file-count", type=int, default=98)
    validate.add_argument("--output-json", type=Path)
    validate.add_argument("--file-summary-csv", type=Path)

    score = subparsers.add_parser(
        "score-predictions",
        help="compute a local MOTA-like scorecard against full train labels",
    )
    score.add_argument("prediction_path", type=Path, help="prediction directory or submission ZIP")
    score.add_argument("--truth-dir", type=Path, required=True)
    score.add_argument("--iou-threshold", type=float, default=0.5)
    score.add_argument("--sequences", nargs="*", help="optional sequence names to score")
    score.add_argument("--output-json", type=Path)
    score.add_argument("--sequence-summary-csv", type=Path)

    package = subparsers.add_parser(
        "package-submission",
        help="package root-level .txt predictions",
    )
    package.add_argument("prediction_dir", type=Path)
    package.add_argument("--output-zip", type=Path, required=True)
    package.add_argument("--template-zip", type=Path)
    package.add_argument("--normalize", action="store_true")
    package.add_argument("--sort-rows", action="store_true")
    package.add_argument("--output-json", type=Path)
    package.add_argument("--file-summary-csv", type=Path)

    constant = subparsers.add_parser(
        "constant-first-frame",
        help="repeat provided first-frame labels over all frames as a sanity baseline",
    )
    constant.add_argument("--sequence-root", type=Path, required=True)
    constant.add_argument("--first-frame-label-dir", type=Path, required=True)
    constant.add_argument("--prediction-dir", type=Path, required=True)
    constant.add_argument("--template-zip", type=Path)
    constant.add_argument("--output-zip", type=Path)
    constant.add_argument("--output-json", type=Path)
    constant.add_argument("--file-summary-csv", type=Path)

    first_frame = subparsers.add_parser(
        "first-frame-labels",
        help="extract first-frame-only labels from full train labels for tracker initialization",
    )
    first_frame.add_argument("--truth-dir", type=Path, required=True)
    first_frame.add_argument("--output-dir", type=Path, required=True)
    first_frame.add_argument("--frame-id", type=int, default=1)
    first_frame.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "inventory":
        summary = inventory_path(args.path)
        if args.output_json:
            _write_json(summary, args.output_json)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "validate-submission":
        validation = validate_submission_zip(
            args.submission_zip,
            template_zip=args.template_zip,
            expected_file_count=args.expected_file_count,
        )
        payload = asdict(validation)
        if args.output_json:
            _write_json(payload, args.output_json)
        if args.file_summary_csv:
            _write_file_summary_csv(validation, args.file_summary_csv)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not validation.valid:
            raise SystemExit(1)
        return

    if args.command == "score-predictions":
        scorecard = score_lts_predictions(
            args.prediction_path,
            args.truth_dir,
            iou_threshold=args.iou_threshold,
            sequences=args.sequences,
        )
        payload = asdict(scorecard)
        if args.output_json:
            _write_json(payload, args.output_json)
        if args.sequence_summary_csv:
            _write_sequence_score_csv(scorecard, args.sequence_summary_csv)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "package-submission":
        validation = package_submission(
            args.prediction_dir,
            args.output_zip,
            template_zip=args.template_zip,
            normalize=args.normalize,
            sort_rows=args.sort_rows,
        )
        payload = asdict(validation)
        if args.output_json:
            _write_json(payload, args.output_json)
        if args.file_summary_csv:
            _write_file_summary_csv(validation, args.file_summary_csv)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not validation.valid:
            raise SystemExit(1)
        return

    if args.command == "constant-first-frame":
        summary = write_constant_first_frame_predictions(
            args.sequence_root,
            args.first_frame_label_dir,
            args.prediction_dir,
        )
        validation_payload = None
        if args.output_zip:
            validation = package_submission(
                args.prediction_dir,
                args.output_zip,
                template_zip=args.template_zip,
            )
            validation_payload = asdict(validation)
            if args.file_summary_csv:
                _write_file_summary_csv(validation, args.file_summary_csv)
        payload = {
            "baseline": "constant_first_frame",
            "prediction_summary": summary,
            "submission_validation": validation_payload,
        }
        if args.output_json:
            _write_json(payload, args.output_json)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if validation_payload and not validation_payload["valid"]:
            raise SystemExit(1)
        return

    if args.command == "first-frame-labels":
        payload = write_first_frame_labels(
            args.truth_dir,
            args.output_dir,
            frame_id=args.frame_id,
        )
        if args.output_json:
            _write_json(payload, args.output_json)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
