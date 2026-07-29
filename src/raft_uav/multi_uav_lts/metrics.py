"""Competition-style HOTA, CLEAR, and identity metrics for Multi-UAV LTS."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ._clear_identity import (
    ClearCounts,
    IdentityCounts,
    combine_clear,
    combine_identity,
    evaluate_clear,
    evaluate_identity,
    finalize_clear,
    finalize_identity,
)
from ._hota import HOTA_ALPHAS, HotaCounts, combine_hota, evaluate_hota, finalize_hota
from ._records import (
    PreparedSequence,
    parse_detection_text,
    prediction_texts,
    prepare_sequence,
    validate_unit_interval,
)


@dataclass(frozen=True)
class SequenceMetrics:
    sequence: str
    frame_count: int
    gt_detections: int
    predicted_detections: int
    gt_ids: int
    predicted_ids: int
    hota: float
    deta: float
    assa: float
    loca: float
    mota: float
    motp: float
    idf1: float
    id_precision: float
    id_recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    id_true_positives: int
    id_false_positives: int
    id_false_negatives: int
    alphas: tuple[float, ...]
    hota_by_alpha: tuple[float, ...]
    deta_by_alpha: tuple[float, ...]
    assa_by_alpha: tuple[float, ...]
    loca_by_alpha: tuple[float, ...]
    hota_true_positives: tuple[int, ...]
    hota_false_positives: tuple[int, ...]
    hota_false_negatives: tuple[int, ...]


@dataclass(frozen=True)
class BenchmarkMetrics:
    prediction_path: str
    truth_dir: str
    clear_iou_threshold: float
    sequence_count: int
    gt_detections: int
    predicted_detections: int
    hota: float
    deta: float
    assa: float
    loca: float
    mota: float
    motp: float
    idf1: float
    id_precision: float
    id_recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    id_true_positives: int
    id_false_positives: int
    id_false_negatives: int
    alphas: tuple[float, ...]
    hota_by_alpha: tuple[float, ...]
    deta_by_alpha: tuple[float, ...]
    assa_by_alpha: tuple[float, ...]
    loca_by_alpha: tuple[float, ...]
    hota_true_positives: tuple[int, ...]
    hota_false_positives: tuple[int, ...]
    hota_false_negatives: tuple[int, ...]
    sequences: tuple[SequenceMetrics, ...]


def evaluate_lts_predictions(
    prediction_path: Path,
    truth_dir: Path,
    *,
    clear_iou_threshold: float = 0.5,
    sequences: Iterable[str] | None = None,
) -> BenchmarkMetrics:
    """Evaluate one prediction directory or submission ZIP."""

    threshold = validate_unit_interval(
        clear_iou_threshold, name="clear_iou_threshold"
    )
    predictions = prediction_texts(prediction_path)
    requested = set(sequences or ())
    truth_paths = sorted(truth_dir.glob("*.txt"))
    if requested:
        missing = sorted(requested - {path.stem for path in truth_paths})
        if missing:
            raise ValueError(f"unknown truth sequences: {', '.join(missing)}")

    sequence_metrics: list[SequenceMetrics] = []
    hota_counts: list[HotaCounts] = []
    clear_counts: list[ClearCounts] = []
    identity_counts: list[IdentityCounts] = []
    for truth_path in truth_paths:
        sequence = truth_path.stem
        if requested and sequence not in requested:
            continue
        truth_rows = parse_detection_text(
            truth_path.read_text(encoding="utf-8"), source=str(truth_path)
        )
        prediction_rows = parse_detection_text(
            predictions.get(f"{sequence}.txt", ""),
            source=f"{prediction_path}:{sequence}.txt",
        )
        prepared = prepare_sequence(truth_rows, prediction_rows)
        hota = evaluate_hota(prepared)
        clear = evaluate_clear(prepared, threshold=threshold)
        identity = evaluate_identity(prepared, threshold=threshold)
        sequence_metrics.append(
            _sequence_metrics(sequence, prepared, hota, clear, identity)
        )
        hota_counts.append(hota)
        clear_counts.append(clear)
        identity_counts.append(identity)

    hota = combine_hota(hota_counts)
    clear = combine_clear(clear_counts)
    identity = combine_identity(identity_counts)
    hota_fields = finalize_hota(hota)
    clear_fields = finalize_clear(clear)
    identity_fields = finalize_identity(identity)
    return BenchmarkMetrics(
        prediction_path=str(prediction_path),
        truth_dir=str(truth_dir),
        clear_iou_threshold=threshold,
        sequence_count=len(sequence_metrics),
        gt_detections=sum(row.gt_detections for row in sequence_metrics),
        predicted_detections=sum(row.predicted_detections for row in sequence_metrics),
        hota=float(np.mean(hota_fields["hota"])),
        deta=float(np.mean(hota_fields["deta"])),
        assa=float(np.mean(hota_fields["assa"])),
        loca=float(np.mean(hota_fields["loca"])),
        mota=clear_fields["mota"],
        motp=clear_fields["motp"],
        idf1=identity_fields["idf1"],
        id_precision=identity_fields["id_precision"],
        id_recall=identity_fields["id_recall"],
        true_positives=clear.tp,
        false_positives=clear.fp,
        false_negatives=clear.fn,
        id_switches=clear.id_switches,
        id_true_positives=identity.tp,
        id_false_positives=identity.fp,
        id_false_negatives=identity.fn,
        alphas=HOTA_ALPHAS,
        hota_by_alpha=_float_tuple(hota_fields["hota"]),
        deta_by_alpha=_float_tuple(hota_fields["deta"]),
        assa_by_alpha=_float_tuple(hota_fields["assa"]),
        loca_by_alpha=_float_tuple(hota_fields["loca"]),
        hota_true_positives=_int_tuple(hota.tp),
        hota_false_positives=_int_tuple(hota.fp),
        hota_false_negatives=_int_tuple(hota.fn),
        sequences=tuple(sequence_metrics),
    )


def _sequence_metrics(
    sequence: str,
    prepared: PreparedSequence,
    hota: HotaCounts,
    clear: ClearCounts,
    identity: IdentityCounts,
) -> SequenceMetrics:
    hota_fields = finalize_hota(hota)
    clear_fields = finalize_clear(clear)
    identity_fields = finalize_identity(identity)
    return SequenceMetrics(
        sequence=sequence,
        frame_count=prepared.frame_count,
        gt_detections=prepared.num_gt_detections,
        predicted_detections=prepared.num_tracker_detections,
        gt_ids=prepared.num_gt_ids,
        predicted_ids=prepared.num_tracker_ids,
        hota=float(np.mean(hota_fields["hota"])),
        deta=float(np.mean(hota_fields["deta"])),
        assa=float(np.mean(hota_fields["assa"])),
        loca=float(np.mean(hota_fields["loca"])),
        mota=clear_fields["mota"],
        motp=clear_fields["motp"],
        idf1=identity_fields["idf1"],
        id_precision=identity_fields["id_precision"],
        id_recall=identity_fields["id_recall"],
        true_positives=clear.tp,
        false_positives=clear.fp,
        false_negatives=clear.fn,
        id_switches=clear.id_switches,
        id_true_positives=identity.tp,
        id_false_positives=identity.fp,
        id_false_negatives=identity.fn,
        alphas=HOTA_ALPHAS,
        hota_by_alpha=_float_tuple(hota_fields["hota"]),
        deta_by_alpha=_float_tuple(hota_fields["deta"]),
        assa_by_alpha=_float_tuple(hota_fields["assa"]),
        loca_by_alpha=_float_tuple(hota_fields["loca"]),
        hota_true_positives=_int_tuple(hota.tp),
        hota_false_positives=_int_tuple(hota.fp),
        hota_false_negatives=_int_tuple(hota.fn),
    )


def _float_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _int_tuple(values: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def write_metrics_json(metrics: BenchmarkMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True), encoding="utf-8"
    )


def write_sequence_csv(metrics: BenchmarkMetrics, path: Path) -> None:
    fields = [
        "sequence",
        "frame_count",
        "gt_detections",
        "predicted_detections",
        "gt_ids",
        "predicted_ids",
        "hota",
        "deta",
        "assa",
        "loca",
        "mota",
        "motp",
        "idf1",
        "id_precision",
        "id_recall",
        "true_positives",
        "false_positives",
        "false_negatives",
        "id_switches",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metrics.sequences:
            payload = asdict(row)
            writer.writerow({field: payload[field] for field in fields})


def write_alpha_csv(metrics: BenchmarkMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha", "hota", "deta", "assa", "loca", "tp", "fp", "fn"])
        for values in zip(
            metrics.alphas,
            metrics.hota_by_alpha,
            metrics.deta_by_alpha,
            metrics.assa_by_alpha,
            metrics.loca_by_alpha,
            metrics.hota_true_positives,
            metrics.hota_false_positives,
            metrics.hota_false_negatives,
            strict=True,
        ):
            writer.writerow(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--clear-iou-threshold", type=float, default=0.5)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sequence-summary-csv", type=Path)
    parser.add_argument("--alpha-summary-csv", type=Path)
    args = parser.parse_args(argv)
    metrics = evaluate_lts_predictions(
        args.prediction_path,
        args.truth_dir,
        clear_iou_threshold=args.clear_iou_threshold,
        sequences=args.sequences,
    )
    if args.output_json:
        write_metrics_json(metrics, args.output_json)
    if args.sequence_summary_csv:
        write_sequence_csv(metrics, args.sequence_summary_csv)
    if args.alpha_summary_csv:
        write_alpha_csv(metrics, args.alpha_summary_csv)
    for name in ("hota", "deta", "assa", "loca", "mota", "idf1"):
        print(f"{name.upper()}={getattr(metrics, name):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
