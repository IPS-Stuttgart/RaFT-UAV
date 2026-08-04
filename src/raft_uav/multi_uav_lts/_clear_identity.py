"""TrackEval-compatible CLEAR and identity metric cores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._records import PreparedSequence

_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class ClearCounts:
    tp: int
    fp: int
    fn: int
    id_switches: int
    motp_sum: float


@dataclass(frozen=True)
class IdentityCounts:
    tp: int
    fp: int
    fn: int


def evaluate_clear(data: PreparedSequence, *, threshold: float) -> ClearCounts:
    if data.num_tracker_detections == 0:
        return ClearCounts(0, 0, data.num_gt_detections, 0, 0.0)
    if data.num_gt_detections == 0:
        return ClearCounts(0, data.num_tracker_detections, 0, 0, 0.0)
    tp = fp = fn = switches = 0
    motp_sum = 0.0
    previous_id = np.full(data.num_gt_ids, np.nan)
    previous_timestep_id = np.full(data.num_gt_ids, np.nan)
    for gt_ids, tracker_ids, similarity in zip(
        data.gt_ids, data.tracker_ids, data.similarity_scores, strict=True
    ):
        if len(gt_ids) == 0:
            fp += len(tracker_ids)
            continue
        if len(tracker_ids) == 0:
            fn += len(gt_ids)
            continue
        continuity = tracker_ids[np.newaxis, :] == previous_timestep_id[gt_ids[:, None]]
        score = continuity.astype(float) * 1000.0 + similarity
        score[similarity < threshold - _EPS] = 0.0
        match_rows, match_cols = linear_sum_assignment(-score)
        valid = score[match_rows, match_cols] > _EPS
        match_rows = match_rows[valid]
        match_cols = match_cols[valid]
        matched_gt = gt_ids[match_rows]
        matched_tracker = tracker_ids[match_cols]
        previous = previous_id[matched_gt]
        switches += int(np.sum((~np.isnan(previous)) & (matched_tracker != previous)))
        previous_id[matched_gt] = matched_tracker
        previous_timestep_id[:] = np.nan
        previous_timestep_id[matched_gt] = matched_tracker
        matches = len(match_rows)
        tp += matches
        fn += len(gt_ids) - matches
        fp += len(tracker_ids) - matches
        if matches:
            motp_sum += float(similarity[match_rows, match_cols].sum())
    return ClearCounts(tp, fp, fn, switches, motp_sum)


def evaluate_identity(data: PreparedSequence, *, threshold: float) -> IdentityCounts:
    if data.num_tracker_detections == 0:
        return IdentityCounts(0, 0, data.num_gt_detections)
    if data.num_gt_detections == 0:
        return IdentityCounts(0, data.num_tracker_detections, 0)
    potential = np.zeros((data.num_gt_ids, data.num_tracker_ids), dtype=float)
    gt_count = np.zeros(data.num_gt_ids, dtype=float)
    tracker_count = np.zeros(data.num_tracker_ids, dtype=float)
    for gt_ids, tracker_ids, similarity in zip(
        data.gt_ids, data.tracker_ids, data.similarity_scores, strict=True
    ):
        eligible = (similarity >= threshold) & (similarity > _EPS)
        match_gt, match_tracker = np.nonzero(eligible)
        if len(match_gt):
            np.add.at(potential, (gt_ids[match_gt], tracker_ids[match_tracker]), 1)
        if len(gt_ids):
            gt_count[gt_ids] += 1
        if len(tracker_ids):
            tracker_count[tracker_ids] += 1
    size = data.num_gt_ids + data.num_tracker_ids
    fp_matrix = np.zeros((size, size), dtype=float)
    fn_matrix = np.zeros((size, size), dtype=float)
    fp_matrix[data.num_gt_ids :, : data.num_tracker_ids] = 1e10
    fn_matrix[: data.num_gt_ids, data.num_tracker_ids :] = 1e10
    for gt_id in range(data.num_gt_ids):
        fn_matrix[gt_id, : data.num_tracker_ids] = gt_count[gt_id]
        fn_matrix[gt_id, data.num_tracker_ids + gt_id] = gt_count[gt_id]
    for tracker_id in range(data.num_tracker_ids):
        fp_matrix[: data.num_gt_ids, tracker_id] = tracker_count[tracker_id]
        fp_matrix[data.num_gt_ids + tracker_id, tracker_id] = tracker_count[tracker_id]
    fn_matrix[: data.num_gt_ids, : data.num_tracker_ids] -= potential
    fp_matrix[: data.num_gt_ids, : data.num_tracker_ids] -= potential
    rows, cols = linear_sum_assignment(fn_matrix + fp_matrix)
    false_negatives = int(round(float(fn_matrix[rows, cols].sum())))
    false_positives = int(round(float(fp_matrix[rows, cols].sum())))
    true_positives = int(round(float(gt_count.sum()))) - false_negatives
    return IdentityCounts(true_positives, false_positives, false_negatives)


def combine_clear(rows: list[ClearCounts]) -> ClearCounts:
    return ClearCounts(
        sum(row.tp for row in rows),
        sum(row.fp for row in rows),
        sum(row.fn for row in rows),
        sum(row.id_switches for row in rows),
        sum(row.motp_sum for row in rows),
    )


def combine_identity(rows: list[IdentityCounts]) -> IdentityCounts:
    return IdentityCounts(
        sum(row.tp for row in rows),
        sum(row.fp for row in rows),
        sum(row.fn for row in rows),
    )


def finalize_clear(counts: ClearCounts) -> dict[str, float]:
    gt = max(1, counts.tp + counts.fn)
    return {
        "mota": (counts.tp - counts.fp - counts.id_switches) / gt,
        "motp": counts.motp_sum / max(1, counts.tp),
    }


def finalize_identity(counts: IdentityCounts) -> dict[str, float]:
    return {
        "idf1": counts.tp / max(1.0, counts.tp + 0.5 * counts.fp + 0.5 * counts.fn),
        "id_precision": counts.tp / max(1.0, counts.tp + counts.fp),
        "id_recall": counts.tp / max(1.0, counts.tp + counts.fn),
    }
