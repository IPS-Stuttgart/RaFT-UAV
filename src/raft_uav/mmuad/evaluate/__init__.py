"""Maintained MMUAD submission evaluation overrides.

The legacy implementation remains in the sibling ``evaluate.py`` module.  This
package preserves the public import path while replacing nearest-time matching
with a cardinality-first one-to-one assignment that is independent of CSV row
order.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_IMPL_PATH = Path(__file__).resolve().parent.parent / "evaluate.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._evaluate_impl",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise ImportError(f"cannot load MMUAD evaluation implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)


def match_submission_to_truth(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
) -> pd.DataFrame:
    """Match predictions to truth with cardinality-first optimal assignment.

    Existing sequence and track-ID gating semantics are retained.  Within each
    sequence, the assignment first maximizes the number of truth rows matched
    inside the time gate and then minimizes total absolute timestamp error.
    """

    if truth.empty or submission.empty:
        return pd.DataFrame()
    submission = submission.copy()
    if "sequence_id" not in submission.columns:
        submission["sequence_id"] = "default"
    else:
        submission["sequence_id"] = _normalize_submission_sequence_ids(
            submission["sequence_id"]
        )
    if "track_id" in submission.columns:
        submission["track_id"] = submission["track_id"].map(
            lambda value: _valid_track_id_text(value) or ""
        )
    truth = normalize_truth_columns(truth)
    if "track_id" in truth.columns:
        truth["track_id"] = truth["track_id"].map(
            lambda value: _valid_track_id_text(value) or ""
        )

    rows: list[dict[str, Any]] = []
    for sequence_id, pred_seq in submission.groupby("sequence_id", sort=True):
        truth_seq = truth.loc[truth["sequence_id"] == sequence_id].copy()
        if truth_seq.empty:
            rows.extend(
                _unmatched_prediction_row(pred, reason="missing_sequence_truth")
                for _, pred in pred_seq.iterrows()
            )
            continue

        pred_seq = pred_seq.reset_index(drop=True)
        truth_seq = truth_seq.reset_index(drop=True)
        truth_track_ids = _track_ids(truth_seq) if "track_id" in truth_seq.columns else set()
        submitted_track_ids = _track_ids(pred_seq) if "track_id" in pred_seq.columns else set()
        restrict_to_track_id = _should_restrict_to_track_id(
            truth_track_ids,
            submitted_track_ids,
        )
        assignments, eligible = _optimal_time_assignment(
            pred_seq,
            truth_seq,
            restrict_to_track_id=restrict_to_track_id,
            max_time_delta_s=float(max_time_delta_s),
        )
        truth_track_values = (
            truth_seq["track_id"].map(_valid_track_id_text).to_numpy(dtype=object)
            if "track_id" in truth_seq.columns
            else np.full(len(truth_seq), None, dtype=object)
        )

        for pred_position, pred in pred_seq.iterrows():
            pred_track_id = _valid_track_id_text(pred.get("track_id", ""))
            if restrict_to_track_id and (
                pred_track_id is None or pred_track_id not in truth_track_ids
            ):
                rows.append(_unmatched_prediction_row(pred, reason="track_id_mismatch"))
                continue

            truth_position = assignments.get(int(pred_position))
            if truth_position is not None:
                rows.append(
                    _matched_prediction_row(
                        sequence_id=sequence_id,
                        prediction=pred,
                        truth=truth_seq.iloc[truth_position],
                    )
                )
                continue

            candidate_mask = np.ones(len(truth_seq), dtype=bool)
            if restrict_to_track_id:
                candidate_mask = truth_track_values == pred_track_id
            if not bool(candidate_mask.any()):
                reason = "missing_track_truth"
            elif bool(eligible[int(pred_position)].any()):
                reason = "duplicate_truth_match"
            else:
                reason = "time_gate"
            rows.append(_unmatched_prediction_row(pred, reason=reason))
    return pd.DataFrame.from_records(rows)


def _optimal_time_assignment(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    restrict_to_track_id: bool,
    max_time_delta_s: float,
) -> tuple[dict[int, int], np.ndarray]:
    """Return a cardinality-first one-to-one assignment and eligibility mask."""

    pred_times = pd.to_numeric(predictions["time_s"], errors="coerce").to_numpy(float)
    truth_times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(float)
    time_delta = np.abs(pred_times[:, np.newaxis] - truth_times[np.newaxis, :])
    eligible = np.isfinite(time_delta) & (time_delta <= max_time_delta_s)

    if restrict_to_track_id:
        pred_track_ids = predictions["track_id"].map(_valid_track_id_text).to_numpy(dtype=object)
        truth_track_ids = truth["track_id"].map(_valid_track_id_text).to_numpy(dtype=object)
        eligible &= pred_track_ids[:, np.newaxis] == truth_track_ids[np.newaxis, :]

    if not bool(eligible.any()):
        return {}, eligible

    max_matches = min(len(predictions), len(truth))
    distance_weight = 0.5 / float(max_matches + 1)
    distance_scale = max(float(np.max(time_delta[eligible])), 1.0)
    costs = np.full(
        (len(predictions), len(truth) + len(predictions)),
        1.0,
        dtype=float,
    )
    costs[:, : len(truth)] = np.where(
        eligible,
        distance_weight * time_delta / distance_scale,
        2.0,
    )
    pred_positions, assignment_columns = linear_sum_assignment(costs)
    assignments = {
        int(pred_position): int(truth_position)
        for pred_position, truth_position in zip(pred_positions, assignment_columns)
        if truth_position < len(truth) and eligible[pred_position, truth_position]
    }
    return assignments, eligible


def _matched_prediction_row(
    *,
    sequence_id: str,
    prediction: pd.Series,
    truth: pd.Series,
) -> dict[str, Any]:
    error = np.array(
        [
            float(prediction["x_m"]) - float(truth["x_m"]),
            float(prediction["y_m"]) - float(truth["y_m"]),
            float(prediction["z_m"]) - float(truth["z_m"]),
        ],
        dtype=float,
    )
    return {
        "sequence_id": sequence_id,
        "time_s": float(prediction["time_s"]),
        "track_id": _valid_track_id_text(prediction.get("track_id", "uav0")) or "uav0",
        "truth_time_s": float(truth["time_s"]),
        "truth_track_id": _truth_track_id(truth),
        "time_delta_s": abs(float(truth["time_s"]) - float(prediction["time_s"])),
        "matched": True,
        "unmatched_reason": "",
        "error_2d_m": float(np.linalg.norm(error[:2])),
        "error_3d_m": float(np.linalg.norm(error)),
        "vertical_error_m": float(error[2]),
    }


_IMPL.match_submission_to_truth = match_submission_to_truth
