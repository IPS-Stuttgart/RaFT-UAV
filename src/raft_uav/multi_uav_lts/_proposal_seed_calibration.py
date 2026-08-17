"""First-frame detector calibration for Multi-UAV proposal banks."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._records import Detection, box_iou

_EPS = 1e-12


@dataclass(frozen=True)
class SeedCalibration:
    applied: bool
    matched_pairs: int
    inlier_pairs: int
    shift_x: float
    shift_y: float
    width_scale: float
    height_scale: float


def calibrate_proposals(
    seeds: Sequence[Detection],
    proposals: Sequence[Detection],
    *,
    min_pairs: int = 2,
    max_match_center_distance: float = 3.0,
    max_normalized_shift: float = 2.0,
    max_log_scale: float = math.log(2.0),
    max_normalized_residual: float = 1.5,
) -> tuple[tuple[Detection, ...], SeedCalibration]:
    """Calibrate a sequence-wide detector bias from exact frame-one boxes."""
    if min_pairs <= 0:
        raise ValueError("min_pairs must be positive")
    frame_one = tuple(row for row in proposals if row.frame_id == 1)
    matches = _matches(
        seeds,
        frame_one,
        max_center_distance=max_match_center_distance,
    )
    if len(matches) < min_pairs:
        return tuple(proposals), _identity(len(matches))

    shifts_x = np.asarray(
        [seed.center_x - proposal.center_x for seed, proposal in matches],
        dtype=float,
    )
    shifts_y = np.asarray(
        [seed.center_y - proposal.center_y for seed, proposal in matches],
        dtype=float,
    )
    log_width = np.asarray(
        [math.log(seed.width / proposal.width) for seed, proposal in matches],
        dtype=float,
    )
    log_height = np.asarray(
        [math.log(seed.height / proposal.height) for seed, proposal in matches],
        dtype=float,
    )
    scales = np.asarray(
        [
            max(
                4.0,
                math.sqrt(max(_EPS, seed.width * seed.height)),
                math.sqrt(max(_EPS, proposal.width * proposal.height)),
            )
            for seed, proposal in matches
        ],
        dtype=float,
    )

    shift_x = float(np.median(shifts_x))
    shift_y = float(np.median(shifts_y))
    width_log_scale = float(np.median(log_width))
    height_log_scale = float(np.median(log_height))
    residual = np.hypot(shifts_x - shift_x, shifts_y - shift_y) / scales
    residual += 0.25 * (
        np.abs(log_width - width_log_scale)
        + np.abs(log_height - height_log_scale)
    )
    inliers = residual <= max_normalized_residual
    if int(np.count_nonzero(inliers)) < min_pairs:
        return tuple(proposals), _identity(len(matches))

    shift_x = float(np.median(shifts_x[inliers]))
    shift_y = float(np.median(shifts_y[inliers]))
    width_log_scale = float(np.median(log_width[inliers]))
    height_log_scale = float(np.median(log_height[inliers]))
    reference_scale = float(np.median(scales[inliers]))
    if (
        math.hypot(shift_x, shift_y) / reference_scale > max_normalized_shift
        or abs(width_log_scale) > max_log_scale
        or abs(height_log_scale) > max_log_scale
    ):
        return tuple(proposals), _identity(len(matches))

    width_scale = math.exp(width_log_scale)
    height_scale = math.exp(height_log_scale)
    calibrated = tuple(
        _calibrate_row(
            row,
            shift_x=shift_x,
            shift_y=shift_y,
            width_scale=width_scale,
            height_scale=height_scale,
        )
        for row in proposals
    )
    return calibrated, SeedCalibration(
        applied=True,
        matched_pairs=len(matches),
        inlier_pairs=int(np.count_nonzero(inliers)),
        shift_x=shift_x,
        shift_y=shift_y,
        width_scale=width_scale,
        height_scale=height_scale,
    )


def _matches(
    seeds: Sequence[Detection],
    proposals: Sequence[Detection],
    *,
    max_center_distance: float,
) -> tuple[tuple[Detection, Detection], ...]:
    if not seeds or not proposals:
        return ()
    costs = np.full((len(seeds), len(proposals)), 1e9, dtype=float)
    valid = np.zeros_like(costs, dtype=bool)
    for seed_index, seed in enumerate(seeds):
        for proposal_index, proposal in enumerate(proposals):
            scale = max(
                4.0,
                math.sqrt(max(_EPS, seed.width * seed.height)),
                math.sqrt(max(_EPS, proposal.width * proposal.height)),
            )
            center = math.hypot(
                seed.center_x - proposal.center_x,
                seed.center_y - proposal.center_y,
            ) / scale
            size = abs(math.log(seed.width / proposal.width)) + abs(
                math.log(seed.height / proposal.height)
            )
            if center <= max_center_distance and size <= 2.0 * math.log(2.0):
                valid[seed_index, proposal_index] = True
                costs[seed_index, proposal_index] = (
                    center
                    + 0.25 * size
                    + 0.5 * (1.0 - box_iou(seed, proposal))
                )
    rows, columns = linear_sum_assignment(costs)
    return tuple(
        (seeds[int(seed_index)], proposals[int(proposal_index)])
        for seed_index, proposal_index in zip(rows, columns, strict=True)
        if valid[int(seed_index), int(proposal_index)]
    )


def _calibrate_row(
    row: Detection,
    *,
    shift_x: float,
    shift_y: float,
    width_scale: float,
    height_scale: float,
) -> Detection:
    width = row.width * width_scale
    height = row.height * height_scale
    center_x = row.center_x + shift_x
    center_y = row.center_y + shift_y
    return replace(
        row,
        x1=center_x - 0.5 * width,
        y1=center_y - 0.5 * height,
        width=width,
        height=height,
    )


def _identity(matched_pairs: int) -> SeedCalibration:
    return SeedCalibration(
        applied=False,
        matched_pairs=matched_pairs,
        inlier_pairs=0,
        shift_x=0.0,
        shift_y=0.0,
        width_scale=1.0,
        height_scale=1.0,
    )
