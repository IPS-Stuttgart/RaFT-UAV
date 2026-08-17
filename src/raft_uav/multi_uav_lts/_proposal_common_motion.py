"""Fast reliability-gated common motion estimation for proposal graphs."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ._records import Detection

_MAX_ROWS = 96
_EPS = 1e-12


class _NodeLike(Protocol):
    index: int
    row: Detection


class _Parameters(Protocol):
    enable_common_motion: bool
    common_motion_min_pairs: int
    common_motion_max_normalized_step: float
    common_motion_max_normalized_residual: float


def estimate_common_motion(
    nodes: Sequence[_NodeLike],
    parameters: _Parameters,
) -> dict[int, tuple[float, float]]:
    """Estimate a robust shared translation without per-frame cubic assignment."""
    if not parameters.enable_common_motion:
        return {}
    frames: dict[int, list[_NodeLike]] = {}
    for node in nodes:
        frames.setdefault(node.row.frame_id, []).append(node)

    estimates: dict[int, tuple[float, float]] = {}
    for frame in sorted(frames):
        left = _motion_rows(frames[frame])
        right = _motion_rows(frames.get(frame + 1, ()))
        if min(len(left), len(right)) < parameters.common_motion_min_pairs:
            continue
        left_x, left_y, left_width, left_height = _arrays(left)
        right_x, right_y, right_width, right_height = _arrays(right)

        initial_dx = float(np.median(right_x) - np.median(left_x))
        initial_dy = float(np.median(right_y) - np.median(left_y))
        delta_x = right_x[None, :] - (left_x[:, None] + initial_dx)
        delta_y = right_y[None, :] - (left_y[:, None] + initial_dy)
        scale = np.maximum(
            4.0,
            np.maximum(
                np.sqrt(np.maximum(_EPS, left_width * left_height))[:, None],
                np.sqrt(np.maximum(_EPS, right_width * right_height))[None, :],
            ),
        )
        center_cost = np.hypot(delta_x, delta_y) / scale
        size_cost = np.abs(
            np.log(right_width[None, :] / left_width[:, None])
        ) + np.abs(np.log(right_height[None, :] / left_height[:, None]))
        costs = center_cost + 0.25 * size_cost

        forward = np.argmin(costs, axis=1)
        backward = np.argmin(costs, axis=0)
        left_indices = np.arange(len(left), dtype=int)
        reciprocal = backward[forward] == left_indices
        if not np.any(reciprocal):
            continue
        selected_left = left_indices[reciprocal]
        selected_right = forward[reciprocal]
        dx = right_x[selected_right] - left_x[selected_left]
        dy = right_y[selected_right] - left_y[selected_left]
        pair_scale = scale[selected_left, selected_right]
        normalized_step = np.hypot(dx, dy) / pair_scale
        plausible = (
            normalized_step <= parameters.common_motion_max_normalized_step
        )
        if int(np.count_nonzero(plausible)) < parameters.common_motion_min_pairs:
            continue
        dx = dx[plausible]
        dy = dy[plausible]
        pair_scale = pair_scale[plausible]

        median_dx = float(np.median(dx))
        median_dy = float(np.median(dy))
        residual = np.hypot(dx - median_dx, dy - median_dy) / pair_scale
        inliers = residual <= parameters.common_motion_max_normalized_residual
        if int(np.count_nonzero(inliers)) < parameters.common_motion_min_pairs:
            continue
        refined_dx = float(np.median(dx[inliers]))
        refined_dy = float(np.median(dy[inliers]))
        refined_residual = (
            np.hypot(dx[inliers] - refined_dx, dy[inliers] - refined_dy)
            / pair_scale[inliers]
        )
        if (
            float(np.median(refined_residual))
            > parameters.common_motion_max_normalized_residual
        ):
            continue
        estimates[frame] = (refined_dx, refined_dy)
    return estimates


def _motion_rows(nodes: Sequence[_NodeLike]) -> tuple[_NodeLike, ...]:
    return tuple(
        sorted(
            nodes,
            key=lambda node: (
                -node.row.confidence,
                node.row.x1,
                node.row.y1,
                node.index,
            ),
        )[:_MAX_ROWS]
    )


def _arrays(
    nodes: Sequence[_NodeLike],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([node.row.center_x for node in nodes], dtype=float),
        np.asarray([node.row.center_y for node in nodes], dtype=float),
        np.asarray([node.row.width for node in nodes], dtype=float),
        np.asarray([node.row.height for node in nodes], dtype=float),
    )
