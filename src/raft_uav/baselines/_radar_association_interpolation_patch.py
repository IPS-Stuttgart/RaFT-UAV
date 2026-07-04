"""Runtime patch for radar-association stable-segment interpolation bounds.

The stable-segment interpolation helpers live in the large legacy
``radar_association`` module.  Keeping the replacement helpers here lets the
package apply a minimal, well-tested fix without changing unrelated association
logic.
"""

from __future__ import annotations

from importlib import import_module

import numpy as np


def apply_radar_association_interpolation_patch() -> None:
    """Install interpolation masks that reject frames outside the anchor span."""

    radar_association = import_module("raft_uav.baselines.radar_association")
    radar_association._within_interpolation_gap = _within_interpolation_gap
    radar_association._within_interpolation_speed = _within_interpolation_speed


def _inside_anchor_span(frame_times: np.ndarray, anchor_times: np.ndarray) -> np.ndarray:
    """Return whether each frame time lies between the first and last anchor."""

    if anchor_times.size == 0:
        return np.zeros(frame_times.shape, dtype=bool)
    return (frame_times >= anchor_times[0]) & (frame_times <= anchor_times[-1])


def _anchor_hits(
    frame_times: np.ndarray,
    anchor_times: np.ndarray,
    inside_span: np.ndarray,
    insertion: np.ndarray,
) -> np.ndarray:
    """Return exact anchor matches without treating out-of-span frames as anchors."""

    on_anchor = inside_span & (insertion < anchor_times.size)
    if frame_times.size == 0:
        return on_anchor
    anchor_indices = np.minimum(insertion, anchor_times.size - 1)
    return on_anchor & np.isclose(anchor_times[anchor_indices], frame_times)


def _within_interpolation_gap(
    frame_times: np.ndarray,
    anchor_times: np.ndarray,
    *,
    max_gap_s: float,
) -> np.ndarray:
    """Return frames bracketed by anchors no farther apart than ``max_gap_s``.

    Frames before the first anchor or after the last anchor are not bracketed and
    must remain rejected even when the nearest internal anchor gap is short.
    """

    frame_times = np.asarray(frame_times, dtype=float).reshape(-1)
    anchor_times = np.asarray(anchor_times, dtype=float).reshape(-1)
    if frame_times.size == 0:
        return np.zeros(0, dtype=bool)
    if anchor_times.size <= 1:
        return np.isin(frame_times, anchor_times)
    insertion = np.searchsorted(anchor_times, frame_times, side="left")
    inside_span = _inside_anchor_span(frame_times, anchor_times)
    on_anchor = _anchor_hits(frame_times, anchor_times, inside_span, insertion)
    right = np.clip(insertion, 1, anchor_times.size - 1)
    left = right - 1
    bracket_gap_s = anchor_times[right] - anchor_times[left]
    return on_anchor | (inside_span & (bracket_gap_s <= float(max_gap_s)))


def _within_interpolation_speed(
    frame_times: np.ndarray,
    anchor_times: np.ndarray,
    anchor_positions: np.ndarray,
    *,
    max_speed_mps: float,
) -> np.ndarray:
    """Return frames bracketed by anchors no faster than ``max_speed_mps``.

    Like the gap check, this is a true interpolation mask: extrapolation outside
    the anchor span is deliberately rejected.
    """

    frame_times = np.asarray(frame_times, dtype=float).reshape(-1)
    anchor_times = np.asarray(anchor_times, dtype=float).reshape(-1)
    anchor_positions = np.asarray(anchor_positions, dtype=float)
    if frame_times.size == 0:
        return np.zeros(0, dtype=bool)
    if anchor_times.size <= 1:
        return np.isin(frame_times, anchor_times)
    insertion = np.searchsorted(anchor_times, frame_times, side="left")
    inside_span = _inside_anchor_span(frame_times, anchor_times)
    on_anchor = _anchor_hits(frame_times, anchor_times, inside_span, insertion)
    right = np.clip(insertion, 1, anchor_times.size - 1)
    left = right - 1
    dt_s = anchor_times[right] - anchor_times[left]
    distance_m = np.linalg.norm(anchor_positions[right] - anchor_positions[left], axis=1)
    speeds_mps = np.divide(
        distance_m,
        dt_s,
        out=np.full_like(distance_m, np.inf, dtype=float),
        where=dt_s > 0.0,
    )
    return on_anchor | (inside_span & (speeds_mps <= float(max_speed_mps)))
