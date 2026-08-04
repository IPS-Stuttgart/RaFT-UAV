"""Normalize serialized radar chronology in research diagnostics."""

from __future__ import annotations

from types import ModuleType

import pandas as pd

from raft_uav.numeric import optional_float, optional_int

_PATCH_MARKER = "_raft_uav_normalizes_research_diagnostic_radar_frames"


def _event_index_value(value: object) -> int | float | None:
    """Return an exact integer frame index or a finite fractional fallback."""

    integer = optional_int(value)
    if integer is not None:
        return integer
    return optional_float(value)


def _sort_value(value: object, *, column: str) -> tuple[int, int | float, str]:
    """Return a numeric-first stable sort key for one radar scalar."""

    if column == "frame_index":
        number = _event_index_value(value)
    elif column == "track_id":
        number = optional_int(value)
    else:
        number = optional_float(value)
    if number is not None:
        return (0, number, "")
    return (1, 0, str(value))


def _ordered_radar_rows(radar: pd.DataFrame) -> pd.DataFrame:
    """Order numeric-like chronology keys without mutating identifier payloads."""

    rows = pd.DataFrame(radar).copy()
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id")
        if column in rows.columns
    ]
    positions = sorted(
        range(len(rows)),
        key=lambda position: tuple(
            _sort_value(rows.iloc[position][column], column=column)
            for column in sort_columns
        ),
    )
    ordered = rows.iloc[positions].copy()
    if "time_s" in ordered.columns:
        ordered["time_s"] = [
            optional_float(value) for value in ordered["time_s"]
        ]
    return ordered


def apply_diagnostics_numeric_frame_patch(module: ModuleType) -> None:
    """Patch radar grouping used by public research diagnostic helpers."""

    original = module._radar_frame_groups
    if getattr(original, _PATCH_MARKER, False):
        return

    def radar_frame_groups(
        radar: pd.DataFrame,
    ) -> list[tuple[tuple[str, int | float], pd.DataFrame]]:
        if radar.empty:
            return []
        ordered = _ordered_radar_rows(radar)
        frame_values = (
            ordered["frame_index"].tolist()
            if "frame_index" in ordered.columns
            else [None] * len(ordered)
        )
        time_values = (
            ordered["time_s"].tolist()
            if "time_s" in ordered.columns
            else [None] * len(ordered)
        )

        positions_by_key: dict[tuple[str, int | float], list[int]] = {}
        for position, (frame_index, time_s) in enumerate(
            zip(frame_values, time_values, strict=True)
        ):
            event_index = _event_index_value(frame_index)
            if event_index is not None:
                event_key = ("frame_index", event_index)
            else:
                event_time = optional_float(time_s)
                if event_time is None:
                    continue
                event_key = ("time_s", round(event_time, 9))
            positions_by_key.setdefault(event_key, []).append(position)

        return [
            (event_key, ordered.iloc[positions].copy())
            for event_key, positions in positions_by_key.items()
        ]

    setattr(radar_frame_groups, _PATCH_MARKER, True)
    implementation = getattr(module, "_LEGACY", module)
    implementation._radar_frame_groups = radar_frame_groups
    module._radar_frame_groups = radar_frame_groups
