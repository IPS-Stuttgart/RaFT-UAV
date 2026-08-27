from __future__ import annotations

import pytest

from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.trajectory_birth_filter import (
    BirthFilterParameters,
    filter_sequence,
)


def _row(
    frame: int,
    object_id: int,
    x1: float,
    *,
    confidence: float = 0.5,
) -> Detection:
    return Detection(frame, object_id, x1, 40.0, 10.0, 10.0, confidence, 1, 1.0)


def test_birth_filter_preserves_seed_and_drops_short_birth() -> None:
    rows = (
        _row(1, 1, 20.0, confidence=0.001),
        _row(2, 1, 21.0, confidence=0.001),
        _row(10, 2, 50.0),
        _row(11, 2, 51.0),
        _row(20, 3, 70.0),
        _row(21, 3, 71.0),
        _row(22, 3, 72.0),
        _row(23, 3, 73.0),
        _row(24, 3, 74.0),
    )
    filtered, summary = filter_sequence(
        "S_00",
        rows,
        seed_ids={1},
        parameters=BirthFilterParameters(min_hits=5, min_span=4),
    )

    assert {row.object_id for row in filtered} == {1, 3}
    assert summary.birth_tracks == 2
    assert summary.kept_birth_tracks == 1
    assert summary.dropped_birth_tracks == 1
    assert summary.dropped_birth_rows == 2


def test_birth_filter_requires_border_entry_and_inward_motion() -> None:
    accepted = tuple(_row(frame, 4, float(frame - 1) * 4.0) for frame in range(5, 10))
    rejected = tuple(_row(frame, 5, 45.0 + frame) for frame in range(5, 10))
    parameters = BirthFilterParameters(
        min_hits=5,
        min_span=4,
        require_border_entry=True,
        min_inward_motion=0.5,
        image_width=100.0,
        image_height=100.0,
        border_margin_fraction=0.1,
    )

    filtered, summary = filter_sequence(
        "S_00",
        (*accepted, *rejected),
        seed_ids=set(),
        parameters=parameters,
    )

    assert {row.object_id for row in filtered} == {4}
    assert summary.kept_birth_tracks == 1
    assert summary.dropped_birth_tracks == 1


def test_drop_all_births_is_seed_only_diagnostic() -> None:
    rows = (
        _row(1, 7, 20.0),
        _row(2, 7, 21.0),
        _row(10, 8, 0.0),
        _row(11, 8, 4.0),
        _row(12, 8, 8.0),
        _row(13, 8, 12.0),
        _row(14, 8, 16.0),
    )
    filtered, summary = filter_sequence(
        "S_00",
        rows,
        seed_ids={7},
        parameters=BirthFilterParameters(drop_all_births=True),
    )

    assert {row.object_id for row in filtered} == {7}
    assert summary.dropped_birth_tracks == 1


def test_border_filter_requires_dimensions() -> None:
    with pytest.raises(ValueError, match="requires image dimensions"):
        BirthFilterParameters(require_border_entry=True).validate()
