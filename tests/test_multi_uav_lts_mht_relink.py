from __future__ import annotations

from dataclasses import dataclass

import pytest

from raft_uav.multi_uav_lts._records import Detection, box_iou
from raft_uav.multi_uav_lts.mht_relink import relink_tracklets_beam


@dataclass(frozen=True)
class Tracklet:
    source_id: int
    index: int
    rows: tuple[Detection, ...]

    @property
    def start_frame(self) -> int:
        return self.rows[0].frame_id


def _detection(
    frame: int,
    object_id: int,
    center_x: float,
    confidence: float = 1.0,
) -> Detection:
    return Detection(
        frame,
        object_id,
        center_x - 5.0,
        0.0,
        10.0,
        10.0,
        confidence,
        1,
        1.0,
    )


def _predict(path: list[Detection], frame_id: int) -> Detection:
    last = path[-1]
    if len(path) < 2:
        return Detection(
            frame_id,
            last.object_id,
            last.x1,
            last.y1,
            last.width,
            last.height,
            last.confidence,
            last.class_id,
            last.visibility,
        )
    previous = path[-2]
    delta = last.frame_id - previous.frame_id
    velocity = (last.center_x - previous.center_x) / delta
    center = last.center_x + velocity * (frame_id - last.frame_id)
    return _detection(frame_id, last.object_id, center, last.confidence)


def _link_cost(
    path: list[Detection], rows: tuple[Detection, ...], *, gap: int
) -> float:
    predicted = _predict(path, rows[0].frame_id)
    center_distance = abs(predicted.center_x - rows[0].center_x) / 10.0
    return center_distance + 0.25 * (1.0 - box_iou(predicted, rows[0])) + 0.05 * gap


def _ambiguous_case() -> tuple[
    dict[int, list[Detection]], tuple[Tracklet, Tracklet]
]:
    assigned = {
        1: [_detection(1, 1, 0.0), _detection(2, 1, 0.0)],
        2: [_detection(1, 2, 10.0), _detection(2, 2, 10.0)],
    }
    ambiguous = Tracklet(
        201,
        0,
        (_detection(3, 201, 4.0), _detection(4, 201, 8.0)),
    )
    return_to_first = Tracklet(
        202,
        0,
        (_detection(5, 202, 0.0), _detection(6, 202, 0.0)),
    )
    return assigned, (ambiguous, return_to_first)


def test_beam_uses_later_fragment_to_reverse_ambiguous_assignment() -> None:
    assigned, tracklets = _ambiguous_case()

    local = relink_tracklets_beam(
        assigned,
        tracklets,
        max_gap=2,
        max_cost=2.0,
        beam_width=1,
        drop_cost=2.0,
        link_cost=_link_cost,
    )
    delayed = relink_tracklets_beam(
        assigned,
        tracklets,
        max_gap=2,
        max_cost=2.0,
        beam_width=2,
        drop_cost=2.0,
        link_cost=_link_cost,
    )

    assert [row.frame_id for row in local.assigned[1][-2:]] == [3, 4]
    assert [row.frame_id for row in delayed.assigned[2][-2:]] == [3, 4]
    assert [row.frame_id for row in delayed.assigned[1][-2:]] == [5, 6]
    assert delayed.best_cost < local.best_cost
    assert delayed.relinked_tracklets == 2
    assert delayed.relinked_source_ids == frozenset({201, 202})
    assert delayed.second_best_margin is not None
    assert delayed.second_best_margin > 0.0


def test_beam_width_must_be_positive() -> None:
    with pytest.raises(ValueError, match="beam_width"):
        relink_tracklets_beam(
            {1: [_detection(1, 1, 0.0)]},
            [],
            max_gap=0,
            max_cost=1.0,
            beam_width=0,
            drop_cost=1.0,
            link_cost=_link_cost,
        )
