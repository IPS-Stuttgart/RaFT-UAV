from __future__ import annotations

import numpy as np
import pytest

from raft_uav.multi_uav_lts import fixed_population
from raft_uav.multi_uav_lts._records import Detection


def _detection(object_id: int) -> Detection:
    return Detection(
        frame_id=1,
        object_id=object_id,
        x1=0.0,
        y1=0.0,
        width=10.0,
        height=10.0,
        confidence=1.0,
        class_id=1,
        visibility=1.0,
    )


def test_seed_mapping_excludes_subthreshold_edges_before_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlaps = np.asarray([[0.90, 0.49], [0.49, 0.00]], dtype=float)
    monkeypatch.setattr(fixed_population, "iou_matrix", lambda _left, _right: overlaps)

    mapping = fixed_population._seed_track_mapping(
        (_detection(7), _detection(9)),
        (_detection(1), _detection(2)),
        min_iou=0.5,
    )

    assert mapping == {1: 7}
