from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raft_uav.multi_uav_lts.cli import LtsDetection
from raft_uav.multi_uav_lts.cli import _match_rows_by_iou
from raft_uav.multi_uav_lts.cli import score_lts_predictions


def _detection(*, object_id: int, x1: float = 0.0) -> LtsDetection:
    return LtsDetection(
        frame_id=1,
        object_id=object_id,
        x1=x1,
        y1=0.0,
        w=10.0,
        h=10.0,
        confidence=1.0,
        class_id=1,
        visibility=1.0,
    )


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        1.1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        0.5 + 1.0j,
        np.array([0.5]),
        np.ma.masked,
        np.ma.array(0.5, mask=True),
        "not-a-threshold",
    ],
)
def test_lts_iou_matching_rejects_invalid_thresholds(threshold: object) -> None:
    with pytest.raises(ValueError, match=r"iou_threshold.*\[0, 1\]"):
        _match_rows_by_iou([], [], iou_threshold=threshold)


def test_lts_public_scorer_rejects_invalid_threshold_before_io(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        score_lts_predictions(
            tmp_path / "missing-predictions",
            tmp_path / "missing-truth",
            iou_threshold=np.nan,
        )


def test_lts_iou_matching_rejects_negative_threshold_with_real_rows() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        _match_rows_by_iou(
            [_detection(object_id=1)],
            [_detection(object_id=2, x1=100.0)],
            iou_threshold=-0.1,
        )


@pytest.mark.parametrize(
    "threshold",
    [0.0, 0.5, 1.0, np.float64(0.5), np.array(0.5)],
)
def test_lts_iou_matching_accepts_valid_scalar_thresholds(
    threshold: object,
) -> None:
    matches = _match_rows_by_iou(
        [_detection(object_id=1)],
        [_detection(object_id=2)],
        iou_threshold=threshold,
    )

    assert matches == [(0, 0, 1.0)]
