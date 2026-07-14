from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raft_uav.multi_uav_lts.cli import LtsDetection
from raft_uav.multi_uav_lts.cli import _match_rows_by_iou
from raft_uav.multi_uav_lts.cli import score_lts_predictions


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
        np.array([0.5]),
        "not-a-threshold",
    ],
)
def test_lts_iou_matching_rejects_invalid_thresholds(threshold: object) -> None:
    with pytest.raises(ValueError, match=r"iou_threshold.*\[0, 1\]"):
        _match_rows_by_iou([], [], iou_threshold=threshold)


def test_lts_public_scorer_rejects_invalid_threshold_before_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        score_lts_predictions(
            tmp_path / "missing-predictions",
            tmp_path / "missing-truth",
            iou_threshold=float("nan"),
        )


def test_lts_iou_matching_accepts_finite_numpy_scalar_threshold() -> None:
    truth = [
        LtsDetection(
            frame_id=1,
            object_id=1,
            x1=0.0,
            y1=0.0,
            w=10.0,
            h=10.0,
            confidence=1.0,
            class_id=1,
            visibility=1.0,
        )
    ]
    predictions = [
        LtsDetection(
            frame_id=1,
            object_id=2,
            x1=0.0,
            y1=0.0,
            w=10.0,
            h=10.0,
            confidence=1.0,
            class_id=1,
            visibility=1.0,
        )
    ]

    matches = _match_rows_by_iou(
        truth,
        predictions,
        iou_threshold=np.float64(0.5),
    )

    assert matches == [(0, 0, 1.0)]
