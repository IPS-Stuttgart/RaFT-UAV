from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.reid_tracklet_portfolio import (
    PortfolioParameters,
    _parse_scales,
    fuse_prediction_portfolio,
    fuse_sequence,
)
from raft_uav.multi_uav_lts.trajectory_box_calibration import BoxCalibrationParameters


class TableAppearance:
    def __init__(self, table: dict[tuple[int, int], tuple[float, float]]) -> None:
        self.table = table

    def embed(self, sequence: str, rows):
        del sequence
        return np.asarray(
            [self.table[(row.frame_id, int(round(row.x1)))] for row in rows],
            dtype=float,
        )


def _row(
    frame: int,
    x1: float,
    *,
    object_id: int = 1,
    confidence: float = 0.9,
) -> Detection:
    return Detection(frame, object_id, x1, 8.0, 4.0, 4.0, confidence, 1, 1.0)


def _controls(**changes) -> PortfolioParameters:
    defaults = dict(
        window_frames=4,
        appearance_weight=0.0,
        support_weight=0.0,
        smoothness_weight=0.0,
        transition_speed_weight=0.0,
        transition_size_weight=0.0,
        bridge_max_gap_frames=0,
    )
    defaults.update(changes)
    return PortfolioParameters(**defaults)


def test_anchor_consistency_can_select_lower_confidence_source() -> None:
    raw = tuple(_row(frame, 100.0) for frame in range(1, 5))
    alternative = tuple(
        _row(frame, 0.0, confidence=0.5) for frame in range(1, 5)
    )
    seed = (_row(1, 0.0),)
    appearance = TableAppearance(
        {
            **{(frame, 100): (0.0, 1.0) for frame in range(1, 5)},
            **{(frame, 0): (1.0, 0.0) for frame in range(1, 5)},
        }
    )

    fused, summary = fuse_sequence(
        "C_00",
        {"raw": raw, "alternative": alternative},
        seed_rows=seed,
        parameters=_controls(appearance_weight=2.0, confidence_weight=0.1),
        appearance_provider=appearance,
        smoother_parameters=BoxCalibrationParameters(),
        sequence_frame_count=4,
    )

    assert [row.x1 for row in fused] == [0.0, 0.0, 0.0, 0.0]
    assert summary.source_window_counts == {"alternative": 1}


def test_source_switch_penalty_can_keep_a_consistent_portfolio() -> None:
    source_a = tuple(
        _row(frame, 0.0, confidence=0.80) for frame in range(1, 9)
    )
    source_b = tuple(
        _row(
            frame,
            0.0,
            confidence=0.70 if frame <= 4 else 0.81,
        )
        for frame in range(1, 9)
    )

    fused, summary = fuse_sequence(
        "T_00",
        {"raw": source_a, "alternative": source_b},
        seed_rows=(_row(1, 0.0),),
        parameters=_controls(
            confidence_weight=1.0,
            source_switch_penalty=0.1,
        ),
        appearance_provider=None,
        smoother_parameters=BoxCalibrationParameters(),
        sequence_frame_count=8,
    )

    assert len(fused) == 8
    assert summary.source_window_counts == {"raw": 2}
    assert summary.source_switches == 0


def test_reid_gate_completes_only_identity_consistent_gap() -> None:
    rows = (_row(1, 0.0), _row(4, 3.0))
    seed = (_row(1, 0.0),)
    controls = _controls(
        window_frames=10,
        bridge_max_gap_frames=2,
        bridge_require_appearance=True,
        bridge_endpoint_appearance_threshold=0.1,
        bridge_anchor_appearance_threshold=0.1,
        bridge_use_smoothed_endpoints=False,
    )

    accepted, accepted_summary = fuse_sequence(
        "TF_00",
        {"raw": rows},
        seed_rows=seed,
        parameters=controls,
        appearance_provider=TableAppearance(
            {(1, 0): (1.0, 0.0), (4, 3): (1.0, 0.0)}
        ),
        smoother_parameters=BoxCalibrationParameters(),
        sequence_frame_count=4,
    )
    rejected, rejected_summary = fuse_sequence(
        "TF_00",
        {"raw": rows},
        seed_rows=seed,
        parameters=controls,
        appearance_provider=TableAppearance(
            {(1, 0): (1.0, 0.0), (4, 3): (0.0, 1.0)}
        ),
        smoother_parameters=BoxCalibrationParameters(),
        sequence_frame_count=4,
    )

    assert [row.frame_id for row in accepted] == [1, 2, 3, 4]
    assert accepted_summary.bridge.inserted_rows == 2
    assert rejected == rows
    assert rejected_summary.bridge.rejected_endpoint_appearance == 1


def test_bridge_only_mode_preserves_every_observed_box() -> None:
    rows = (_row(1, 0.0), _row(4, 4.0))
    fused, summary = fuse_sequence(
        "BB2P_00",
        {"raw": rows},
        seed_rows=(_row(1, 0.0),),
        parameters=_controls(
            window_frames=10,
            bridge_max_gap_frames=2,
            bridge_require_appearance=False,
            bridge_use_smoothed_endpoints=True,
        ),
        appearance_provider=None,
        smoother_parameters=BoxCalibrationParameters(),
        sequence_frame_count=4,
    )

    by_frame = {row.frame_id: row for row in fused}
    assert by_frame[1] == rows[0]
    assert by_frame[4] == rows[1]
    assert summary.bridge.inserted_rows == 2


def test_unseeded_tracks_come_only_from_raw_source() -> None:
    raw = (_row(1, 0.0), _row(2, 1.0), _row(2, 50.0, object_id=9))
    alternative = (
        _row(1, 0.0),
        _row(2, 1.0),
        _row(2, 80.0, object_id=9),
    )
    fused, summary = fuse_sequence(
        "C_01",
        {"raw": raw, "alternative": alternative},
        seed_rows=(_row(1, 0.0),),
        parameters=_controls(),
        appearance_provider=None,
        smoother_parameters=BoxCalibrationParameters(),
        sequence_frame_count=2,
    )

    birth = [row for row in fused if row.object_id == 9]
    assert len(birth) == 1
    assert birth[0].x1 == 50.0
    assert summary.raw_birth_track_count == 1


def test_prediction_set_rejects_candidate_coverage_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    alternative = tmp_path / "alternative"
    labels = tmp_path / "labels"
    output = tmp_path / "output"
    raw.mkdir()
    alternative.mkdir()
    labels.mkdir()
    text = "1,1,0,0,4,4,0.9,1,1\n"
    (raw / "C_00.txt").write_text(text)
    (alternative / "T_00.txt").write_text(text)
    (labels / "C_00.txt").write_text(text)

    with pytest.raises(ValueError, match="coverage mismatch"):
        fuse_prediction_portfolio(
            {"raw": raw, "alternative": alternative},
            labels,
            output,
            parameters=_controls(),
        )


def test_crop_scale_parser_rejects_duplicates() -> None:
    with pytest.raises(Exception, match="duplicates"):
        _parse_scales("1.0,1.0")
    assert _parse_scales("1.0,1.25") == (1.0, 1.25)
