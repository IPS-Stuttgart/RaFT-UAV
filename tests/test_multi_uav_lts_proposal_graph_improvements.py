from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts._records import (
    Detection,
    format_detection,
    parse_detection_text,
)
from raft_uav.multi_uav_lts.proposal_graph_tracker import track_proposal_graph


def row(
    frame: int,
    proposal_id: int,
    x: float,
    *,
    y: float = 10.0,
    width: float = 4.0,
    height: float = 4.0,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        frame,
        proposal_id,
        x,
        y,
        width,
        height,
        confidence,
        1,
        1.0,
    )


def write_rows(path: Path, rows: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(format_detection(item) + "\n" for item in rows),
        encoding="utf-8",
    )


def run_tracker(
    tmp_path: Path,
    seeds: list[Detection],
    proposals: list[Detection],
    **kwargs: object,
):
    labels = tmp_path / "labels"
    proposal_dir = tmp_path / "proposals"
    output = tmp_path / "output"
    write_rows(labels / "SEQ.txt", seeds)
    write_rows(proposal_dir / "SEQ.txt", proposals)
    summary = track_proposal_graph(proposal_dir, labels, output, **kwargs)
    rows = parse_detection_text(
        (output / "SEQ.txt").read_text(encoding="utf-8"),
        source="output",
    )
    return summary, rows


def test_common_motion_preserves_large_shared_translation(tmp_path: Path) -> None:
    starts = [0.0, 50.0, 100.0, 150.0]
    seeds = [row(1, index + 1, x) for index, x in enumerate(starts)]
    proposals = [
        row(frame, index + 1, x + 20.0 * (frame - 1))
        for frame in (1, 2, 3)
        for index, x in enumerate(starts)
    ]

    summary, rows = run_tracker(
        tmp_path,
        seeds,
        proposals,
        enable_common_motion=True,
        common_motion_min_pairs=4,
        common_motion_max_normalized_step=8.0,
        common_motion_max_normalized_residual=0.5,
        birth_min_hits=100,
    )

    by_id: dict[int, list[int]] = {}
    for item in rows:
        by_id.setdefault(item.object_id, []).append(item.frame_id)
    assert by_id == {
        1: [1, 2, 3],
        2: [1, 2, 3],
        3: [1, 2, 3],
        4: [1, 2, 3],
    }
    assert summary.common_motion_steps == 2


def test_common_motion_gate_rejects_incoherent_steps(tmp_path: Path) -> None:
    starts = [0.0, 50.0, 100.0, 150.0]
    shifts = [-20.0, -5.0, 10.0, 30.0]
    seeds = [row(1, index + 1, x) for index, x in enumerate(starts)]
    proposals = [*seeds]
    proposals.extend(
        row(2, index + 1, x + shift)
        for index, (x, shift) in enumerate(zip(starts, shifts, strict=True))
    )

    summary, _rows = run_tracker(
        tmp_path,
        seeds,
        proposals,
        enable_common_motion=True,
        common_motion_min_pairs=4,
        common_motion_max_normalized_step=10.0,
        common_motion_max_normalized_residual=0.25,
        birth_min_hits=100,
    )

    assert summary.common_motion_steps == 0


def test_interpolates_one_missing_frame_when_enabled(tmp_path: Path) -> None:
    seeds = [row(1, 1, 0.0, width=10.0, height=10.0)]
    proposals = [
        row(1, 1, 0.0, width=10.0, height=10.0),
        row(2, 1, 2.0, width=10.0, height=10.0),
        row(4, 1, 6.0, width=10.0, height=10.0),
        row(5, 1, 8.0, width=10.0, height=10.0),
    ]

    summary, rows = run_tracker(
        tmp_path,
        seeds,
        proposals,
        max_link_gap=2,
        interpolate_max_gap=1,
        birth_min_hits=100,
    )

    selected = [item for item in rows if item.object_id == 1]
    assert [item.frame_id for item in selected] == [1, 2, 3, 4, 5]
    assert selected[2].x1 == pytest.approx(4.0)
    assert summary.interpolated_rows == 1


def test_border_birth_gate_keeps_entry_and_drops_interior_track(
    tmp_path: Path,
) -> None:
    seeds = [row(1, 1, 70.0)]
    proposals = [
        row(1, 1, 70.0),
        row(2, 1, 71.0),
        row(3, 1, 72.0),
        row(3, 2, 0.0),
        row(4, 2, 3.0),
        row(5, 2, 7.0),
        row(3, 3, 45.0),
        row(4, 3, 47.0),
        row(5, 3, 49.0),
    ]

    summary, rows = run_tracker(
        tmp_path,
        seeds,
        proposals,
        image_width=100.0,
        image_height=100.0,
        border_margin_fraction=0.1,
        birth_require_border_entry=True,
        birth_min_inward_motion=0.25,
        birth_min_hits=3,
        birth_min_span=2,
    )

    by_id: dict[int, list[float]] = {}
    for item in rows:
        by_id.setdefault(item.object_id, []).append(item.x1)
    assert set(by_id) == {1, 2}
    assert by_id[2] == [0.0, 3.0, 7.0]
    assert summary.confirmed_birth_paths == 1
    assert summary.dropped_unseeded_paths == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"common_motion_min_pairs": 0}, "common_motion_min_pairs"),
        ({"enable_common_motion": 1}, "enable_common_motion"),
        ({"interpolate_max_gap": -1}, "interpolate_max_gap"),
        (
            {"birth_require_border_entry": True},
            "required for border-gated births",
        ),
    ],
)
def test_rejects_invalid_improvement_parameters(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    write_rows(labels / "SEQ.txt", [row(1, 1, 0.0)])
    write_rows(proposals / "SEQ.txt", [row(1, 1, 0.0)])

    with pytest.raises(ValueError, match=message):
        track_proposal_graph(
            proposals,
            labels,
            tmp_path / "output",
            **kwargs,
        )
