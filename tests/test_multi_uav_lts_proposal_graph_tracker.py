from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

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
    width: float = 10.0,
    height: float = 10.0,
    confidence: float = 0.9,
) -> Detection:
    return Detection(frame, proposal_id, x, y, width, height, confidence, 1, 1.0)


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
    summary = track_proposal_graph(
        proposal_dir,
        labels,
        output,
        **kwargs,
    )
    rows = parse_detection_text(
        (output / "SEQ.txt").read_text(encoding="utf-8"),
        source="output",
    )
    return summary, rows


def test_confirms_persistent_late_birth_and_drops_one_frame_noise(tmp_path: Path) -> None:
    seeds = [row(1, 1, 0.0)]
    proposals = [row(frame, 1, float(frame - 1)) for frame in range(1, 6)]
    proposals.extend(row(frame, 2, 100.0 + frame) for frame in range(3, 6))
    proposals.append(row(2, 3, 200.0, confidence=0.8))

    summary, rows = run_tracker(tmp_path, seeds, proposals)

    by_id = {}
    for item in rows:
        by_id.setdefault(item.object_id, []).append(item)
    assert set(by_id) == {1, 2}
    assert [item.frame_id for item in by_id[1]] == [1, 2, 3, 4, 5]
    assert [item.frame_id for item in by_id[2]] == [3, 4, 5]
    assert summary.confirmed_birth_paths == 1
    assert summary.dropped_unseeded_paths == 1


def test_global_link_bridges_a_detector_gap(tmp_path: Path) -> None:
    seeds = [row(1, 1, 0.0)]
    proposals = [
        row(1, 1, 0.0),
        row(2, 1, 2.0),
        row(5, 1, 8.0),
        row(6, 1, 10.0),
    ]

    summary, rows = run_tracker(
        tmp_path,
        seeds,
        proposals,
        max_link_gap=3,
        birth_min_hits=3,
    )

    assert [(item.frame_id, item.object_id) for item in rows] == [
        (1, 1),
        (2, 1),
        (5, 1),
        (6, 1),
    ]
    assert summary.graph_links == 1
    assert summary.confirmed_birth_paths == 0


def test_delayed_global_link_preserves_two_crossing_identities(tmp_path: Path) -> None:
    seeds = [row(1, 1, 0.0, width=4.0), row(1, 2, 20.0, width=4.0)]
    proposals = [
        row(1, 1, 0.0, width=4.0),
        row(1, 2, 20.0, width=4.0),
        row(2, 1, 4.0, width=4.0),
        row(2, 2, 16.0, width=4.0),
        row(4, 1, 12.0, width=4.0),
        row(4, 2, 8.0, width=4.0),
        row(5, 1, 16.0, width=4.0),
        row(5, 2, 4.0, width=4.0),
    ]

    summary, rows = run_tracker(
        tmp_path,
        seeds,
        proposals,
        max_link_gap=2,
        max_link_cost=2.5,
        anchor_max_cost=2.0,
        birth_min_hits=3,
    )

    paths = {}
    for item in rows:
        paths.setdefault(item.object_id, []).append(item.x1)
    assert paths[1] == [0.0, 4.0, 12.0, 16.0]
    assert paths[2] == [20.0, 16.0, 8.0, 4.0]
    assert summary.graph_links == 2


def test_border_discount_recovers_a_long_border_gap(tmp_path: Path) -> None:
    seeds = [row(1, 1, 80.0)]
    proposals = [
        row(1, 1, 80.0),
        row(2, 1, 88.0),
        row(8, 1, 88.0),
        row(9, 1, 88.0),
    ]

    _summary, rows = run_tracker(
        tmp_path,
        seeds,
        proposals,
        image_width=100.0,
        image_height=100.0,
        border_margin_fraction=0.2,
        border_gap_discount=0.1,
        gap_weight=0.4,
        max_link_gap=10,
        max_link_cost=2.0,
        birth_min_hits=3,
    )

    assert [item.frame_id for item in rows if item.object_id == 1] == [1, 2, 8, 9]


def test_without_global_links_the_gap_fragment_is_not_attached(tmp_path: Path) -> None:
    seeds = [row(1, 1, 0.0)]
    proposals = [
        row(1, 1, 0.0),
        row(2, 1, 2.0),
        row(5, 1, 8.0),
        row(6, 1, 10.0),
    ]

    summary, rows = run_tracker(
        tmp_path,
        seeds,
        proposals,
        enable_global_links=False,
        birth_min_hits=3,
    )

    assert [item.frame_id for item in rows] == [1, 2]
    assert summary.graph_links == 0
    assert summary.dropped_unseeded_paths == 1


def test_suppresses_near_duplicate_proposals_before_tracking(tmp_path: Path) -> None:
    seeds = [row(1, 1, 0.0)]
    proposals = [
        row(1, 1, 0.0, confidence=0.9),
        row(1, 2, 0.1, confidence=0.8),
        row(2, 1, 1.0, confidence=0.9),
        row(2, 2, 1.1, confidence=0.8),
    ]

    summary, rows = run_tracker(tmp_path, seeds, proposals, duplicate_iou=0.9)

    assert len(rows) == 2
    assert summary.duplicate_suppressed_rows == 2


def test_reads_root_level_proposal_zip(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    output = tmp_path / "output"
    write_rows(labels / "SEQ.txt", [row(1, 7, 0.0)])
    proposal_rows = [row(1, 1, 0.0), row(2, 1, 1.0), row(3, 1, 2.0)]
    proposal_zip = tmp_path / "proposals.zip"
    with ZipFile(proposal_zip, "w") as archive:
        archive.writestr(
            "SEQ.txt",
            "".join(format_detection(item) + "\n" for item in proposal_rows),
        )

    summary = track_proposal_graph(proposal_zip, labels, output)
    rows = parse_detection_text(
        (output / "SEQ.txt").read_text(encoding="utf-8"),
        source="output",
    )

    assert summary.output_ids == 1
    assert {item.object_id for item in rows} == {7}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"birth_min_hits": 0}, "birth_min_hits"),
        ({"image_width": 100.0}, "supplied together"),
        ({"enable_global_links": 1}, "Boolean"),
        ({"center_weight": 0.0, "size_weight": 0.0, "iou_weight": 0.0}, "geometric"),
    ],
)
def test_rejects_invalid_parameters(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    write_rows(labels / "SEQ.txt", [row(1, 1, 0.0)])
    write_rows(proposals / "SEQ.txt", [row(1, 1, 0.0)])

    with pytest.raises(ValueError, match=message):
        track_proposal_graph(proposals, labels, tmp_path / "output", **kwargs)


def test_rejects_output_nested_in_proposal_input(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    write_rows(labels / "SEQ.txt", [row(1, 1, 0.0)])
    write_rows(proposals / "SEQ.txt", [row(1, 1, 0.0)])

    with pytest.raises(ValueError, match="nested in proposals"):
        track_proposal_graph(proposals, labels, proposals / "output")


def test_zero_seed_iou_does_not_match_a_disjoint_proposal(tmp_path: Path) -> None:
    seeds = [row(1, 1, 0.0)]
    proposals = [
        row(1, 1, 100.0),
        row(2, 1, 101.0),
        row(3, 1, 102.0),
    ]

    summary, rows = run_tracker(
        tmp_path,
        seeds,
        proposals,
        min_seed_iou=0.0,
    )

    by_id = {}
    for item in rows:
        by_id.setdefault(item.object_id, []).append(item)
    assert [item.frame_id for item in by_id[1]] == [1]
    assert [item.frame_id for item in by_id[2]] == [1, 2, 3]
    assert summary.confirmed_birth_paths == 1


def test_preserves_an_unmatched_seed_as_an_exact_singleton(tmp_path: Path) -> None:
    seed = row(1, 7, 12.5, y=4.5, width=3.0, height=5.0)

    summary, rows = run_tracker(tmp_path, [seed], [])

    assert rows == [seed]
    assert summary.seeded_paths == 1
    assert summary.output_ids == 1


def test_rejects_out_of_domain_proposal_confidence(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    write_rows(labels / "SEQ.txt", [row(1, 1, 0.0)])
    write_rows(proposals / "SEQ.txt", [row(1, 1, 0.0, confidence=-0.1)])

    with pytest.raises(ValueError, match=r"confidence must be in \[0, 1\]"):
        track_proposal_graph(proposals, labels, tmp_path / "output")
