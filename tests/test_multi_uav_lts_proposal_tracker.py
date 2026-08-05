from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.proposal_tracker import track_fixed_label_proposals


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_tracks_seeded_identities_through_crossing(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    output = tmp_path / "output"
    _write(
        labels / "S.txt",
        "1,7,0,0,10,10,1,1,1\n"
        "1,9,100,0,10,10,1,1,1\n",
    )
    _write(
        proposals / "S.txt",
        "2,1,30,0,10,10,0.9,1,1\n"
        "2,2,70,0,10,10,0.9,1,1\n"
        "3,1,60,0,10,10,0.9,1,1\n"
        "3,2,40,0,10,10,0.9,1,1\n",
    )

    summary = track_fixed_label_proposals(
        {"detector": proposals},
        labels,
        output,
        velocity_smoothing=1.0,
        global_motion_smoothing=0.0,
        missed_cost=10.0,
    )

    assert summary.assigned_rows == 4
    assert _lines(output / "S.txt") == [
        "1,7,0,0,10,10,1,1,1",
        "1,9,100,0,10,10,1,1,1",
        "2,7,30,0,10,10,0.9,1,1",
        "2,9,70,0,10,10,0.9,1,1",
        "3,7,60,0,10,10,0.9,1,1",
        "3,9,40,0,10,10,0.9,1,1",
    ]


def test_one_candidate_cannot_fill_two_identities(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    output = tmp_path / "output"
    _write(
        labels / "S.txt",
        "1,7,0,0,10,10,1,1,1\n"
        "1,9,20,0,10,10,1,1,1\n",
    )
    _write(proposals / "S.txt", "2,1,10,0,10,10,0.9,1,1\n")

    summary = track_fixed_label_proposals(
        {"detector": proposals},
        labels,
        output,
        max_center_distance=10.0,
        missed_cost=10.0,
    )

    assert summary.assigned_rows == 1
    assert len([line for line in _lines(output / "S.txt") if line.startswith("2,")]) == 1


def test_fuses_complementary_sources(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(source_a / "S.txt", "2,1,1,0,10,10,0.9,1,1\n")
    _write(source_b / "S.txt", "3,1,2,0,10,10,0.8,1,1\n")

    summary = track_fixed_label_proposals(
        {"global": source_a, "local": source_b},
        labels,
        output,
        missed_cost=10.0,
    )

    assert summary.assignments_by_source == {"global": 1, "local": 1}
    assert _lines(output / "S.txt")[-2:] == [
        "2,7,1,0,10,10,0.9,1,1",
        "3,7,2,0,10,10,0.8,1,1",
    ]


def test_near_duplicate_sources_are_not_double_assigned(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    output = tmp_path / "output"
    _write(
        labels / "S.txt",
        "1,7,0,0,10,10,1,1,1\n"
        "1,9,20,0,10,10,1,1,1\n",
    )
    _write(source_a / "S.txt", "2,1,5,0,10,10,0.9,1,1\n")
    _write(source_b / "S.txt", "2,1,5,0,10,10,0.8,1,1\n")

    summary = track_fixed_label_proposals(
        {"a": source_a, "b": source_b},
        labels,
        output,
        max_center_distance=10.0,
        missed_cost=10.0,
    )

    assert summary.candidate_proposals == 1
    assert summary.assigned_rows == 1


def test_optional_one_frame_coasting(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(proposals / "S.txt", "2,1,2,0,10,10,0.8,1,1\n")
    sequence_root = tmp_path / "images"
    _write(sequence_root / "S" / "0001.jpg", "x")
    _write(sequence_root / "S" / "0002.jpg", "x")
    _write(sequence_root / "S" / "0003.jpg", "x")

    summary = track_fixed_label_proposals(
        {"detector": proposals},
        labels,
        output,
        sequence_root=sequence_root,
        coast_frames=1,
        velocity_smoothing=1.0,
        global_motion_smoothing=0.0,
        missed_cost=10.0,
    )

    assert summary.coasted_rows == 1
    assert _lines(output / "S.txt")[-1].startswith("3,7,4,0,10,10,0.4")


def test_far_proposal_does_not_create_unseeded_birth(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(proposals / "S.txt", "2,99,1000,1000,10,10,1,1,1\n")

    summary = track_fixed_label_proposals(
        {"detector": proposals},
        labels,
        output,
        max_center_distance=2.0,
    )

    assert summary.assigned_rows == 0
    assert _lines(output / "S.txt") == ["1,7,0,0,10,10,1,1,1"]


def test_rejects_output_alias(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    proposals = tmp_path / "proposals"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(proposals / "S.txt", "")

    with pytest.raises(ValueError, match="output directory"):
        track_fixed_label_proposals(
            {"detector": proposals},
            labels,
            proposals,
        )


def test_rejects_coast_longer_than_track_memory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="coast_frames"):
        track_fixed_label_proposals(
            {"detector": tmp_path / "proposals"},
            tmp_path / "labels",
            tmp_path / "output",
            max_missed_frames=1,
            coast_frames=2,
        )
