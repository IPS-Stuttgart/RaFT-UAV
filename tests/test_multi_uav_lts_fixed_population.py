from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.fixed_population import postprocess_fixed_population


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_maps_seed_ids_and_drops_late_births(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(
        labels / "S.txt",
        "1,7,0,0,10,10,1,1,1\n1,9,100,0,10,10,1,1,1\n",
    )
    _write(
        predictions / "S.txt",
        "1,1,0,0,10,10,1,1,1\n"
        "1,2,100,0,10,10,1,1,1\n"
        "2,1,1,0,10,10,0.8,1,1\n"
        "2,2,99,0,10,10,0.8,1,1\n"
        "2,3,50,50,10,10,0.9,1,1\n",
    )

    summary = postprocess_fixed_population(predictions, labels, output)

    assert summary.mapped_input_tracks == 2
    assert summary.dropped_input_tracks == 1
    assert (output / "S.txt").read_text(encoding="utf-8").splitlines() == [
        "1,7,0,0,10,10,1,1,1",
        "1,9,100,0,10,10,1,1,1",
        "2,7,1,0,10,10,0.8,1,1",
        "2,9,99,0,10,10,0.8,1,1",
    ]


def test_inserts_unmatched_seed_row(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(predictions / "S.txt", "")

    summary = postprocess_fixed_population(predictions, labels, output)

    assert summary.inserted_seed_rows == 1
    assert (output / "S.txt").read_text(encoding="utf-8") == (
        "1,7,0,0,10,10,1,1,1\n"
    )


def test_relinks_nearby_fragment_to_seed(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(
        predictions / "S.txt",
        "1,1,0,0,10,10,1,1,1\n"
        "2,1,1,0,10,10,0.9,1,1\n"
        "4,5,3,0,10,10,0.8,1,1\n"
        "5,5,4,0,10,10,0.8,1,1\n",
    )

    summary = postprocess_fixed_population(
        predictions,
        labels,
        output,
        relink_max_gap=2,
        relink_max_cost=1.0,
    )

    assert summary.relinked_tracklets == 1
    lines = (output / "S.txt").read_text(encoding="utf-8").splitlines()
    assert lines[-2:] == [
        "4,7,3,0,10,10,0.8,1,1",
        "5,7,4,0,10,10,0.8,1,1",
    ]


def test_does_not_relink_distant_fragment(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(
        predictions / "S.txt",
        "1,1,0,0,10,10,1,1,1\n"
        "2,1,1,0,10,10,0.9,1,1\n"
        "4,5,100,0,10,10,0.8,1,1\n",
    )

    summary = postprocess_fixed_population(
        predictions,
        labels,
        output,
        relink_max_gap=2,
        relink_max_cost=1.0,
    )

    assert summary.relinked_tracklets == 0
    assert "4,7" not in (output / "S.txt").read_text(encoding="utf-8")


def test_single_frame_gap_interpolation(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(
        predictions / "S.txt",
        "1,1,0,0,10,10,1,1,1\n3,1,2,0,10,10,0.8,1,1\n",
    )

    summary = postprocess_fixed_population(
        predictions,
        labels,
        output,
        interpolate_single_frame=True,
    )

    assert summary.interpolated_rows == 1
    assert (output / "S.txt").read_text(encoding="utf-8").splitlines() == [
        "1,7,0,0,10,10,1,1,1",
        "2,7,1,0,10,10,0.8,1,1",
        "3,7,2,0,10,10,0.8,1,1",
    ]


def test_invalid_seed_iou_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_seed_iou"):
        postprocess_fixed_population(
            tmp_path / "predictions",
            tmp_path / "labels",
            tmp_path / "output",
            min_seed_iou=1.1,
        )
