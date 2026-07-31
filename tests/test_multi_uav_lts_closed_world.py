from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.closed_world import postprocess_closed_world


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_reassigns_upstream_ids_after_crossing(tmp_path: Path) -> None:
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
        "2,1,10,0,10,10,0.9,1,1\n"
        "2,2,90,0,10,10,0.9,1,1\n"
        "3,1,80,0,10,10,0.9,1,1\n"
        "3,2,20,0,10,10,0.9,1,1\n",
    )

    summary = postprocess_closed_world(predictions, labels, output)

    assert summary.absorbed_source_switches == 2
    assert (output / "S.txt").read_text(encoding="utf-8").splitlines()[-2:] == [
        "3,7,20,0,10,10,0.9,1,1",
        "3,9,80,0,10,10,0.9,1,1",
    ]


def test_suppresses_candidates_not_supported_by_seed_bank(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(
        predictions / "S.txt",
        "1,1,0,0,10,10,1,1,1\n"
        "2,1,1,0,10,10,0.9,1,1\n"
        "2,99,100,100,10,10,0.99,1,1\n",
    )

    summary = postprocess_closed_world(predictions, labels, output)

    assert summary.dropped_candidate_rows == 1
    assert (output / "S.txt").read_text(encoding="utf-8").splitlines() == [
        "1,7,0,0,10,10,1,1,1",
        "2,7,1,0,10,10,0.9,1,1",
    ]


def test_uncertainty_gated_coasting_fills_short_gap(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(
        predictions / "S.txt",
        "1,1,0,0,10,10,1,1,1\n"
        "2,1,1,0,10,10,0.9,1,1\n"
        "4,5,3,0,10,10,0.8,1,1\n",
    )

    summary = postprocess_closed_world(
        predictions,
        labels,
        output,
        max_gap=2,
        emit_coasts=True,
        coast_max_gap=1,
    )

    assert summary.coasted_rows == 1
    assert (output / "S.txt").read_text(encoding="utf-8").splitlines() == [
        "1,7,0,0,10,10,1,1,1",
        "2,7,1,0,10,10,0.9,1,1",
        "3,7,2,0,10,10,0.765,1,0.85",
        "4,7,3,0,10,10,0.8,1,1",
    ]


def test_coasting_rejects_unstable_motion(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(
        predictions / "S.txt",
        "1,1,0,0,10,10,1,1,1\n"
        "2,1,1,0,10,10,0.9,1,1\n"
        "3,1,20,0,10,10,0.9,1,1\n"
        "5,1,22,0,10,10,0.9,1,1\n",
    )

    summary = postprocess_closed_world(
        predictions,
        labels,
        output,
        max_gap=2,
        emit_coasts=True,
        coast_max_gap=1,
        max_coast_uncertainty=0.5,
    )

    assert summary.coasted_rows == 0
    assert not any(
        line.startswith("4,")
        for line in (output / "S.txt").read_text(encoding="utf-8").splitlines()
    )


def test_rejects_unknown_prediction_sequence(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(predictions / "TYPO.txt", "1,1,0,0,10,10,1,1,1\n")

    with pytest.raises(ValueError, match="unknown sequence files"):
        postprocess_closed_world(predictions, labels, tmp_path / "output")


def test_invalid_controls_are_rejected_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="tiny_scale"):
        postprocess_closed_world(
            tmp_path / "predictions",
            tmp_path / "labels",
            output,
            tiny_scale=0.0,
        )

    assert not output.exists()
