from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.fixed_population_beam import (
    postprocess_fixed_population_beam,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_beam_postprocessor_preserves_seed_ids_after_delayed_choice(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(
        labels / "S.txt",
        "1,7,-5,0,10,10,1,1,1\n"
        "1,9,5,0,10,10,1,1,1\n",
    )
    _write(
        predictions / "S.txt",
        "1,1,-5,0,10,10,1,1,1\n"
        "1,2,5,0,10,10,1,1,1\n"
        "2,1,-5,0,10,10,1,1,1\n"
        "2,2,5,0,10,10,1,1,1\n"
        "3,3,-1,0,10,10,1,1,1\n"
        "4,3,3,0,10,10,1,1,1\n"
        "5,4,-5,0,10,10,1,1,1\n"
        "6,4,-5,0,10,10,1,1,1\n",
    )

    summary = postprocess_fixed_population_beam(
        predictions,
        labels,
        output,
        relink_max_gap=2,
        relink_max_cost=2.0,
        relink_beam_width=2,
        relink_drop_cost=2.0,
        relink_velocity_weight=0.0,
    )

    lines = (output / "S.txt").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("3,9,") for line in lines)
    assert any(line.startswith("5,7,") for line in lines)
    assert summary.relinked_tracklets == 2
    assert summary.sequences[0].evaluated_hypotheses > 1
    assert summary.sequences[0].second_best_margin is not None


def test_rejects_zero_beam_width_before_input_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relink_beam_width"):
        postprocess_fixed_population_beam(
            tmp_path / "predictions",
            tmp_path / "labels",
            tmp_path / "output",
            relink_beam_width=0,
        )
