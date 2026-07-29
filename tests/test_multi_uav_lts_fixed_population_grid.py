from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts.fixed_population_grid import run_fixed_population_grid


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_grid_ranks_identity_preserving_configuration(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "grid"
    _write(
        truth / "S.txt",
        "1,7,0,0,10,10,1,1,1\n"
        "2,7,1,0,10,10,1,1,1\n"
        "3,7,2,0,10,10,1,1,1\n",
    )
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(
        predictions / "S.txt",
        "1,1,0,0,10,10,1,1,1\n"
        "2,1,1,0,10,10,1,1,1\n"
        "3,5,2,0,10,10,1,1,1\n",
    )

    rows = run_fixed_population_grid(
        predictions,
        truth,
        labels,
        output,
        min_seed_ious=(0.5,),
        relink_max_gaps=(0, 1),
        relink_max_costs=(1.0,),
        interpolation_options=(False,),
    )

    assert rows[0].relink_max_gap == 1
    assert rows[0].codabench_hota == 1.0
    assert rows[0].codabench_idf1 == 1.0
    assert rows[0].hota == 1.0
    assert rows[0].idf1 == 1.0
    assert (output / "best_predictions" / "S.txt").exists()
    assert (output / "grid_ranking.csv").exists()
    assert (output / "grid_summary.json").exists()
