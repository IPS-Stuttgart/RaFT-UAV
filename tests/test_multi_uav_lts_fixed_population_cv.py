from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts.fixed_population_cv import (
    build_stratified_folds,
    run_fixed_population_cv,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_stratified_folds_distribute_each_scenario() -> None:
    sequences = ("C_00", "C_01", "T_00", "T_01", "BB2P_00", "BB2P_01")

    folds = build_stratified_folds(sequences, fold_count=2, seed=7)

    assert len(folds) == 2
    for fold in folds:
        assert {sequence.split("_", 1)[0] for sequence in fold} == {"C", "T", "BB2P"}


def test_cv_selects_relinking_by_held_out_hota(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "cv"
    for sequence, x_offset in (("C_00", 0), ("C_01", 20)):
        _write(
            truth / f"{sequence}.txt",
            f"1,7,{x_offset},0,10,10,1,1,1\n"
            f"2,7,{x_offset + 1},0,10,10,1,1,1\n"
            f"3,7,{x_offset + 2},0,10,10,1,1,1\n",
        )
        _write(
            labels / f"{sequence}.txt",
            f"1,7,{x_offset},0,10,10,1,1,1\n",
        )
        _write(
            predictions / f"{sequence}.txt",
            f"1,1,{x_offset},0,10,10,1,1,1\n"
            f"2,1,{x_offset + 1},0,10,10,1,1,1\n"
            f"3,5,{x_offset + 2},0,10,10,1,1,1\n",
        )

    rows = run_fixed_population_cv(
        predictions,
        truth,
        labels,
        output,
        fold_count=2,
        seed=0,
        min_seed_ious=(0.5,),
        relink_max_gaps=(0, 1),
        relink_max_costs=(1.0,),
        interpolation_options=(False,),
    )

    assert rows[0].relink_max_gap == 1
    assert rows[0].mean_codabench_hota == 1.0
    assert rows[0].std_codabench_hota == 0.0
    assert rows[0].mean_codabench_idf1 == 1.0
    assert rows[0].mean_hota == 1.0
    assert rows[0].std_hota == 0.0
    assert rows[0].mean_idf1 == 1.0
    assert (output / "best_predictions" / "C_00.txt").exists()
    assert (output / "cv_ranking.csv").exists()
    assert (output / "fold_assignments.csv").exists()
