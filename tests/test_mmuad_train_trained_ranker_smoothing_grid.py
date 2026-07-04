import argparse
from pathlib import Path

from scripts import mmuad_train_trained_ranker_smoothing_grid as smoothing_grid


def test_estimate_files_excludes_output_dir_with_mixed_relative_absolute_paths(
    tmp_path, monkeypatch
):
    root = tmp_path / "ranker"
    source = root / "run_a" / "mmuad_estimates.csv"
    generated = root / "outputs" / "run_b" / "mmuad_estimates.csv"
    source.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    source.write_text(
        "sequence_id,time_s,state_x_m,state_y_m,state_z_m\n",
        encoding="utf-8",
    )
    generated.write_text(
        "sequence_id,time_s,state_x_m,state_y_m,state_z_m\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        base_estimates_csv=[],
        ranker_output_dir=root,
        output_dir=Path("ranker/outputs"),
    )

    files = smoothing_grid._estimate_files(args)

    assert [path.resolve() for path in files] == [source.resolve()]


def test_ranker_run_name_handles_absolute_path_with_relative_root(tmp_path, monkeypatch):
    root = tmp_path / "ranker"
    path = root / "run_a" / "mmuad_estimates.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "sequence_id,time_s,state_x_m,state_y_m,state_z_m\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    assert smoothing_grid._ranker_run_name(path.resolve(), Path("ranker")) == "run_a"
