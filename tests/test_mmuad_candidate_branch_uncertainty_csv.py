from __future__ import annotations

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from raft_uav.mmuad import candidate_branch_uncertainty


def test_branch_uncertainty_cli_normalizes_class_probability_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities_csv = tmp_path / "probabilities.csv"
    probabilities_csv.write_text(
        " Sequence , predicted_probability_0 , predicted_probability_1\n"
        "001,0.9,0.1\n",
        encoding="utf-8",
    )
    captured: dict[str, pd.DataFrame] = {}
    main_globals = candidate_branch_uncertainty.main.__globals__

    monkeypatch.setitem(main_globals, "load_candidate_file", lambda path: object())
    monkeypatch.setitem(
        main_globals,
        "load_evaluation_truth_file",
        lambda path: SimpleNamespace(rows=pd.DataFrame()),
    )

    def fake_train(candidates, truth, class_probabilities, **kwargs):
        captured["class_probabilities"] = pd.DataFrame(class_probabilities).copy()
        return object(), pd.DataFrame(), {}

    monkeypatch.setitem(
        main_globals,
        "train_branch_aware_candidate_uncertainty",
        fake_train,
    )
    monkeypatch.setitem(
        main_globals,
        "save_candidate_uncertainty_model",
        lambda model, path: None,
    )

    status = candidate_branch_uncertainty.main(
        [
            "train",
            "--candidates-csv",
            str(tmp_path / "candidates.csv"),
            "--truth-csv",
            str(tmp_path / "truth.csv"),
            "--class-probabilities-csv",
            str(probabilities_csv),
            "--model-json",
            str(tmp_path / "model.json"),
            "--model-type",
            "ridge",
        ]
    )

    assert status == 0
    probabilities = captured["class_probabilities"]
    assert probabilities["sequence_id"].tolist() == ["001"]
    assert probabilities.columns.tolist() == [
        "sequence_id",
        "Sequence",
        "predicted_probability_0",
        "predicted_probability_1",
    ]


def test_branch_uncertainty_package_supports_python_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "raft_uav.mmuad.candidate_branch_uncertainty"
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")

    assert exc_info.value.code == 0
