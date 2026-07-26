from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from raft_uav.mmuad import candidate_temporal_consensus_train_cv_cli as cli


def test_temporal_consensus_train_cv_cli_preserves_truth_sequence_ids(
    monkeypatch,
    tmp_path,
) -> None:
    truth_csv = tmp_path / "truth.csv"
    truth_csv.write_text(
        "sequence_id,time_s,x_m,y_m,z_m\n001,0.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        cli,
        "load_candidate_file",
        lambda path: SimpleNamespace(
            rows=pd.DataFrame(
                {
                    "sequence_id": ["001"],
                    "time_s": [0.0],
                    "x_m": [0.0],
                    "y_m": [0.0],
                    "z_m": [0.0],
                }
            )
        ),
    )

    def fake_select(candidates, truth, **kwargs):
        captured["sequence_id"] = truth.loc[0, "sequence_id"]
        return (
            {"selected_config_id": "cfg", "selected_metric_value": 0.0},
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    monkeypatch.setattr(
        cli,
        "select_temporal_consensus_config_by_sequence_cv",
        fake_select,
    )

    assert cli.main(
        [
            "--candidate-csv",
            str(tmp_path / "candidates.csv"),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    ) == 0
    assert captured["sequence_id"] == "001"
