from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from raft_uav.mmuad import candidate_assignment_report


def test_candidate_assignment_report_cli_preserves_zero_padded_sequence_ids(
    tmp_path,
    monkeypatch,
) -> None:
    assignments_path = tmp_path / "assignments.csv"
    assignments_path.write_text(
        "sequence_id,time_s,frame_index,selected_candidate_error_m\n"
        "001,0.0,0,1.5\n",
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.csv"
    truth_path.write_text("unused\n", encoding="utf-8")
    captured: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(
        candidate_assignment_report._IMPL,
        "load_evaluation_truth_file",
        lambda _path: SimpleNamespace(rows=pd.DataFrame()),
    )

    def fake_run_candidate_assignment_report(**kwargs):
        captured["assignments"] = kwargs["assignments"].copy()
        return {
            "frame_count": 0,
            "block_count": 0,
            "action_count": 0,
            "pooled": {},
            "top_action": {},
            "paths": {},
        }

    monkeypatch.setattr(
        candidate_assignment_report._IMPL,
        "run_candidate_assignment_report",
        fake_run_candidate_assignment_report,
    )
    pandas_module = candidate_assignment_report._IMPL.pd

    result = candidate_assignment_report.main(
        [
            "--assignments-csv",
            str(assignments_path),
            "--truth-csv",
            str(truth_path),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert result == 0
    assert candidate_assignment_report._IMPL.pd is pandas_module
    assignments = captured["assignments"]
    assert assignments["sequence_id"].tolist() == ["001"]
    assert assignments["time_s"].tolist() == ["0.0"]
