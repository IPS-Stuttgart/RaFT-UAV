from __future__ import annotations

import pandas as pd

from raft_uav.mmuad import candidate_pool_compare


def test_candidate_pool_compare_direct_cli_replaces_default_top_k(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        candidate_pool_compare,
        "load_candidate_inputs",
        lambda specs: pd.DataFrame({"sequence_id": ["s1"], "time_s": [0.0]}),
    )
    monkeypatch.setattr(
        candidate_pool_compare,
        "_load_labeled_candidate_pools",
        lambda specs: {"pool": pd.DataFrame({"sequence_id": ["s1"], "time_s": [0.0]})},
    )
    monkeypatch.setattr(
        candidate_pool_compare.pd,
        "read_csv",
        lambda path: pd.DataFrame({"sequence_id": ["s1"], "time_s": [0.0]}),
    )

    def fake_build(*args, top_k_values, **kwargs):
        captured["top_k_values"] = top_k_values
        frame_rows = pd.DataFrame({"pool_label": ["pool"], "oracle_all_mse_delta": [0.0]})
        pooled = pd.DataFrame({"pool_label": ["pool"], "oracle_all_mse_delta": [0.0]})
        return frame_rows, pooled, pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(candidate_pool_compare, "build_candidate_pool_compare_tables", fake_build)
    monkeypatch.setattr(
        candidate_pool_compare,
        "write_candidate_pool_compare_outputs",
        lambda **kwargs: {"frame_csv": str(tmp_path / "frames.csv")},
    )

    assert candidate_pool_compare.main(
        [
            "--reference-candidate",
            "raw=input.csv",
            "--candidate",
            "ranked=input.csv",
            "--truth-csv",
            "truth.csv",
            "--output-dir",
            str(tmp_path),
            "--top-k",
            "3",
        ]
    ) == 0
    assert captured["top_k_values"] == (3,)
