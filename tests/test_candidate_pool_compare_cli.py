import pandas as pd

from raft_uav.mmuad import candidate_pool_compare_cli


def test_candidate_pool_compare_cli_replaces_default_top_k(monkeypatch, tmp_path):
    captured = {}
    truth_path = tmp_path / "truth.csv"
    pd.DataFrame({"sequence_id": ["s1"], "time_s": [0.0]}).to_csv(truth_path, index=False)

    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "load_candidate_inputs",
        lambda specs: pd.DataFrame({"sequence_id": ["s1"], "time_s": [0.0]}),
    )
    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "_load_labeled_candidate_pools",
        lambda specs: {"pool": pd.DataFrame({"sequence_id": ["s1"], "time_s": [0.0]})},
    )

    def fake_build(*args, top_k_values, **kwargs):
        captured["top_k_values"] = top_k_values
        frame_rows = pd.DataFrame({"pool_label": ["pool"], "oracle_all_mse_delta": [0.0]})
        pooled = pd.DataFrame({"pool_label": ["pool"], "oracle_all_mse_delta": [0.0]})
        return frame_rows, pooled, pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(candidate_pool_compare_cli, "build_candidate_pool_compare_tables", fake_build)
    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "write_candidate_pool_compare_outputs",
        lambda **kwargs: {"frame_csv": str(tmp_path / "frames.csv")},
    )

    assert candidate_pool_compare_cli.main(
        [
            "--reference-candidate",
            "raw=input.csv",
            "--candidate",
            "ranked=input.csv",
            "--truth-csv",
            str(truth_path),
            "--output-dir",
            str(tmp_path),
            "--top-k",
            "3",
        ]
    ) == 0
    assert captured["top_k_values"] == (3,)


def test_candidate_pool_compare_cli_preserves_opaque_truth_sequence_ids(monkeypatch, tmp_path):
    captured = {}
    truth_path = tmp_path / "truth.csv"
    truth_path.write_text(
        "Sequence,time_s,x_m,y_m,z_m\n001,0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "load_candidate_inputs",
        lambda specs: pd.DataFrame({"sequence_id": ["001"], "time_s": [0.0]}),
    )
    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "_load_labeled_candidate_pools",
        lambda specs: {"pool": pd.DataFrame({"sequence_id": ["001"], "time_s": [0.0]})},
    )

    def fake_build(reference_candidates, candidate_pools, truth, **kwargs):
        captured["truth_sequence_id"] = truth.loc[0, "Sequence"]
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(candidate_pool_compare_cli, "build_candidate_pool_compare_tables", fake_build)
    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "write_candidate_pool_compare_outputs",
        lambda **kwargs: {},
    )

    assert candidate_pool_compare_cli.main(
        [
            "--reference-candidate",
            "raw=input.csv",
            "--candidate",
            "ranked=input.csv",
            "--truth-csv",
            str(truth_path),
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0
    assert captured["truth_sequence_id"] == "001"
