from __future__ import annotations

import pytest

from raft_uav.mmuad import candidate_pool_compare_cli


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--top-k", "0"),
        ("--top-k", "-3"),
        ("--max-truth-time-delta-s", "nan"),
        ("--max-truth-time-delta-s", "inf"),
        ("--max-truth-time-delta-s", "-0.1"),
        ("--good-candidate-threshold-m", "nan"),
        ("--good-candidate-threshold-m", "-1"),
        ("--loss-tolerance-m", "nan"),
        ("--loss-tolerance-m", "-1"),
    ],
)
def test_candidate_pool_compare_cli_rejects_invalid_diagnostic_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    option: str,
    value: str,
) -> None:
    def unexpected_load(_: list[str]):
        pytest.fail("invalid CLI controls must fail before loading candidate data")

    monkeypatch.setattr(candidate_pool_compare_cli, "load_candidate_inputs", unexpected_load)

    with pytest.raises(SystemExit) as exc_info:
        candidate_pool_compare_cli.main(
            [
                "--reference-candidate",
                "raw=input.csv",
                "--candidate",
                "ranked=input.csv",
                "--truth-csv",
                "truth.csv",
                "--output-dir",
                str(tmp_path),
                option,
                value,
            ]
        )

    assert exc_info.value.code == 2


def test_candidate_pool_compare_cli_accepts_zero_distance_tolerances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "load_candidate_inputs",
        lambda specs: candidate_pool_compare_cli.pd.DataFrame(
            {"sequence_id": ["s1"], "time_s": [0.0]}
        ),
    )
    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "_load_labeled_candidate_pools",
        lambda specs: {
            "pool": candidate_pool_compare_cli.pd.DataFrame(
                {"sequence_id": ["s1"], "time_s": [0.0]}
            )
        },
    )
    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "_read_truth_csv",
        lambda path: candidate_pool_compare_cli.pd.DataFrame(
            {"sequence_id": ["s1"], "time_s": [0.0]}
        ),
    )

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        frame_rows = candidate_pool_compare_cli.pd.DataFrame(
            {"pool_label": ["pool"], "oracle_all_mse_delta": [0.0]}
        )
        pooled = candidate_pool_compare_cli.pd.DataFrame(
            {"pool_label": ["pool"], "oracle_all_mse_delta": [0.0]}
        )
        empty = candidate_pool_compare_cli.pd.DataFrame()
        return frame_rows, pooled, empty, empty

    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "build_candidate_pool_compare_tables",
        fake_build,
    )
    monkeypatch.setattr(
        candidate_pool_compare_cli,
        "write_candidate_pool_compare_outputs",
        lambda **kwargs: {"frame_csv": tmp_path / "frames.csv"},
    )

    assert (
        candidate_pool_compare_cli.main(
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
                "1",
                "--max-truth-time-delta-s",
                "0",
                "--good-candidate-threshold-m",
                "0",
                "--loss-tolerance-m",
                "0",
            ]
        )
        == 0
    )
    assert captured["top_k_values"] == (1,)
    assert captured["max_truth_time_delta_s"] == 0.0
    assert captured["good_candidate_threshold_m"] == 0.0
    assert captured["loss_tolerance_m"] == 0.0
