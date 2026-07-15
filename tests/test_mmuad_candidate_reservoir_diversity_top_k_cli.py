from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import raft_uav.mmuad.candidate_reservoir_diversity as diversity


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    input_csv = tmp_path / "reservoir.csv"
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["track-1"],
            "candidate_branch": ["raw"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
            "candidate_reservoir_score": [0.9],
            "confidence": [0.9],
        }
    ).to_csv(input_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    ).to_csv(truth_csv, index=False)
    return input_csv, truth_csv


@pytest.mark.parametrize(
    ("top_k_args", "expected"),
    [
        ([], (1, 3, 5, 10, 20)),
        (["--top-k", "2", "--top-k", "7"], (2, 7)),
    ],
)
def test_diversity_cap_cli_custom_top_k_replaces_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    top_k_args: list[str],
    expected: tuple[int, ...],
) -> None:
    input_csv, truth_csv = _write_inputs(tmp_path)
    captured: dict[str, tuple[int, ...]] = {}

    def fake_oracle_tables(
        candidates: pd.DataFrame,
        truth: pd.DataFrame,
        *,
        top_k_values: tuple[int, ...],
        max_truth_time_delta_s: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        del candidates, truth, max_truth_time_delta_s
        captured["top_k_values"] = top_k_values
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(diversity._IMPL, "build_oracle_recall_tables", fake_oracle_tables)
    status = diversity.main(
        [
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(tmp_path / "capped.csv"),
            "--truth-csv",
            str(truth_csv),
            *top_k_args,
        ]
    )

    assert status == 0
    assert captured["top_k_values"] == expected
