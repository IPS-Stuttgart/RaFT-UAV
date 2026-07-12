from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad import candidate_pair_forward_backward_adaptive as adaptive
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


def _summary_candidates() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "source": "lidar_360",
                "track_id": "adaptive-top",
                "candidate_branch": "raw",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "candidate_pair_forward_backward_local_posterior": 0.9,
                "candidate_pair_forward_backward_adaptive_score": 0.8,
                "custom_pair_score": 0.1,
                "candidate_pair_forward_backward_adaptive_pair_weight": 0.0,
                "candidate_pair_forward_backward_adaptive_confidence": 0.0,
                "candidate_pair_forward_backward_adaptive_normalized_entropy": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "source": "dynamic",
                "track_id": "pair-top",
                "candidate_branch": "dynamic",
                "x_m": 2.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "candidate_pair_forward_backward_local_posterior": 0.1,
                "candidate_pair_forward_backward_adaptive_score": 0.2,
                "custom_pair_score": 0.9,
                "candidate_pair_forward_backward_adaptive_pair_weight": 0.0,
                "candidate_pair_forward_backward_adaptive_confidence": 0.0,
                "candidate_pair_forward_backward_adaptive_normalized_entropy": 1.0,
            },
        ]
    )


def test_adaptive_summary_uses_configured_pair_score_column() -> None:
    summary = adaptive.entropy_adaptive_pair_summary(
        _summary_candidates(),
        pair_score_column="custom_pair_score",
    )

    assert summary["pair_score_column"] == "custom_pair_score"
    assert summary["adaptive_top_differs_from_pair_fraction"] == pytest.approx(1.0)


def test_output_writer_forwards_configured_pair_score_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = CandidateFrame(normalize_candidate_columns(_summary_candidates()))
    output_csv = tmp_path / "adaptive.csv"
    summary_json = tmp_path / "adaptive.json"

    def _fake_original_writer(
        source: CandidateFrame,
        *,
        output_csv: Path,
        summary_json: Path | None = None,
        **_: object,
    ) -> None:
        source.rows.to_csv(output_csv, index=False)
        if summary_json is not None:
            summary_json.write_text(
                json.dumps({"adaptive_summary": {"stale": True}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        adaptive,
        "_original_write_entropy_adaptive_pair_outputs",
        _fake_original_writer,
    )
    adaptive.write_entropy_adaptive_pair_outputs(
        candidates,
        output_csv=output_csv,
        summary_json=summary_json,
        pair_config=adaptive.CandidatePairForwardBackwardConfig(
            output_score_column="custom_pair_score",
        ),
    )

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    summary = payload["adaptive_summary"]
    assert summary["pair_score_column"] == "custom_pair_score"
    assert summary["adaptive_top_differs_from_pair_fraction"] == pytest.approx(1.0)
