from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_branch_context import attach_candidate_branch_context
from raft_uav.mmuad.candidate_branch_context import main as branch_context_main
from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table, train_cluster_ranker
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqB"],
            "time_s": [0.0, 0.0, 0.0, 0.0],
            "source": ["lidar_360", "lidar_360", "livox_avia", "lidar_360"],
            "track_id": ["raw_good", "translated_bad", "dynamic", "raw_b"],
            "candidate_branch": ["raw", "source_translation", "dynamic", "raw"],
            "x_m": [0.0, 10.0, 2.0, 0.5],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "confidence": [0.2, 0.9, 0.5, 0.3],
            "cluster_point_count": [5, 20, 10, 7],
            "cluster_extent_3d_m": [0.5, 3.0, 1.0, 0.6],
        }
    )


def test_candidate_branch_context_adds_ranker_consumable_branch_features() -> None:
    augmented = attach_candidate_branch_context(
        CandidateFrame(_candidate_rows()),
        interaction_columns=("confidence", "cluster_point_count"),
    )
    rows = augmented.rows.sort_values("track_id").reset_index(drop=True)

    assert "image_candidate_branch_raw" in rows.columns
    assert "image_candidate_branch_source_translation" in rows.columns
    assert "image_candidate_branch_raw_x_cluster_point_count" in rows.columns
    raw_good = rows.loc[rows["track_id"] == "raw_good"].iloc[0]
    translated_bad = rows.loc[rows["track_id"] == "translated_bad"].iloc[0]
    assert raw_good["image_candidate_branch_raw"] == pytest.approx(1.0)
    assert raw_good["image_candidate_branch_source_translation"] == pytest.approx(0.0)
    assert raw_good["image_candidate_branch_raw_x_cluster_point_count"] == pytest.approx(5.0)
    assert translated_bad["image_candidate_branch_source_translation_x_confidence"] == pytest.approx(0.9)

    features = build_cluster_feature_table(augmented)
    model = train_cluster_ranker(
        features.assign(good_cluster=[True, False, True, True]),
        iterations=2,
        learning_rate=0.1,
    )
    assert "image_candidate_branch_raw" in model.feature_columns
    assert "image_candidate_branch_source_translation_x_confidence" in model.feature_columns


def test_candidate_branch_context_can_use_explicit_branch_column() -> None:
    rows = _candidate_rows().rename(columns={"candidate_branch": "candidate_stream"})

    augmented = attach_candidate_branch_context(
        CandidateFrame(rows),
        branch_column="candidate_stream",
        interaction_columns=("cluster_point_count",),
    )

    assert "image_candidate_branch_source_translation" in augmented.rows.columns
    assert set(augmented.rows["candidate_branch"].astype(str)) == {"dynamic", "raw", "source_translation"}


def test_candidate_branch_context_raises_for_missing_explicit_branch_column() -> None:
    with pytest.raises(ValueError, match="branch column"):
        attach_candidate_branch_context(
            CandidateFrame(_candidate_rows()),
            branch_column="does_not_exist",
        )


def test_candidate_branch_context_cli_writes_outputs(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    output = tmp_path / "branch_context_candidates.csv"
    provenance = tmp_path / "branch_context_provenance.json"
    _candidate_rows().to_csv(candidates, index=False)

    status = branch_context_main(
        [
            "--candidate-csv",
            str(candidates),
            "--output-csv",
            str(output),
            "--provenance-json",
            str(provenance),
            "--interaction-column",
            "cluster_point_count",
        ]
    )

    assert status == 0
    rows = pd.read_csv(output)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert "image_candidate_branch_dynamic" in rows.columns
    assert "image_candidate_branch_raw_x_cluster_point_count" in rows.columns
    assert payload["row_count"] == 4
    assert set(payload["candidate_branch_values"]) == {"dynamic", "raw", "source_translation"}
