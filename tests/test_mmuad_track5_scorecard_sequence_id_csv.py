from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_scorecard import (
    _load_optional_csv,
    build_candidate_regret_summary,
    build_pose_by_sequence_table,
)


def test_scorecard_optional_csvs_preserve_numeric_sequence_identifiers(
    tmp_path: Path,
) -> None:
    selected_path = tmp_path / "selected.csv"
    selected_path.write_text(
        "sequence_id,time_s,source\n"
        "0001,0.0,radar-enhance-pcl\n",
        encoding="utf-8",
    )
    gap_path = tmp_path / "candidate_gap.csv"
    gap_path.write_text(
        "sequence,sequence_id,time_s,sensor,nearest_candidate_found,"
        "selected_candidate_found,selected_source_matches_sensor,"
        "candidate_count_at_nearest_time,selected_minus_truth_error_m,"
        "nearest_minus_truth_error_m,candidate_regret_m,"
        "nearest_candidate_time_delta_s\n"
        "0001,0001,0.0,radar,true,true,true,0,1.0,0.5,0.5,0.0\n",
        encoding="utf-8",
    )

    selected = _load_optional_csv(selected_path)
    gap = _load_optional_csv(gap_path)

    assert selected is not None
    assert gap is not None
    assert selected["sequence_id"].tolist() == ["0001"]
    assert gap["sequence"].tolist() == ["0001"]
    assert gap["sequence_id"].tolist() == ["0001"]

    public_rows = pd.DataFrame(
        {
            "sequence_id": ["0001"],
            "matched": [True],
            "error_3d_m": [1.0],
            "squared_error_3d_m2": [1.0],
        }
    )
    pose = build_pose_by_sequence_table(
        public_rows,
        selected_tracklets=selected,
        candidate_oracle_gap=gap,
    )

    assert pose.loc[0, "sequence"] == "0001"
    assert pose.loc[0, "dominant_sensor"] == "radar"
    assert int(pose.loc[0, "used_radar_count"]) == 1
    assert int(pose.loc[0, "empty_radar_count"]) == 1

    regret = build_candidate_regret_summary(gap)
    assert regret.loc[0, "sequence"] == "0001"


def test_scorecard_optional_csvs_preserve_na_like_sequence_identifiers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "na_like_ids.csv"
    path.write_text(
        "sequence_id,sequence,Sequence,value\n"
        "NA,null,N/A,\n"
        ",,NA,2\n",
        encoding="utf-8",
    )

    loaded = _load_optional_csv(path)

    assert loaded is not None
    assert loaded.loc[0, "sequence_id"] == "NA"
    assert loaded.loc[0, "sequence"] == "null"
    assert loaded.loc[0, "Sequence"] == "N/A"
    assert str(loaded["sequence_id"].dtype).startswith("string")
    assert pd.isna(loaded.loc[0, "value"])
    assert pd.isna(loaded.loc[1, "sequence_id"])
    assert pd.isna(loaded.loc[1, "sequence"])
    assert loaded.loc[1, "Sequence"] == "NA"


@pytest.mark.parametrize("prefix", ["", "\n", "\n   \n"])
def test_scorecard_optional_csv_rejects_duplicate_sequence_header(
    tmp_path: Path,
    prefix: str,
) -> None:
    path = tmp_path / "duplicate_sequence.csv"
    path.write_text(
        prefix
        + "sequence_id,sequence_id,time_s,source\n"
        + "0001,9999,0.0,radar-enhance-pcl\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate physical CSV columns.*sequence_id"):
        _load_optional_csv(path)


def test_scorecard_optional_csv_rejects_duplicate_non_identifier_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate_sensor.csv"
    path.write_text(
        "sequence_id,sensor,sensor\n"
        "0001,radar,lidar_360\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate physical CSV columns.*sensor"):
        _load_optional_csv(path)
