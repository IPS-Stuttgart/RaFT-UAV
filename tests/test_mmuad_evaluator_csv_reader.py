from __future__ import annotations

from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
    load_mmaud_results_csv,
    load_mmaud_results_file,
)


def test_mmaud_evaluator_loader_strips_normalized_csv_headers(tmp_path) -> None:
    csv_path = tmp_path / "mmaud_results.csv"
    csv_path.write_text(
        " sequence_id , timestamp , x , y , z , uav_type , score \n"
        "001,0.0,1.0,2.0,3.0,2,1.0\n",
        encoding="utf-8",
    )

    rows = load_mmaud_results_csv(csv_path).rows

    assert rows["sequence_id"].tolist() == ["001"]
    assert rows.columns.tolist() == [
        "sequence_id",
        "timestamp",
        "x",
        "y",
        "z",
        "uav_type",
        "score",
    ]
    assert float(rows.loc[0, "x"]) == pytest.approx(1.0)


def test_mmaud_evaluator_zip_preserves_zero_padded_official_sequence_ids(tmp_path) -> None:
    zip_path = tmp_path / "ug2_submission.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "mmaud_results.csv",
            "Sequence,Timestamp,Position,Classification\n"
            '001,0.0,"(1.0,2.0,3.0)",2\n',
        )

    results = load_mmaud_results_file(zip_path)
    truth = pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(1.0,2.0,4.0)"],
            "Classification": [2],
        }
    )

    evaluation = evaluate_mmaud_results(
        results,
        truth,
        metric_protocol="public-track5",
    )

    assert results.rows["sequence_id"].tolist() == ["001"]
    assert evaluation["summary"]["matched_count"] == 1
    assert evaluation["summary"]["missing_prediction_count"] == 0
    assert evaluation["summary"]["extra_prediction_count"] == 0
    assert evaluation["summary"]["pooled"]["pose_mse_loss_m2"] == pytest.approx(1.0)
