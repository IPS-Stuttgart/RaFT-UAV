import pandas as pd

from raft_uav.mmuad.submission import validate_official_track5_submission


def test_track5_submission_validator_rejects_out_of_domain_class_id(tmp_path):
    csv_path = tmp_path / "mmaud_results.csv"
    pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [4],
        }
    ).to_csv(csv_path, index=False)
    template = pd.DataFrame({"Sequence": ["seq1"], "Timestamp": [0.0]})

    validation = validate_official_track5_submission(
        csv_path,
        template=template,
        require_zip=False,
    )

    assert validation.summary["valid"] is False
    assert validation.summary["leaderboard_ready"] is False
    assert validation.summary["invalid_classification_count"] == 1
    assert "invalid_classification" in validation.rows["status"].tolist()
