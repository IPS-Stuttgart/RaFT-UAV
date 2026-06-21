from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.cli import main as mmuad_cli_main
from raft_uav.mmuad.track5_scorecard_cli import main as track5_scorecard_cli_main
from raft_uav.mmuad.track5_scorecard import (
    build_track5_scorecard,
    scorecard_summary_frame,
    write_track5_scorecard,
)


def _write_official_zip(path: Path, frame: pd.DataFrame) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))
    return path


def test_track5_scorecard_reports_ready_zero_error_submission(tmp_path: Path) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001", "seq001"],
                "Timestamp": [0.0, 1.0],
                "Position": ["(0,0,10)", "(1,0,10)"],
                "Classification": [1, 1],
            }
        ),
    )
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "class_name": [1, 1],
        }
    ).to_csv(truth_csv, index=False)

    scorecard = build_track5_scorecard(results_path=results_zip, truth_path=truth_csv)

    assert scorecard.summary["validation"]["codabench_upload_ready"] is True
    assert scorecard.summary["scorecard_leaderboard_ready"] is True
    pooled = scorecard.summary["public_track5"]["pooled"]
    assert pooled["pose_mse_loss_m2"] == 0.0
    assert pooled["uav_type_accuracy"] == 1.0
    assert pooled["classification_accuracy"] == 1.0
    flat = scorecard_summary_frame(scorecard.summary)
    assert flat.loc[0, "pose_mse_loss_m2"] == 0.0
    assert flat.loc[0, "uav_type_accuracy"] == 1.0
    assert flat.loc[0, "classification_accuracy"] == 1.0


def test_track5_scorecard_writes_all_requested_artifacts(tmp_path: Path) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001"],
                "Timestamp": [0.0],
                "Position": ["(0,0,10)"],
                "Classification": [2],
            }
        ),
    )
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [10.0],
            "class_name": [2],
        }
    ).to_csv(truth_csv, index=False)
    scorecard = build_track5_scorecard(results_path=results_zip, truth_path=truth_csv)

    paths = write_track5_scorecard(
        scorecard,
        summary_json=tmp_path / "scorecard.json",
        summary_csv=tmp_path / "scorecard.csv",
        validation_rows_csv=tmp_path / "validation_rows.csv",
        public_evaluation_rows_csv=tmp_path / "public_rows.csv",
        nearest_time_rows_csv=tmp_path / "nearest_rows.csv",
    )

    assert set(paths) == {
        "scorecard_json",
        "scorecard_csv",
        "validation_rows_csv",
        "public_evaluation_rows_csv",
        "nearest_time_rows_csv",
    }
    for path in paths.values():
        assert Path(path).exists()


def test_track5_scorecard_writes_paper_pose_and_candidate_regret_tables(
    tmp_path: Path,
) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001", "seq001", "seq002"],
                "Timestamp": [0.0, 1.0, 0.0],
                "Position": ["(0,0,10)", "(4,0,10)", "(0,3,0)"],
                "Classification": [1, 1, 2],
            }
        ),
    )
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq002"],
            "time_s": [0.0, 1.0, 0.0],
            "x_m": [0.0, 1.0, 0.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [10.0, 10.0, 0.0],
            "class_name": [1, 1, 2],
        }
    ).to_csv(truth_csv, index=False)
    selected_csv = tmp_path / "mmuad_selected_tracklets.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq002"],
            "time_s": [0.0, 1.0, 0.0],
            "source": ["lidar_360", "radar-enhance-pcl", "livox_avia"],
            "track_id": ["a", "b", "c"],
            "x_m": [0.0, 4.0, 0.0],
            "y_m": [0.0, 0.0, 3.0],
            "z_m": [10.0, 10.0, 0.0],
        }
    ).to_csv(selected_csv, index=False)
    gap_csv = tmp_path / "mmuad_candidate_oracle_gap.csv"
    pd.DataFrame(
        {
            "sequence": ["seq001", "seq001", "seq002"],
            "sequence_id": ["seq001", "seq001", "seq002"],
            "time_s": [0.0, 1.0, 0.0],
            "sensor": ["lidar_360", "radar_enhance_pcl", "livox_avia"],
            "nearest_candidate_found": [True, False, True],
            "selected_candidate_found": [True, True, True],
            "selected_source_matches_sensor": [True, True, True],
            "candidate_count_at_nearest_time": [2, 0, 1],
            "selected_minus_truth_error_m": [0.0, 3.0, 3.0],
            "nearest_minus_truth_error_m": [0.0, None, 1.0],
            "candidate_regret_m": [0.0, None, 2.0],
            "nearest_candidate_time_delta_s": [0.0, None, 0.1],
        }
    ).to_csv(gap_csv, index=False)

    scorecard = build_track5_scorecard(
        results_path=results_zip,
        truth_path=truth_csv,
        selected_tracklets_path=selected_csv,
        candidate_oracle_gap_path=gap_csv,
    )
    paths = write_track5_scorecard(
        scorecard,
        summary_json=tmp_path / "scorecard.json",
        summary_csv=tmp_path / "scorecard.csv",
        pose_by_sequence_csv=tmp_path / "mmuad_pose_by_sequence.csv",
        candidate_regret_summary_csv=tmp_path / "mmuad_candidate_regret_summary.csv",
    )

    assert "pose_by_sequence_csv" in paths
    assert "candidate_regret_summary_csv" in paths
    pose = pd.read_csv(paths["pose_by_sequence_csv"])
    seq001 = pose.loc[pose["sequence"] == "seq001"].iloc[0]
    assert int(seq001["count"]) == 2
    assert float(seq001["mse"]) == 4.5
    assert float(seq001["rmse"]) == pytest.approx(4.5**0.5)
    assert seq001["dominant_sensor"] in {"lidar_360", "radar"}
    assert int(seq001["used_lidar_360_count"]) == 1
    assert int(seq001["used_radar_count"]) == 1
    assert int(seq001["empty_radar_count"]) == 1
    regret = pd.read_csv(paths["candidate_regret_summary_csv"])
    livox = regret.loc[
        (regret["sequence"] == "seq002") & (regret["sensor"] == "livox_avia")
    ].iloc[0]
    assert float(livox["mean_candidate_regret_m"]) == 2.0
    assert float(livox["positive_regret_fraction"]) == 1.0
    summary = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert summary["selected_tracklets_path"] == str(selected_csv)
    assert summary["candidate_oracle_gap_path"] == str(gap_csv)
    assert summary["paper_artifacts"]["pose_by_sequence_rows"] == 2
    assert summary["paper_artifacts"]["candidate_regret_summary_rows"] == 3
    flat = pd.read_csv(tmp_path / "scorecard.csv")
    assert int(flat.loc[0, "paper_pose_by_sequence_rows"]) == 2
    assert int(flat.loc[0, "paper_candidate_regret_summary_rows"]) == 3


def test_track5_scorecard_reports_sequence_classifier_provenance(tmp_path: Path) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001"],
                "Timestamp": [0.0],
                "Position": ["(0,0,10)"],
                "Classification": [3],
            }
        ),
    )
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [10.0],
            "class_name": [3],
        }
    ).to_csv(truth_csv, index=False)
    provenance_json = tmp_path / "classifier_provenance.json"
    provenance_json.write_text(
        json.dumps(
            {
                "classification_model_path": "outputs/mmuad_sequence_classifier_rf.joblib",
                "classification_method": "random-forest",
                "classification_train_sequences": ["seq_train_0", "seq_train_3"],
                "classification_feature_columns": ["row_count", "z_m_mean"],
                "classification_class_map": {"seq001": "3"},
                "classification_prediction_mode": "sequence_level",
            }
        ),
        encoding="utf-8",
    )

    scorecard = build_track5_scorecard(
        results_path=results_zip,
        truth_path=truth_csv,
        classification_provenance_path=provenance_json,
    )

    assert scorecard.summary["classification_method"] == "random-forest"
    assert scorecard.summary["classification_prediction_mode"] == "sequence_level"
    assert scorecard.summary["classification_class_map"] == {"seq001": "3"}
    flat = scorecard_summary_frame(scorecard.summary)
    assert flat.loc[0, "classification_model_path"].endswith(
        "mmuad_sequence_classifier_rf.joblib"
    )
    assert flat.loc[0, "classification_train_sequences"] == "seq_train_0;seq_train_3"
    assert json.loads(flat.loc[0, "classification_class_map"]) == {"seq001": "3"}


def test_track5_scorecard_cli_writes_ready_artifacts(tmp_path: Path, capsys) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001", "seq001"],
                "Timestamp": [0.0, 1.0],
                "Position": ["(0,0,10)", "(1,0,10)"],
                "Classification": [2, 2],
            }
        ),
    )
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "class_name": [2, 2],
        }
    ).to_csv(truth_csv, index=False)
    selected_csv = tmp_path / "selected.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "source": ["lidar_360", "lidar_360"],
            "track_id": ["a", "a"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
        }
    ).to_csv(selected_csv, index=False)
    gap_csv = tmp_path / "candidate_gap.csv"
    pd.DataFrame(
        {
            "sequence": ["seq001"],
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "sensor": ["lidar_360"],
            "nearest_candidate_found": [True],
            "selected_candidate_found": [True],
            "selected_source_matches_sensor": [True],
            "candidate_count_at_nearest_time": [1],
            "selected_minus_truth_error_m": [0.0],
            "nearest_minus_truth_error_m": [0.0],
            "candidate_regret_m": [0.0],
            "nearest_candidate_time_delta_s": [0.0],
        }
    ).to_csv(gap_csv, index=False)
    output = tmp_path / "scorecard"

    status = track5_scorecard_cli_main(
        [
            "--results",
            str(results_zip),
            "--truth",
            str(truth_csv),
            "--output-json",
            str(output / "scorecard.json"),
            "--summary-csv",
            str(output / "scorecard.csv"),
            "--validation-rows-csv",
            str(output / "validation_rows.csv"),
            "--public-evaluation-rows-csv",
            str(output / "public_rows.csv"),
            "--nearest-time-rows-csv",
            str(output / "nearest_rows.csv"),
            "--selected-tracklets-csv",
            str(selected_csv),
            "--candidate-oracle-gap-csv",
            str(gap_csv),
            "--pose-by-sequence-csv",
            str(output / "mmuad_pose_by_sequence.csv"),
            "--candidate-regret-summary-csv",
            str(output / "mmuad_candidate_regret_summary.csv"),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    stdout = capsys.readouterr().out
    assert "track5_scorecard=ok" in stdout
    assert "leaderboard_ready=True" in stdout
    assert "uav_type_accuracy=1.0" in stdout
    assert "classification_accuracy=1.0" in stdout
    summary = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "raft-uav-mmuad-track5-scorecard-v1"
    assert summary["scorecard_leaderboard_ready"] is True
    assert summary["codabench_upload_ready"] is True
    assert summary["public_track5"]["pooled"]["pose_mse_loss_m2"] == 0.0
    flat = pd.read_csv(output / "scorecard.csv")
    assert flat.loc[0, "classification_accuracy"] == 1.0
    assert (output / "scorecard.csv").exists()
    assert (output / "validation_rows.csv").exists()
    assert (output / "public_rows.csv").exists()
    assert (output / "nearest_rows.csv").exists()
    assert (output / "mmuad_pose_by_sequence.csv").exists()
    assert (output / "mmuad_candidate_regret_summary.csv").exists()


def test_track5_scorecard_cli_verifies_official_upload_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001", "seq001"],
                "Timestamp": [0.0, 1.0],
                "Position": ["(0,0,10)", "(1,0,10)"],
                "Classification": [2, 2],
            }
        ),
    )
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "class_name": [2, 2],
        }
    ).to_csv(truth_csv, index=False)
    manifest_path = tmp_path / "official_upload_manifest.json"
    validation_output = tmp_path / "validation"
    assert (
        mmuad_cli_main(
            [
                "--validate-ug2-official-codabench-zip",
                str(results_zip),
                "--official-validation-template-file",
                str(truth_csv),
                "--official-upload-manifest-json",
                str(manifest_path),
                "--output-dir",
                str(validation_output),
            ]
        )
        == 0
    )
    output = tmp_path / "scorecard"

    status = track5_scorecard_cli_main(
        [
            "--results",
            str(results_zip),
            "--truth",
            str(truth_csv),
            "--official-upload-manifest",
            str(manifest_path),
            "--output-json",
            str(output / "scorecard.json"),
            "--summary-csv",
            str(output / "scorecard.csv"),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    stdout = capsys.readouterr().out
    assert "upload_manifest_valid=True" in stdout
    summary = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
    assert summary["scorecard_leaderboard_ready"] is True
    assert summary["upload_manifest_valid"] is True
    assert summary["upload_manifest_codabench_upload_ready"] is True
    assert summary["upload_manifest_verification"]["artifact_sha256_matches"] is True
    flat = pd.read_csv(output / "scorecard.csv")
    assert bool(flat.loc[0, "upload_manifest_valid"]) is True


def test_track5_scorecard_cli_blocks_tampered_manifest_artifact(
    tmp_path: Path,
) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001", "seq001"],
                "Timestamp": [0.0, 1.0],
                "Position": ["(0,0,10)", "(1,0,10)"],
                "Classification": [2, 2],
            }
        ),
    )
    truth_csv = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [10.0, 10.0],
            "class_name": [2, 2],
        }
    ).to_csv(truth_csv, index=False)
    manifest_path = tmp_path / "official_upload_manifest.json"
    validation_output = tmp_path / "validation"
    assert (
        mmuad_cli_main(
            [
                "--validate-ug2-official-codabench-zip",
                str(results_zip),
                "--official-validation-template-file",
                str(truth_csv),
                "--official-upload-manifest-json",
                str(manifest_path),
                "--output-dir",
                str(validation_output),
            ]
        )
        == 0
    )
    _write_official_zip(
        results_zip,
        pd.DataFrame(
            {
                "Sequence": ["seq001", "seq001"],
                "Timestamp": [0.0, 1.0],
                "Position": ["(0,0,10)", "(999,0,10)"],
                "Classification": [2, 2],
            }
        ),
    )
    output = tmp_path / "scorecard"

    with pytest.raises(SystemExit, match="official_upload_manifest_invalid"):
        track5_scorecard_cli_main(
            [
                "--results",
                str(results_zip),
                "--truth",
                str(truth_csv),
                "--official-upload-manifest",
                str(manifest_path),
                "--output-json",
                str(output / "scorecard.json"),
                "--require-leaderboard-ready",
            ]
        )

    summary = json.loads((output / "scorecard.json").read_text(encoding="utf-8"))
    assert summary["validation"]["codabench_upload_ready"] is True
    assert summary["public_track5"]["leaderboard_ready"] is True
    assert summary["scorecard_leaderboard_ready"] is False
    assert summary["upload_manifest_valid"] is False
    assert "official_upload_manifest_invalid" in summary["leaderboard_blocking_reasons"]
    assert (
        summary["upload_manifest_verification"]["mmaud_results_csv_sha256_matches"]
        is False
    )


def test_track5_scorecard_cli_requires_truth_for_leaderboard_ready(
    tmp_path: Path,
) -> None:
    results_zip = _write_official_zip(
        tmp_path / "submission.zip",
        pd.DataFrame(
            {
                "Sequence": ["seq001"],
                "Timestamp": [0.0],
                "Position": ["(0,0,10)"],
                "Classification": [2],
            }
        ),
    )
    template_csv = tmp_path / "template.csv"
    pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    ).to_csv(template_csv, index=False)
    output_json = tmp_path / "scorecard.json"

    with pytest.raises(SystemExit, match="public_track5_evaluation_not_run"):
        track5_scorecard_cli_main(
            [
                "--results",
                str(results_zip),
                "--template",
                str(template_csv),
                "--output-json",
                str(output_json),
                "--require-leaderboard-ready",
            ]
        )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["validation"]["codabench_upload_ready"] is True
    assert summary["scorecard_leaderboard_ready"] is False
    assert summary["leaderboard_blocking_reasons"] == [
        "public_track5_evaluation_not_run"
    ]
