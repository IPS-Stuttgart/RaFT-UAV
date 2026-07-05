from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.evaluator import validate_mmaud_results_frame
from raft_uav.mmuad.submission import parse_official_classification_cell


def _official_track5_frame(classification: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [1.0],
            "Position": ["(1,2,3)"],
            "Classification": [classification],
        }
    )


def test_track5_public_parser_rejects_bad_class_id():
    with pytest.raises(ValueError):
        parse_official_classification_cell(4)


def test_track5_result_validation_rejects_bad_class_id():
    with pytest.raises(ValueError, match="Classification values must be one of"):
        validate_mmaud_results_frame(_official_track5_frame(4))


def test_track5_truth_loader_keeps_legacy_class_ids_permissive(tmp_path):
    truth_zip = tmp_path / "truth.zip"
    with ZipFile(truth_zip, "w") as archive:
        archive.writestr("mmaud_results.csv", _official_track5_frame(4).to_csv(index=False))

    truth = load_evaluation_truth_file(truth_zip)

    assert truth.rows.loc[0, "class_name"] == "4"
