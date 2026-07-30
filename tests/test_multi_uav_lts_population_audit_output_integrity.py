from __future__ import annotations

import json
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.population_audit import (
    audit_first_frame_population,
    write_population_audit,
)


def _audit_with_truth(tmp_path: Path):
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    truth_path = truth_dir / "S.txt"
    truth_text = "1,1,0,0,10,10,1,1,1\n"
    truth_path.write_text(truth_text, encoding="utf-8")
    return audit_first_frame_population(truth_dir), truth_path, truth_text


def test_population_audit_rejects_aliased_outputs_without_modifying_existing_file(
    tmp_path: Path,
) -> None:
    audit, _, _ = _audit_with_truth(tmp_path)
    output = tmp_path / "audit.out"
    output.write_text("original\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="population audit JSON and CSV outputs must differ",
    ):
        write_population_audit(audit, output, output)

    assert output.read_text(encoding="utf-8") == "original\n"


@pytest.mark.parametrize(
    ("aliased_output", "message"),
    [
        ("json", "population audit JSON output must differ from truth input file"),
        ("csv", "population audit CSV output must differ from truth input file"),
    ],
)
def test_population_audit_rejects_truth_output_alias_without_partial_outputs(
    tmp_path: Path,
    aliased_output: str,
    message: str,
) -> None:
    audit, truth_path, truth_text = _audit_with_truth(tmp_path)
    json_path = truth_path if aliased_output == "json" else tmp_path / "audit.json"
    csv_path = truth_path if aliased_output == "csv" else tmp_path / "audit.csv"

    with pytest.raises(ValueError, match=message):
        write_population_audit(audit, json_path, csv_path)

    assert truth_path.read_text(encoding="utf-8") == truth_text
    if json_path != truth_path:
        assert not json_path.exists()
    if csv_path != truth_path:
        assert not csv_path.exists()


def test_population_audit_writes_distinct_outputs(tmp_path: Path) -> None:
    audit, _, _ = _audit_with_truth(tmp_path)
    json_path = tmp_path / "outputs" / "audit.json"
    csv_path = tmp_path / "outputs" / "audit.csv"

    write_population_audit(audit, json_path, csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["sequence_count"] == 1
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("sequence,")
