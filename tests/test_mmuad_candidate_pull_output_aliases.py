from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from raft_uav.mmuad.candidate_pull import write_candidate_pull_artifacts


@pytest.mark.parametrize(
    "alias_name",
    [
        "submission_zip",
        "provenance_json",
        "centers_csv",
        "sequence_features_csv",
        "alpha_assignments_csv",
    ],
)
def test_candidate_pull_rejects_result_output_alias_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_name: str,
) -> None:
    shared_path = tmp_path / "shared-output"
    zip_write_reached = False

    def fail_if_zip_write_is_reached(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal zip_write_reached
        zip_write_reached = True
        raise AssertionError("ZIP self-copy boundary was reached")

    monkeypatch.setattr(zipfile.ZipFile, "write", fail_if_zip_write_is_reached)
    outputs = {alias_name: shared_path}

    with pytest.raises(
        ValueError,
        match=rf"results_csv and {alias_name}",
    ):
        write_candidate_pull_artifacts(
            object(),
            results_csv=shared_path,
            **outputs,
        )

    assert not shared_path.exists()
    assert not zip_write_reached


def test_candidate_pull_rejects_normalized_optional_output_aliases(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    provenance_json = tmp_path / "shared.json"
    centers_csv = tmp_path / "nested" / ".." / "shared.json"

    with pytest.raises(
        ValueError,
        match=r"provenance_json and centers_csv",
    ):
        write_candidate_pull_artifacts(
            object(),
            results_csv=results_csv,
            provenance_json=provenance_json,
            centers_csv=centers_csv,
        )

    assert not results_csv.exists()
    assert not provenance_json.exists()
    assert not (tmp_path / "nested").exists()
