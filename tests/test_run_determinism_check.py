from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from scripts import run_determinism_check


@pytest.mark.parametrize("bad_atol", ["nan", "inf", "-inf", "-1.0"])
def test_determinism_check_rejects_invalid_atol(monkeypatch, tmp_path, capsys, bad_atol):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_determinism_check.py",
            str(tmp_path / "run_a"),
            str(tmp_path / "run_b"),
            "--atol",
            bad_atol,
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_determinism_check.main()

    assert excinfo.value.code == 2
    assert "--atol must be finite and non-negative" in capsys.readouterr().err


def test_determinism_check_rejects_missing_estimate_coordinate(monkeypatch, tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [1.0, 2.0],
            "north_m": [3.0, 4.0],
            "up_m": [5.0, 6.0],
        }
    )
    estimates.to_csv(run_a / "estimates.csv", index=False)
    estimates.drop(columns="up_m").to_csv(run_b / "estimates.csv", index=False)
    output_json = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_determinism_check.py",
            str(run_a),
            str(run_b),
            "--output-json",
            str(output_json),
            "--fail-on-difference",
        ],
    )

    assert run_determinism_check.main() == 1
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["estimate_schema_equal"] is False
    assert summary["estimate_missing_columns_a"] == []
    assert summary["estimate_missing_columns_b"] == ["up_m"]
    assert summary["estimates_nearly_equal"] is False
