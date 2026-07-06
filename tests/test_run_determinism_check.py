from __future__ import annotations

import sys

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
