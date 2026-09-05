"""Regression tests for persisted diagnostics and output API preflight checks."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad import track5_speed_limit as speed_limit


@pytest.fixture(autouse=True)
def _mock_official_writers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Archive serialization has integration coverage in test_mmuad_track5_speed_limit.
    # These tests exercise preflight, actual diagnostics CSVs, and manifest JSONs.
    monkeypatch.setattr(speed_limit, "write_official_mmaud_results_csv", Mock())
    monkeypatch.setattr(speed_limit, "write_official_ug2_codabench_zip", Mock())


def _limited_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq-a"] * 4,
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "state_x_m": [0.0, 10.0, 20.0, 30.0],
            "state_y_m": [0.0] * 4,
            "state_z_m": [0.0] * 4,
            "Classification": [1] * 4,
        }
    )


def _diagnostics(flags: pd.Series | list[object]) -> pd.DataFrame:
    rows = pd.DataFrame({"speed_limit_applied": flags})
    rows["speed_limit_correction_m"] = 0.0
    return rows


def _write(tmp_path: Path, diagnostics: pd.DataFrame, **kwargs: object) -> dict[str, Path]:
    return speed_limit.write_track5_speed_limit_outputs(
        limited=_limited_rows(),
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        input_submission_path=tmp_path / "input.csv",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([False, True, False, True], [False, True, False, True]),
        (["False", "True", "0", "1"], [False, True, False, True]),
        ([" FALSE ", " tRuE ", "off", "ON"], [False, True, False, True]),
        (["no", "YES", "n", "y"], [False, True, False, True]),
        (["f", "T", "0.0", "1.0"], [False, True, False, True]),
        ([0, 1, 0.0, 1.0], [False, True, False, True]),
        (["-0.0", "1e0", False, np.bool_(True)], [False, True, False, True]),
        ([None, np.nan, pd.NA, pd.NaT], [False, False, False, False]),
        (["", "null", "None", "NaN"], [False, False, False, False]),
        (["NA", "N/A", "<NA>", "NaT"], [False, False, False, False]),
        (
            pd.Series([False, True, pd.NA, False], dtype="boolean"),
            [False, True, False, False],
        ),
        (
            pd.Series(["False", "True", pd.NA, "0"], dtype="string"),
            [False, True, False, False],
        ),
        (
            pd.Series([0, 1, pd.NA, 0], dtype="Int64"),
            [False, True, False, False],
        ),
    ],
)
def test_persisted_flags_have_exact_manifest_counts(
    tmp_path: Path, flags: pd.Series | list[object], expected: list[bool]
) -> None:
    diagnostics = _diagnostics(flags)
    # Non-default and duplicate labels must not introduce index alignment errors.
    diagnostics.index = [8, 3, 8, 1]
    before = diagnostics.copy(deep=True)

    paths = _write(tmp_path, diagnostics)

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["changed_row_count"] == sum(expected)
    assert manifest["changed_fraction"] == pytest.approx(sum(expected) / len(expected))
    persisted = pd.read_csv(paths["diagnostics_csv"])
    assert persisted["speed_limit_applied"].tolist() == expected
    pd.testing.assert_frame_equal(diagnostics, before)
    pd.testing.assert_frame_equal(pd.read_csv(paths["estimates_csv"]), _limited_rows())


def test_string_typed_csv_round_trip_does_not_count_false_as_true(tmp_path: Path) -> None:
    original = _diagnostics([False, True, False, True])
    diagnostics = pd.read_csv(StringIO(original.to_csv(index=False)), dtype=str)

    paths = _write(tmp_path, diagnostics)

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["changed_row_count"] == 2
    assert manifest["changed_fraction"] == 0.5


@pytest.mark.parametrize(
    "value",
    ["maybe", "2", "-1", "inf", np.inf, -np.inf, 2, -1, 0.5, [1], {"flag": True}, 1 + 0j],
)
def test_invalid_flags_fail_before_output_creation(tmp_path: Path, value: object) -> None:
    diagnostics = _diagnostics(pd.Series([False, value, True, False], dtype=object))

    with pytest.raises(ValueError, match="speed_limit_applied contains invalid Boolean"):
        _write(tmp_path, diagnostics)

    assert not (tmp_path / "out").exists()
    speed_limit.write_official_mmaud_results_csv.assert_not_called()
    speed_limit.write_official_ug2_codabench_zip.assert_not_called()


@pytest.mark.parametrize("failure", ["invalid-flag", "missing-template"])
def test_preflight_failure_does_not_overwrite_existing_outputs(tmp_path: Path, failure: str) -> None:
    output = tmp_path / "out"
    output.mkdir()
    filenames = [
        speed_limit.SPEED_LIMIT_ESTIMATES_CSV,
        speed_limit.SPEED_LIMIT_RESULTS_CSV,
        speed_limit.SPEED_LIMIT_ZIP,
        speed_limit.SPEED_LIMIT_DIAGNOSTICS_CSV,
        speed_limit.SPEED_LIMIT_MANIFEST_JSON,
    ]
    for filename in filenames:
        (output / filename).write_bytes(b"previous successful output\n")
    diagnostics = _diagnostics([False, True, False, True])
    kwargs = {}
    if failure == "invalid-flag":
        diagnostics["speed_limit_applied"] = ["False", "invalid", "False", "True"]
    else:
        kwargs["require_leaderboard_ready"] = True

    with pytest.raises(ValueError):
        _write(tmp_path, diagnostics, **kwargs)

    for filename in filenames:
        assert (output / filename).read_bytes() == b"previous successful output\n"
    speed_limit.write_official_mmaud_results_csv.assert_not_called()
    speed_limit.write_official_ug2_codabench_zip.assert_not_called()


@pytest.mark.parametrize(
    "diagnostics",
    [pd.DataFrame(), pd.DataFrame({"speed_limit_correction_m": [0.0] * 4})],
)
def test_absent_optional_flags_keep_zero_summary(tmp_path: Path, diagnostics: pd.DataFrame) -> None:
    paths = _write(tmp_path, diagnostics)

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["changed_row_count"] == 0
    assert manifest["changed_fraction"] == 0.0
    assert manifest["validation"] is None


def test_output_api_requires_template_before_any_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require_leaderboard_ready.*requires.*template"):
        _write(
            tmp_path,
            _diagnostics([False, True, False, True]),
            require_leaderboard_ready=True,
        )

    assert not (tmp_path / "out").exists()
    speed_limit.write_official_mmaud_results_csv.assert_not_called()
    speed_limit.write_official_ug2_codabench_zip.assert_not_called()


@pytest.mark.parametrize("ready", [True, False])
def test_required_template_validation_is_not_bypassed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ready: bool
) -> None:
    template = pd.DataFrame({"Sequence": ["seq-a"] * 4})
    validation = SimpleNamespace(
        summary={"leaderboard_ready": ready, "leaderboard_blocking_reasons": [] if ready else ["bad grid"]},
        rows=pd.DataFrame({"valid": [ready]}),
    )
    validator = Mock(return_value=validation)
    monkeypatch.setattr(speed_limit, "validate_official_track5_submission", validator)
    diagnostics = _diagnostics([False, True, False, True])

    if ready:
        paths = _write(tmp_path, diagnostics, template=template, require_leaderboard_ready=True)
        manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
        assert manifest["validation"]["leaderboard_ready"] is True
    else:
        with pytest.raises(SystemExit, match="not leaderboard-ready: bad grid"):
            _write(tmp_path, diagnostics, template=template, require_leaderboard_ready=True)

    validator.assert_called_once()
    assert validator.call_args.kwargs["template"] is template
    assert validator.call_args.kwargs["require_zip"] is True
