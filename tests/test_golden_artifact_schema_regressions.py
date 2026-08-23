from __future__ import annotations

import json

import pandas as pd
import pytest

from raft_uav.evaluation.golden_artifacts import _check_csv, _check_metrics


@pytest.mark.parametrize(
    "payload",
    [
        ["posterior_records", "accepted_measurements", "position_error_3d"],
        42,
        None,
    ],
)
def test_metrics_check_rejects_non_object_json(tmp_path, payload) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    results = _check_metrics(path)

    assert results[0]["check"] == "metrics_json_parse"
    assert results[0]["passed"] is True
    assert results[1] == {
        "check": "metrics_json_object",
        "file": str(path),
        "passed": False,
        "message": "metrics JSON root must be an object",
    }
    assert all(row["check"] != "metrics_required_key" for row in results)


def test_csv_check_rejects_nonnumeric_time_column(tmp_path) -> None:
    path = tmp_path / "estimates.csv"
    pd.DataFrame(
        {
            "time_s": ["not-a-time", "still-not-a-time"],
            "east_m": [1.0, 2.0],
        }
    ).to_csv(path, index=False)

    results = {row["check"]: row for row in _check_csv(path, max_nan_fraction=0.0)}

    assert results["time_nonfinite_fraction"]["passed"] is False
    assert results["time_nonfinite_fraction"]["value"] == 1.0


def test_csv_time_validation_respects_configured_fraction(tmp_path) -> None:
    path = tmp_path / "diagnostics.csv"
    pd.DataFrame(
        {
            "time_s": ["0.0", "bad"],
            "nis": [1.0, 2.0],
        }
    ).to_csv(path, index=False)

    strict = {row["check"]: row for row in _check_csv(path, max_nan_fraction=0.49)}
    tolerant = {row["check"]: row for row in _check_csv(path, max_nan_fraction=0.5)}

    assert strict["time_nonfinite_fraction"]["passed"] is False
    assert tolerant["time_nonfinite_fraction"]["passed"] is True
    assert tolerant["time_monotonic"]["passed"] is True
