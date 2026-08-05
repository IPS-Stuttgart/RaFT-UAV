from __future__ import annotations

import json

import pytest

from raft_uav.evaluation.golden_artifacts import _check_metrics


@pytest.mark.parametrize(
    "payload",
    [
        "posterior_records accepted_measurements position_error_3d",
        ["posterior_records", "accepted_measurements", "position_error_3d"],
        None,
    ],
)
def test_check_metrics_rejects_non_object_json_roots(tmp_path, payload) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows = _check_metrics(path)
    by_check = {row["check"]: row for row in rows}

    assert by_check["metrics_json_parse"]["passed"] is True
    assert by_check["metrics_json_object"]["passed"] is False
    assert by_check["metrics_json_object"]["message"] == "metrics JSON root must be an object"
    assert not any(row["check"] == "metrics_required_key" for row in rows)


def test_check_metrics_accepts_object_root_and_checks_required_keys(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "posterior_records": 1,
                "accepted_measurements": 1,
                "position_error_3d": {"mean_m": 2.0},
            }
        ),
        encoding="utf-8",
    )

    rows = _check_metrics(path)

    assert next(row for row in rows if row["check"] == "metrics_json_object")["passed"] is True
    required = [row for row in rows if row["check"] == "metrics_required_key"]
    assert len(required) == 3
    assert all(row["passed"] is True for row in required)
