import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "aggregate_stateful_lofo.py"


def _write_summary(root: Path, artifact_name: str, payload: dict) -> None:
    artifact = root / artifact_name
    artifact.mkdir(parents=True)
    (artifact / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _run_aggregate(
    tmp_path: Path,
    expected_flights: list[str],
    *,
    enforce_thresholds: bool = True,
    target_mean_rmse_3d_m: float = 10.0,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "aggregate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifacts-dir",
            str(artifacts),
            "--output",
            str(output),
            "--expected-flights-json",
            json.dumps(expected_flights),
            "--smoke-mode",
            "false",
            "--enforce-thresholds",
            "true" if enforce_thresholds else "false",
            "--target-mean-rmse-3d-m",
            str(target_mean_rmse_3d_m),
            "--target-opt1-p95-3d-m",
            "10.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_threshold_gate_rejects_non_ok_expected_flight(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_summary(
        artifacts,
        "stateful-lofo-Opt1",
        {"flight": "Opt1", "status": "ok", "rmse_3d_m": 1.0, "p95_3d_m": 2.0},
    )
    _write_summary(
        artifacts,
        "stateful-lofo-Opt2",
        {"flight": "Opt2", "status": "missing_metrics", "rmse_3d_m": None, "p95_3d_m": None},
    )

    completed, summary = _run_aggregate(tmp_path, ["Opt1", "Opt2"])

    assert completed.returncode == 1
    assert summary["threshold_failures"] == ["Opt2: summary status is missing_metrics"]


def test_threshold_gate_rejects_duplicate_expected_flight_summary(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    for artifact_name in ("stateful-lofo-Opt1-a", "stateful-lofo-Opt1-b"):
        _write_summary(
            artifacts,
            artifact_name,
            {"flight": "Opt1", "status": "ok", "rmse_3d_m": 1.0, "p95_3d_m": 2.0},
        )
    _write_summary(
        artifacts,
        "stateful-lofo-Opt2",
        {"flight": "Opt2", "status": "ok", "rmse_3d_m": 1.0, "p95_3d_m": 2.0},
    )

    completed, summary = _run_aggregate(tmp_path, ["Opt1", "Opt2"])

    assert completed.returncode == 1
    assert summary["threshold_failures"] == ["Opt1: found 2 summary artifacts; expected exactly one"]


def test_threshold_mean_ignores_unrequested_flight_summaries(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_summary(
        artifacts,
        "stateful-lofo-Opt1",
        {"flight": "Opt1", "status": "ok", "rmse_3d_m": 1.0, "p95_3d_m": 2.0},
    )
    _write_summary(
        artifacts,
        "stateful-lofo-Opt2",
        {"flight": "Opt2", "status": "ok", "rmse_3d_m": 1.0, "p95_3d_m": 2.0},
    )
    _write_summary(
        artifacts,
        "stateful-lofo-Opt3-unrequested",
        {"flight": "Opt3", "status": "ok", "rmse_3d_m": 1000.0, "p95_3d_m": 1000.0},
    )

    completed, summary = _run_aggregate(tmp_path, ["Opt1", "Opt2"], target_mean_rmse_3d_m=2.0)

    assert completed.returncode == 0
    assert summary["threshold_failures"] == []


def test_reporting_only_mode_remains_permissive_for_incomplete_flights(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_summary(
        artifacts,
        "stateful-lofo-Opt1",
        {"flight": "Opt1", "status": "ok", "rmse_3d_m": 1.0, "p95_3d_m": 2.0},
    )
    _write_summary(
        artifacts,
        "stateful-lofo-Opt2",
        {"flight": "Opt2", "status": "missing_metrics", "rmse_3d_m": None, "p95_3d_m": None},
    )

    completed, summary = _run_aggregate(tmp_path, ["Opt1", "Opt2"], enforce_thresholds=False)

    assert completed.returncode == 0
    assert summary["threshold_failures"] == []
