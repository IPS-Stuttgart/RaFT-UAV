from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest

from raft_uav.mmuad.track5_estimate_ensemble import apply_estimate_weight_config
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble_grid_config import build_weight_config_from_grid_manifest
from raft_uav.mmuad.track5_estimate_ensemble_grid_config import main as config_main
from raft_uav.mmuad.track5_estimate_ensemble_grid_config import write_weight_config_from_grid_manifest


def _manifest() -> dict[str, object]:
    return {
        "schema": "raft-uav-mmuad-track5-estimate-ensemble-weight-grid-v2",
        "estimate_inputs": [
            {"label": "robust", "path": "robust.csv"},
            {"label": "reservoir", "path": "reservoir.csv"},
        ],
        "grid_row_count": 9,
        "best_weights": [0.75, 0.25],
        "best_aggregation_policy": "weighted-median",
        "best_trim_fraction": 0.1,
        "best": {"pose_mse": 12.5, "p95_error_m": 7.0},
    }


def test_build_weight_config_from_grid_manifest_is_ensemble_compatible() -> None:
    config = build_weight_config_from_grid_manifest(_manifest(), source_manifest="grid.json")

    assert config["weights"] == {"robust": 0.75, "reservoir": 0.25}
    assert config["aggregation_policy"] == "weighted-median"
    assert config["trim_fraction"] == pytest.approx(0.1)
    assert config["metrics"]["pose_mse"] == pytest.approx(12.5)

    updated = apply_estimate_weight_config(
        [
            EstimateInput("robust", Path("robust.csv"), 1.0),
            EstimateInput("reservoir", Path("reservoir.csv"), 1.0),
        ],
        config["weights"],
    )
    assert [item.weight for item in updated] == [pytest.approx(0.75), pytest.approx(0.25)]


def test_build_weight_config_rejects_mismatched_lengths() -> None:
    manifest = _manifest()
    manifest["best_weights"] = [1.0]
    with pytest.raises(ValueError, match="lengths differ"):
        build_weight_config_from_grid_manifest(manifest)


def test_write_weight_config_from_grid_manifest(tmp_path: Path) -> None:
    manifest_json = tmp_path / "manifest.json"
    output_json = tmp_path / "weights.json"
    manifest_json.write_text(json.dumps(_manifest()), encoding="utf-8")

    config = write_weight_config_from_grid_manifest(
        manifest_json=manifest_json,
        output_json=output_json,
    )

    assert output_json.exists()
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded == config
    assert loaded["weights"]["robust"] == pytest.approx(0.75)


def test_grid_config_cli_writes_default_output(tmp_path: Path) -> None:
    manifest_json = tmp_path / "mmuad_track5_estimate_ensemble_weight_grid_manifest.json"
    manifest_json.write_text(json.dumps(_manifest()), encoding="utf-8")

    status = config_main(["--grid-manifest-json", str(manifest_json)])

    assert status == 0
    output_json = tmp_path / "mmuad_track5_estimate_ensemble_best_weight_config.json"
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["weights"] == {"robust": 0.75, "reservoir": 0.25}


def test_grid_config_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-ensemble-grid-config"]
        == "raft_uav.mmuad.track5_estimate_ensemble_grid_config:main"
    )
