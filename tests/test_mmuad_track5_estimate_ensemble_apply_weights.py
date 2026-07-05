from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import apply_ensemble_weight_config
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import load_ensemble_weight_config
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import main as apply_weights_main
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import write_apply_weights_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _estimate_a() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [0.0, 1.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _estimate_b() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "state_x_m": [2.0, 3.0, 8.0],
            "state_y_m": [2.0, 2.0, 8.0],
            "state_z_m": [2.0, 2.0, 8.0],
        }
    )


def _weights_payload() -> dict[str, object]:
    return {
        "schema": "raft-uav-mmuad-track5-estimate-ensemble-weight-search-v1",
        "weights": {"a": 0.75, "b": 0.25},
        "aggregation_policy": "weighted-mean",
        "trim_fraction": 0.2,
    }


def test_load_ensemble_weight_config_validates_weights(tmp_path: Path) -> None:
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(_weights_payload()), encoding="utf-8")

    payload = load_ensemble_weight_config(path)

    assert payload["weights"] == {"a": 0.75, "b": 0.25}


def test_load_ensemble_weight_config_rejects_normalized_label_collisions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weights.json"
    payload = _weights_payload()
    payload["weights"] = {"model a": 0.6, "model/a": 0.4}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unique after normalization"):
        load_ensemble_weight_config(path)


def test_apply_ensemble_weight_config_maps_labels_to_estimates(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)

    estimate_inputs = apply_ensemble_weight_config(
        [f"a={a_csv}", f"b={b_csv}@1.0"],
        _weights_payload(),
    )

    assert estimate_inputs == [
        EstimateInput("a", a_csv, 0.75),
        EstimateInput("b", b_csv, 0.25),
    ]


def test_apply_ensemble_weight_config_rejects_missing_label(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    _estimate_a().to_csv(a_csv, index=False)

    with pytest.raises(ValueError, match="missing selected ensemble weight"):
        apply_ensemble_weight_config([f"missing={a_csv}"], _weights_payload())


def test_apply_ensemble_weight_config_rejects_duplicate_normalized_estimate_labels(
    tmp_path: Path,
) -> None:
    a_csv = tmp_path / "a.csv"
    _estimate_a().to_csv(a_csv, index=False)

    with pytest.raises(ValueError, match="duplicate estimate label"):
        apply_ensemble_weight_config(
            [f"model a={a_csv}", f"model/a={a_csv}"],
            {"weights": {"model_a": 1.0}},
        )


def test_write_apply_weights_outputs_manifest_preserves_generator_inputs(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)
    _template().to_csv(template_csv, index=False)
    estimate_inputs = (
        EstimateInput(label, path, weight)
        for label, path, weight in (("a", a_csv, 0.75), ("b", b_csv, 0.25))
    )

    paths = write_apply_weights_outputs(
        estimate_inputs=estimate_inputs,
        weight_config=_weights_payload(),
        template_path=template_csv,
        output_dir=output_dir,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    manifest = json.loads(paths["apply_manifest_json"].read_text(encoding="utf-8"))
    assert manifest["applied_weights"] == {"a": 0.75, "b": 0.25}


def test_apply_ensemble_weights_cli_writes_leaderboard_ready_outputs(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    weights_json = tmp_path / "weights.json"
    output_dir = tmp_path / "out"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )
    weights_json.write_text(json.dumps(_weights_payload()), encoding="utf-8")

    status = apply_weights_main(
        [
            "--estimate-csv",
            f"a={a_csv}",
            "--estimate-csv",
            f"b={b_csv}",
            "--weights-json",
            str(weights_json),
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_ensemble_applied_weights_manifest.json").exists()
    manifest = json.loads(
        (output_dir / "mmuad_track5_ensemble_applied_weights_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["applied_weights"] == {"a": 0.75, "b": 0.25}


def test_apply_ensemble_weights_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-apply-ensemble-weights"]
        == "raft_uav.mmuad.track5_estimate_ensemble_apply_weights:main"
    )
