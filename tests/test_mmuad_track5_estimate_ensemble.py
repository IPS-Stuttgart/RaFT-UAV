from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import main as ensemble_main
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble import write_track5_estimate_ensemble_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 5.0, 0.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2, 2, 1],
        }
    )


def _estimate_a() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 4.0],
            "state_y_m": [0.0, 0.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0],
        }
    )


def _estimate_b() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [2.0, 12.0, 8.0],
            "state_y_m": [2.0, 2.0, 8.0],
            "state_z_m": [2.0, 2.0, 8.0],
        }
    )


def _estimate_outlier() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [1000.0, 1010.0, 1000.0],
            "state_y_m": [1000.0, 1000.0, 1000.0],
            "state_z_m": [1000.0, 1000.0, 1000.0],
        }
    )


def test_parse_estimate_spec_accepts_label_path_and_weight() -> None:
    item = parse_estimate_spec("robust=/tmp/estimates.csv@0.25")
    assert item.label == "robust"
    assert str(item.path).endswith("estimates.csv")
    assert item.weight == pytest.approx(0.25)


@pytest.mark.parametrize("weight", ["nan", "inf", "-0.1"])
def test_parse_estimate_spec_rejects_invalid_weights(weight: str) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        parse_estimate_spec(f"bad=/tmp/estimates.csv@{weight}")


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1.0])
def test_track5_estimate_ensemble_rejects_invalid_runtime_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_track5_estimate_ensemble([("bad", _estimate_a(), weight)], _template())


def test_track5_estimate_ensemble_weighted_average_after_template_resample() -> None:
    ensemble, diagnostics = build_track5_estimate_ensemble(
        [
            ("a", _estimate_a(), 0.75),
            ("b", _estimate_b(), 0.25),
        ],
        _template(),
    )

    midpoint = ensemble.loc[
        (ensemble["sequence_id"] == "seq0001") & (ensemble["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(5.5)
    assert midpoint["state_y_m"] == pytest.approx(0.5)
    assert midpoint["state_z_m"] == pytest.approx(0.5)
    seq2 = ensemble.loc[ensemble["sequence_id"] == "seq0002"].iloc[0]
    assert seq2["state_x_m"] == pytest.approx(5.0)
    assert seq2["ensemble_source_count"] == 2
    assert diagnostics["valid_input_count"].tolist() == [2, 2, 2]


def test_track5_estimate_ensemble_epoch_timestamps_do_not_cross_match() -> None:
    base_time = 1_706_255_054.386069
    template = pd.DataFrame(
        {
            "Sequence": ["seq_epoch", "seq_epoch"],
            "Timestamp": [base_time, base_time + 1.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [1, 1],
        }
    )
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq_epoch", "seq_epoch"],
            "time_s": [base_time, base_time + 1.0],
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [1.0, 1.0],
        }
    )

    ensemble, diagnostics = build_track5_estimate_ensemble(
        [("epoch", estimates, 1.0)],
        template,
        max_nearest_time_delta_s=0.0,
    )

    assert ensemble["state_x_m"].tolist() == pytest.approx([0.0, 100.0])
    assert ensemble["ensemble_source_count"].tolist() == [1, 1]
    assert diagnostics["candidate_input_count"].tolist() == [1, 1]
    assert diagnostics["valid_input_count"].tolist() == [1, 1]


def test_track5_estimate_ensemble_weighted_median_rejects_outlier() -> None:
    mean_ensemble, _ = build_track5_estimate_ensemble(
        [
            ("a", _estimate_a(), 1.0),
            ("b", _estimate_b(), 1.0),
            ("outlier", _estimate_outlier(), 1.0),
        ],
        _template(),
    )
    robust_ensemble, diagnostics = build_track5_estimate_ensemble(
        [
            ("a", _estimate_a(), 1.0),
            ("b", _estimate_b(), 1.0),
            ("outlier", _estimate_outlier(), 1.0),
        ],
        _template(),
        aggregation_policy="weighted-median",
    )

    midpoint_mean = mean_ensemble.loc[
        (mean_ensemble["sequence_id"] == "seq0001") & (mean_ensemble["time_s"] == 5.0)
    ].iloc[0]
    midpoint_robust = robust_ensemble.loc[
        (robust_ensemble["sequence_id"] == "seq0001") & (robust_ensemble["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint_mean["state_x_m"] > 300.0
    assert midpoint_robust["state_x_m"] == pytest.approx(7.0)
    assert midpoint_robust["state_y_m"] == pytest.approx(2.0)
    assert midpoint_robust["ensemble_policy"] == "weighted-median"
    assert diagnostics["position_spread_m"].notna().all()


def test_track5_estimate_ensemble_trimmed_mean_rejects_single_low_high_outliers() -> None:
    robust_ensemble, _ = build_track5_estimate_ensemble(
        [
            ("a", _estimate_a(), 1.0),
            ("b", _estimate_b(), 1.0),
            ("outlier", _estimate_outlier(), 1.0),
            ("low_outlier", _estimate_a().assign(state_x_m=-1000.0), 1.0),
        ],
        _template(),
        aggregation_policy="trimmed-mean",
        trim_fraction=0.25,
    )
    midpoint = robust_ensemble.loc[
        (robust_ensemble["sequence_id"] == "seq0001") & (robust_ensemble["time_s"] == 5.0)
    ].iloc[0]
    assert midpoint["state_x_m"] == pytest.approx(6.0)


def test_track5_estimate_ensemble_writes_leaderboard_ready_artifacts(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)
    paths = write_track5_estimate_ensemble_outputs(
        estimate_inputs=[
            parse_estimate_spec(f"a={a_csv}@0.75"),
            parse_estimate_spec(f"b={b_csv}@0.25"),
        ],
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2", "seq0002": "1"},
        aggregation_policy="weighted-median",
    )

    assert paths["official_zip"].exists()
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    assert manifest["valid_ensemble_rows"] == 3
    assert manifest["aggregation_policy"] == "weighted-median"
    with ZipFile(paths["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    official = pd.read_csv(paths["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 1]


def test_track5_estimate_ensemble_manifest_preserves_generator_inputs(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)
    estimate_inputs = (
        parse_estimate_spec(spec)
        for spec in (f"a={a_csv}@0.75", f"b={b_csv}@0.25")
    )

    paths = write_track5_estimate_ensemble_outputs(
        estimate_inputs=estimate_inputs,
        template=_template(),
        output_dir=tmp_path / "out",
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert [item["label"] for item in manifest["estimate_inputs"]] == ["a", "b"]
    assert [item["weight"] for item in manifest["estimate_inputs"]] == pytest.approx([0.75, 0.25])
    assert [item["label"] for item in manifest["input_summaries"]] == ["a", "b"]


def test_track5_estimate_ensemble_cli_writes_outputs(tmp_path: Path) -> None:
    a_csv = tmp_path / "a.csv"
    b_csv = tmp_path / "b.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _estimate_a().to_csv(a_csv, index=False)
    _estimate_b().to_csv(b_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seq0001", "seq0002"], "uav_type": [2, 1]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = ensemble_main(
        [
            "--estimate-csv",
            f"a={a_csv}@0.75",
            "--estimate-csv",
            f"b={b_csv}@0.25",
            "--template",
            str(template_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--aggregation-policy",
            "weighted-median",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_track5_ensemble_manifest.json").exists()
    manifest = json.loads((output_dir / "mmuad_track5_ensemble_manifest.json").read_text())
    assert manifest["aggregation_policy"] == "weighted-median"


def test_track5_estimate_ensemble_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-estimate-ensemble"]
        == "raft_uav.mmuad.track5_estimate_ensemble_text_cli:main"
    )
