from __future__ import annotations

from raft_uav.experiments.method_registry import (
    get_method_spec,
    method_registry_frame,
    resolve_method_spec,
)


def test_method_registry_resolves_softk_dnh_environment_and_command() -> None:
    resolved = resolve_method_spec(
        "imm_tracklet_viterbi_fixed_lag_softk_dnh",
        dataset_root="data/raw/AADM2025Dryad",
        flight="Opt2",
        output_dir="outputs/demo",
    )

    assert resolved["env"]["RAFT_UAV_TRACKLET_SOFT_TOP_K_PATHS"] == "3"
    assert resolved["env"]["RAFT_UAV_DO_NO_HARM_RADAR_UPDATES"] == "1"
    assert resolved["env"]["RAFT_UAV_DO_NO_HARM_RADAR_UPDATE_POLICY"] == "1"
    assert resolved["command"][:3] == [
        "raft-uav",
        "run-baseline",
        "data/raw/AADM2025Dryad",
    ]
    assert "outputs/demo" in resolved["command"]


def test_method_registry_resolves_heteroscedastic_lofo_nis_command_and_env() -> None:
    resolved = resolve_method_spec(
        "hetero_cv_lofo_nis_fixed_lag",
        dataset_root="data/raw/AADM2025Dryad",
        flight="Opt2",
        output_dir="outputs/demo",
        uncertainty_model="outputs/models/uncertainty.json",
        nis_covariance_calibration_json="outputs/models/nis_covariance.json",
    )

    command = resolved["command"]
    assert command[:3] == [
        "raft-uav-heteroscedastic",
        "run-baseline",
        "data/raw/AADM2025Dryad",
    ]
    assert command[command.index("--uncertainty-model") + 1] == (
        "outputs/models/uncertainty.json"
    )
    assert "--fixed-lag-s" not in command
    assert command[command.index("--smoother") + 1] == "fixed-lag"
    assert command[command.index("--smoother-lag-s") + 1] == "20"
    assert resolved["env"]["RAFT_UAV_NIS_COVARIANCE_CALIBRATION_JSON"] == (
        "outputs/models/nis_covariance.json"
    )
    assert "{nis_covariance_calibration_json}" not in resolved["shell_command"]


def test_method_registry_dnh_rows_enable_radar_update_policy_gate() -> None:
    for method_id in (
        "imm_tracklet_viterbi_fixed_lag_dnh",
        "imm_tracklet_viterbi_fixed_lag_softk_dnh",
    ):
        resolved = resolve_method_spec(
            method_id,
            dataset_root="data/raw/AADM2025Dryad",
            flight="Opt2",
            output_dir="outputs/demo",
        )

        assert resolved["env"]["RAFT_UAV_DO_NO_HARM_RADAR_UPDATES"] == "1"
        assert resolved["env"]["RAFT_UAV_DO_NO_HARM_RADAR_UPDATE_POLICY"] == "1"


def test_method_registry_resolves_pyrecest_evidence_support_metadata() -> None:
    resolved = resolve_method_spec(
        "imm_tracklet_viterbi_fixed_lag_softk_dnh",
        dataset_root="data/raw/AADM2025Dryad",
        flight="Opt2",
        output_dir="outputs/demo",
    )

    support = resolved["evidence_support"]
    assert support["support_type"] == "truncated_lower_bound"
    assert not support["comparable"]
    assert support["lower_bound"]
    assert not support["headline_comparable"]
    assert support["diagnostics"]["top_k"] == 3


def test_method_registry_exposes_new_diagnostics() -> None:
    frame = method_registry_frame()
    ids = set(frame["method_id"])

    assert "radar_geometry_audit" in ids
    assert "nis_reliability" in ids
    assert "tracklet_feature_store" in ids
    assert "diagnostic" in set(get_method_spec("radar_geometry_audit").tags)
    assert "evidence_support_type" in frame.columns
    assert "evidence_lower_bound" in frame.columns
    softk = frame.loc[frame["method_id"] == "imm_tracklet_viterbi_fixed_lag_softk_dnh"].iloc[0]
    assert softk["evidence_support_type"] == "truncated_lower_bound"
    assert bool(softk["evidence_lower_bound"])
