import pytest

import raft_uav.imm_cli as imm_cli


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_imm_cli_rejects_nonfinite_mode_switch_time_constant(tmp_path, value):
    with pytest.raises(
        ValueError,
        match="imm_mode_switch_time_constant must be positive and finite",
    ):
        imm_cli.run_experiment(
            dataset_root=tmp_path,
            flight_name="flight",
            output_dir=tmp_path,
            imm_mode_switch_time_constant=value,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_imm_cli_rejects_nonfinite_fixed_lag_horizon(tmp_path, value):
    with pytest.raises(ValueError, match="smoother_lag_s must be nonnegative and finite"):
        imm_cli.run_experiment(
            dataset_root=tmp_path,
            flight_name="flight",
            output_dir=tmp_path,
            smoother="fixed-lag",
            smoother_lag_s=value,
        )


@pytest.mark.parametrize("field", ["rf_inflation_alpha", "radar_inflation_alpha"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_imm_cli_rejects_nonfinite_inflation_alpha(tmp_path, field, value):
    with pytest.raises(ValueError, match="inflation alphas must be positive and finite"):
        imm_cli.run_experiment(
            dataset_root=tmp_path,
            flight_name="flight",
            output_dir=tmp_path,
            **{field: value},
        )
