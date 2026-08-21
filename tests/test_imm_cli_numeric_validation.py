import numpy as np
import pytest

import raft_uav._imm_cli_numeric_validation_patch as numeric_patch
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


@pytest.mark.parametrize(
    ("field", "extra", "message"),
    [
        (
            "imm_mode_switch_time_constant",
            {},
            "imm_mode_switch_time_constant must be positive and finite",
        ),
        (
            "smoother_lag_s",
            {"smoother": "fixed-lag"},
            "smoother_lag_s must be nonnegative and finite",
        ),
        ("rf_inflation_alpha", {}, "inflation alphas must be positive and finite"),
        ("radar_inflation_alpha", {}, "inflation alphas must be positive and finite"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        np.asarray(True),
        np.asarray([1.0]),
    ],
)
def test_imm_cli_rejects_lossy_numeric_control_coercions(
    tmp_path,
    field,
    extra,
    message,
    value,
):
    with pytest.raises(ValueError, match=message):
        imm_cli.run_experiment(
            dataset_root=tmp_path,
            flight_name="flight",
            output_dir=tmp_path,
            **extra,
            **{field: value},
        )


def test_imm_cli_numeric_guard_accepts_zero_dimensional_real_scalars():
    numeric_patch._require_finite(np.asarray(2.0), message="invalid")
    numeric_patch._require_finite(np.float64(2.0), message="invalid")


def test_imm_cli_numeric_guard_rejects_nested_non_scalar_array():
    nested = np.empty((), dtype=object)
    nested[()] = np.asarray([2.0])
    with pytest.raises(ValueError, match="invalid"):
        numeric_patch._require_finite(nested, message="invalid")
