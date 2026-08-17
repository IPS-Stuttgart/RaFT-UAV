from pathlib import Path

import pytest

from raft_uav.bias_cli import train_bias_model


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_time_delta_s", float("nan"), "max_time_delta_s must be positive"),
        ("max_time_delta_s", float("inf"), "max_time_delta_s must be positive"),
        (
            "max_position_error_m",
            float("nan"),
            "max_position_error_m must be positive",
        ),
        (
            "max_position_error_m",
            float("inf"),
            "max_position_error_m must be positive",
        ),
        ("ridge_alpha", float("nan"), "ridge_alpha must be nonnegative"),
        ("ridge_alpha", float("inf"), "ridge_alpha must be nonnegative"),
    ],
)
def test_train_bias_model_rejects_nonfinite_hyperparameters(
    field: str,
    value: float,
    message: str,
) -> None:
    kwargs = {
        "dataset_root": Path("unused-dataset"),
        "requested_flights": None,
        "output_path": Path("unused-model.json"),
        "max_time_delta_s": 2.0,
        "max_position_error_m": 250.0,
        "ridge_alpha": 1.0,
        "min_samples": 5,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        train_bias_model(**kwargs)
