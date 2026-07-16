from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_soft_class_ensemble import (
    build_soft_class_conditioned_estimate_ensemble,
)


def test_soft_class_ensemble_accepts_integer_like_float_labels(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "time_s": [0.0, 0.0],
            "state_x_m": [0.0, 10.0],
            "state_y_m": [0.0, 10.0],
            "state_z_m": [0.0, 10.0],
        }
    ).to_csv(first_path, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "time_s": [0.0, 0.0],
            "state_x_m": [100.0, 20.0],
            "state_y_m": [100.0, 20.0],
            "state_z_m": [100.0, 20.0],
        }
    ).to_csv(second_path, index=False)

    estimates, diagnostics = build_soft_class_conditioned_estimate_ensemble(
        [EstimateInput("first", first_path), EstimateInput("second", second_path)],
        template=pd.DataFrame(
            {
                "Sequence": ["seq0001", "seq0002"],
                "Timestamp": [0.0, 0.0],
            }
        ),
        class_probabilities=pd.DataFrame(
            {
                "sequence_id": ["seq0001", "seq0002"],
                "predicted_class": [0.0, 1.0],
            }
        ),
        weight_config={
            "global_weights": {"first": 1.0, "second": 0.0},
            "class_weights": {
                "0": {"first": 0.0, "second": 1.0},
                "1": {"first": 0.0, "second": 1.0},
            },
        },
    )

    assert estimates["state_x_m"].tolist() == pytest.approx([100.0, 20.0])
    assert estimates["soft_class_probability_available"].tolist() == [True, True]
    assert diagnostics["effective_probability_sum"].tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("predicted_class", [1.5, 1.000001, "1.000001", 2.999999])
def test_soft_class_ensemble_rejects_non_integral_predicted_labels(
    tmp_path: Path,
    predicted_class: object,
) -> None:
    estimate_path = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    ).to_csv(estimate_path, index=False)

    with pytest.raises(ValueError, match="official Track 5 class IDs"):
        build_soft_class_conditioned_estimate_ensemble(
            [EstimateInput("estimate", estimate_path)],
            template=pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]}),
            class_probabilities=pd.DataFrame(
                {"sequence_id": ["seq0001"], "predicted_class": [predicted_class]}
            ),
            weight_config={
                "global_weights": {"estimate": 1.0},
                "class_weights": {},
            },
        )
