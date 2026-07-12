from __future__ import annotations

import math

import pandas as pd
import pytest

from raft_uav.uncertainty import (
    HeteroscedasticUncertaintyModel,
    VarianceHead,
    fit_heteroscedastic_uncertainty_model,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ridge_lambda": -1.0}, "ridge_lambda"),
        ({"ridge_lambda": math.nan}, "ridge_lambda"),
        ({"max_time_delta_s": -1.0}, "max_time_delta_s"),
        ({"max_time_delta_s": math.inf}, "max_time_delta_s"),
        ({"min_std_m": {"rf": {"east": 0.0}}}, "rf.east min_std_m"),
        ({"max_std_m": {"radar": {"up": math.nan}}}, "radar.up max_std_m"),
        (
            {
                "min_std_m": {"rf": {"north": 20.0}},
                "max_std_m": {"rf": {"north": 10.0}},
            },
            "rf.north min_std_m must not exceed max_std_m",
        ),
    ],
)
def test_fitter_rejects_invalid_controls_before_empty_source_shortcut(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_heteroscedastic_uncertainty_model(
            rf=None,
            radar=None,
            truth=pd.DataFrame(),
            **kwargs,
        )


def test_variance_head_rejects_reversed_bounds_before_prediction() -> None:
    head = VarianceHead(
        source="rf",
        dimension="east",
        feature_names=("intercept",),
        coefficients=(0.0,),
        min_std_m=20.0,
        max_std_m=10.0,
        training_rows=1,
    )

    with pytest.raises(ValueError, match="rf.east min_std_m must not exceed max_std_m"):
        head.predict(pd.DataFrame(index=[0]))


def test_model_loader_rejects_nonfinite_variance_bounds() -> None:
    payload = {
        "schema_version": 1,
        "metadata": {},
        "heads": [
            {
                "source": "radar",
                "dimension": "up",
                "feature_names": ["intercept"],
                "coefficients": [0.0],
                "min_std_m": 1.0,
                "max_std_m": math.inf,
                "training_rows": 1,
            }
        ],
    }

    with pytest.raises(ValueError, match="radar.up max_std_m"):
        HeteroscedasticUncertaintyModel.from_dict(payload)
