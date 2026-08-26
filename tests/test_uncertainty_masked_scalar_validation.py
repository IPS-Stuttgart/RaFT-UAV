import numpy as np
import pandas as pd
import pytest

from raft_uav.uncertainty import (
    VarianceHead,
    fit_heteroscedastic_uncertainty_model,
)


@pytest.mark.parametrize("field", ["ridge_lambda", "max_time_delta_s"])
@pytest.mark.parametrize(
    "masked_value",
    [np.ma.masked, np.ma.array(3.0, mask=True)],
    ids=["masked-singleton", "masked-hidden-payload"],
)
def test_fit_rejects_masked_scalar_controls(field, masked_value):
    with pytest.raises(ValueError, match=rf"{field} must be a finite number"):
        fit_heteroscedastic_uncertainty_model(
            rf=None,
            radar=None,
            truth=pd.DataFrame(),
            **{field: masked_value},
        )


@pytest.mark.parametrize("field", ["min_std_m", "max_std_m"])
def test_variance_head_rejects_masked_scalar_bounds(field):
    values = {
        "source": "rf",
        "dimension": "east",
        "feature_names": ("intercept",),
        "coefficients": (0.0,),
        "min_std_m": 1.0,
        "max_std_m": 10.0,
        "training_rows": 1,
    }
    values[field] = np.ma.array(values[field], mask=True)

    with pytest.raises(ValueError, match=rf"{field} must be a finite real scalar"):
        VarianceHead(**values)


def test_variance_head_accepts_unmasked_zero_dimensional_bounds():
    head = VarianceHead(
        source="rf",
        dimension="east",
        feature_names=("intercept",),
        coefficients=(0.0,),
        min_std_m=np.ma.array(1.0, mask=False),
        max_std_m=np.ma.array(10.0, mask=False),
        training_rows=1,
    )

    assert float(head.min_std_m) == 1.0
    assert float(head.max_std_m) == 10.0
