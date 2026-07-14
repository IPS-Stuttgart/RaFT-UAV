import numpy as np
import pandas as pd
import pytest

from raft_uav.uncertainty import (
    HeteroscedasticUncertaintyModel,
    VarianceHead,
    _aligned_residuals,
)


def make_head(**overrides):
    values = {
        "source": "rf",
        "dimension": "east",
        "feature_names": ("intercept",),
        "coefficients": (1.0,),
        "min_std_m": 1.0,
        "max_std_m": 10.0,
        "training_rows": 3,
    }
    values.update(overrides)
    return VarianceHead(**values)


def test_rejects_nonfinite_coefficients():
    with pytest.raises(ValueError, match="coefficients must be finite"):
        make_head(coefficients=(np.nan,))


def test_rejects_reversed_std_bounds():
    with pytest.raises(ValueError, match="greater than or equal"):
        make_head(min_std_m=5.0, max_std_m=4.0)


def test_rejects_unknown_features():
    with pytest.raises(ValueError, match="unknown rf uncertainty features"):
        make_head(feature_names=("intercept", "log1p_cepp"), coefficients=(1.0, 2.0))


def test_rejects_duplicate_heads():
    head = make_head()
    with pytest.raises(ValueError, match="duplicate uncertainty variance head"):
        HeteroscedasticUncertaintyModel(heads=(head, head), metadata={})


def test_empty_model_sentinel_remains_loadable():
    model = HeteroscedasticUncertaintyModel.from_dict(
        {"schema_version": 1, "metadata": {}, "heads": []}
    )
    assert model.heads == ()


def test_valid_partial_model_still_applies():
    head = make_head(coefficients=(np.log(4.0),))
    model = HeteroscedasticUncertaintyModel(heads=(head,), metadata={})
    out = model.apply(pd.DataFrame({"time_s": [0.0]}), source="rf")
    assert out["cov_ee"].tolist() == pytest.approx([4.0])


def test_preserves_private_legacy_helpers():
    assert callable(_aligned_residuals)
