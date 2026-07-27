import pandas as pd
import pytest

from raft_uav.uncertainty import HeteroscedasticUncertaintyModel, VarianceHead


def _rf_head() -> VarianceHead:
    return VarianceHead(
        source="rf",
        dimension="east",
        feature_names=("intercept",),
        coefficients=(0.0,),
        min_std_m=1.0,
        max_std_m=10.0,
        training_rows=1,
    )


def test_empty_apply_rejects_unknown_source() -> None:
    model = HeteroscedasticUncertaintyModel(heads=(_rf_head(),), metadata={})

    with pytest.raises(ValueError, match="model has no heads for source 'unknown'"):
        model.apply(pd.DataFrame(), source="unknown")


def test_empty_apply_rejects_missing_source_heads() -> None:
    model = HeteroscedasticUncertaintyModel(heads=(_rf_head(),), metadata={})

    with pytest.raises(ValueError, match="model has no heads for source 'radar'"):
        model.apply_radar(pd.DataFrame())


def test_empty_apply_preserves_valid_empty_frame() -> None:
    model = HeteroscedasticUncertaintyModel(heads=(_rf_head(),), metadata={})
    frame = pd.DataFrame(columns=["time_s"])

    result = model.apply_rf(frame)

    pd.testing.assert_frame_equal(result, frame)
