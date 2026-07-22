import numpy as np
import pandas as pd
import pytest

import raft_uav.io.aerpaw as aerpaw


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        False,
        1.0 + 0.0j,
        np.array([75.0]),
        np.ma.masked,
    ],
)
def test_normalize_rf_rejects_invalid_default_std_before_legacy_execution(value):
    with pytest.raises(
        ValueError,
        match="default_std_m must be a positive finite real scalar",
    ):
        aerpaw.normalize_rf(
            pd.DataFrame(),
            object(),
            pd.Timestamp("2026-01-01"),
            default_std_m=value,
        )


def test_normalize_rf_normalizes_valid_scalar_like_default_std(monkeypatch):
    sentinel = pd.DataFrame({"normalized": [True]})
    captured = {}

    def fake_normalize_rf(
        rf,
        projector,
        truth_origin_time,
        default_std_m=75.0,
        clock_offset_s=aerpaw.DEFAULT_RF_CLOCK_OFFSET_S,
    ):
        captured.update(
            {
                "rf": rf,
                "projector": projector,
                "truth_origin_time": truth_origin_time,
                "default_std_m": default_std_m,
                "clock_offset_s": clock_offset_s,
            }
        )
        return sentinel

    monkeypatch.setattr(aerpaw, "_original_normalize_rf", fake_normalize_rf)
    frame = pd.DataFrame({"Time": []})
    projector = object()
    origin_time = pd.Timestamp("2026-01-01")

    result = aerpaw.normalize_rf(
        frame,
        projector,
        origin_time,
        default_std_m=np.array("12.5"),
        clock_offset_s=-3.0,
    )

    assert result is sentinel
    assert captured["rf"] is frame
    assert captured["projector"] is projector
    assert captured["truth_origin_time"] == origin_time
    assert captured["default_std_m"] == 12.5
    assert type(captured["default_std_m"]) is float
    assert captured["clock_offset_s"] == -3.0
