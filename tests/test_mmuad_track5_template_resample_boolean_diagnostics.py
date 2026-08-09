from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import (
    summarize_template_resample_diagnostics,
)


def test_template_resample_summary_parses_csv_float_boolean_flags() -> None:
    diagnostics = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "valid": pd.Series([1.0, 0.0, pd.NA], dtype="Float64"),
            "extrapolated": ["1.0", "0.0", None],
            "large_gap_fallback": pd.Series([0.0, 1.0, pd.NA], dtype="Float64"),
            "resample_method": ["linear", "nearest", "linear"],
            "source_row_count": [2, 2, 2],
            "nearest_time_delta_s": [0.0, 1.0, 2.0],
            "interpolation_gap_s": [0.0, 5.0, 0.0],
            "classification_source": ["sequence-mode", "nearest", "none"],
        }
    )

    summary = summarize_template_resample_diagnostics(diagnostics).iloc[0]

    assert summary["valid_row_count"] == 1
    assert summary["invalid_row_count"] == 2
    assert summary["extrapolated_row_count"] == 1
    assert summary["large_gap_fallback_row_count"] == 1


def _object_series(value: object) -> pd.Series:
    values = np.empty(1, dtype=object)
    values[0] = value
    return pd.Series(values, index=["row-a"])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("valid", 2),
        ("valid", 1.0 + 0.0j),
        ("extrapolated", np.array([1, 0])),
        ("large_gap_fallback", "maybe"),
    ],
)
def test_template_resample_summary_rejects_malformed_boolean_flags(
    column: str,
    value: object,
) -> None:
    diagnostics = pd.DataFrame(
        {
            "sequence_id": _object_series("seq0001"),
            "valid": _object_series(False),
            "extrapolated": _object_series(False),
            "large_gap_fallback": _object_series(False),
        }
    )
    diagnostics[column] = _object_series(value)

    with pytest.raises(ValueError, match=column):
        summarize_template_resample_diagnostics(diagnostics)


def test_template_resample_summary_accepts_scalar_boxes_and_masked_missing() -> None:
    masked = np.ma.array(1.0, mask=True)
    valid_values = np.empty(3, dtype=object)
    valid_values[:] = [np.array(1.0), np.array(0.0), masked]
    diagnostics = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "valid": pd.Series(valid_values),
            "extrapolated": pd.Series(["true", "false", None], dtype=object),
            "large_gap_fallback": pd.Series(
                [np.bool_(False), np.bool_(True), pd.NA],
                dtype=object,
            ),
        }
    )

    summary = summarize_template_resample_diagnostics(diagnostics).iloc[0]

    assert summary["valid_row_count"] == 1
    assert summary["invalid_row_count"] == 2
    assert summary["extrapolated_row_count"] == 1
    assert summary["large_gap_fallback_row_count"] == 1
