from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import (
    resample_estimates_to_track5_template,
)


def _estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
            "classification": [1],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]})


@pytest.mark.parametrize(
    ("extra_column", "extra_value"),
    [
        (" Sequence ", "seq9999"),
        (" timestamp_s ", 10.0),
        (" Class ", 2),
    ],
)
def test_template_resample_rejects_ambiguous_estimate_aliases(
    extra_column: str,
    extra_value: object,
) -> None:
    estimates = _estimates()
    estimates[extra_column] = [extra_value]

    with pytest.raises(ValueError, match="table contains ambiguous columns matching"):
        resample_estimates_to_track5_template(estimates, _template())


def test_template_resample_rejects_ambiguous_template_aliases() -> None:
    template = _template()
    template[" sequence "] = ["seq9999"]

    with pytest.raises(ValueError, match="table contains ambiguous columns matching"):
        resample_estimates_to_track5_template(_estimates(), template)


def test_template_resample_rejects_exact_duplicate_template_columns() -> None:
    template = pd.DataFrame(
        [["seq0001", "seq9999", 0.0]],
        columns=["Sequence", "Sequence", "Timestamp"],
    )

    with pytest.raises(ValueError, match="table contains ambiguous columns matching"):
        resample_estimates_to_track5_template(_estimates(), template)


def test_template_resample_accepts_unique_padded_aliases() -> None:
    estimates = _estimates().rename(
        columns={"sequence_id": " Sequence ", "time_s": " Time "}
    )
    template = _template().rename(
        columns={"Sequence": " Sequence ", "Timestamp": " Timestamp "}
    )

    resampled, diagnostics = resample_estimates_to_track5_template(estimates, template)

    assert resampled["sequence_id"].tolist() == ["seq0001"]
    assert diagnostics["valid"].tolist() == [True]
