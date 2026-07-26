from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_column_adapter import normalize_uncertainty_estimate_inputs
from raft_uav.mmuad.track5_uncertainty_ensemble import build_track5_uncertainty_ensemble


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def _zero_padded_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def _estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
            "predicted_sigma_m": [1.0],
        }
    )


def test_uncertainty_ensemble_preserves_padded_sequence_id_headers(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    estimate_csv.write_text(
        " sequence_id ,time_s,state_x_m,state_y_m,state_z_m,predicted_sigma_m\n"
        "001,0.0,1.0,2.0,3.0,2.0\n",
        encoding="utf-8",
    )

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        [EstimateInput("estimate", estimate_csv, 1.0)],
        template=_zero_padded_template(),
    )

    assert estimates.loc[0, "sequence_id"] == "001"
    assert estimates.loc[0, "ensemble_source_count"] == 1
    assert estimates.loc[0, "state_x_m"] == pytest.approx(1.0)
    assert estimates.loc[0, "state_y_m"] == pytest.approx(2.0)
    assert estimates.loc[0, "state_z_m"] == pytest.approx(3.0)
    assert estimates.loc[0, "ensemble_effective_sigma_m"] == pytest.approx(2.0)
    assert diagnostics.loc[0, "valid_input_count"] == 1


def test_uncertainty_adapter_rejects_nonfinite_fallback_sigma(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimate().drop(columns=["predicted_sigma_m"]).to_csv(estimate_csv, index=False)

    with pytest.raises(ValueError, match="fallback_sigma_m"):
        normalize_uncertainty_estimate_inputs(
            [EstimateInput("estimate", estimate_csv, 1.0)],
            output_dir=tmp_path / "out",
            fallback_sigma_m=np.inf,
            require_uncertainty=False,
        )


def test_uncertainty_ensemble_rejects_nonfinite_or_inverted_sigma_knobs(
    tmp_path: Path,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    _estimate().to_csv(estimate_csv, index=False)
    inputs = [EstimateInput("estimate", estimate_csv, 1.0)]

    with pytest.raises(ValueError, match="fallback_sigma_m"):
        build_track5_uncertainty_ensemble(
            inputs,
            template=_template(),
            fallback_sigma_m=np.nan,
        )

    with pytest.raises(ValueError, match="sigma_max_m"):
        build_track5_uncertainty_ensemble(
            inputs,
            template=_template(),
            sigma_min_m=10.0,
            sigma_max_m=1.0,
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("fallback_sigma_m", True),
        ("sigma_min_m", np.array([1.0])),
        ("sigma_max_m", np.ma.masked),
        ("fallback_sigma_m", 1.0 + 0.0j),
    ],
)
def test_uncertainty_ensemble_rejects_malformed_sigma_controls_before_empty_return(
    tmp_path: Path,
    parameter: str,
    value: object,
) -> None:
    inputs = [EstimateInput("estimate", tmp_path / "missing.csv", 1.0)]
    empty_template = _template().iloc[0:0]
    controls = {
        "fallback_sigma_m": 30.0,
        "sigma_min_m": 1.0,
        "sigma_max_m": 100.0,
    }
    controls[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        build_track5_uncertainty_ensemble(
            inputs,
            template=empty_template,
            **controls,
        )


def test_uncertainty_ensemble_accepts_scalar_like_sigma_controls_on_empty_template(
    tmp_path: Path,
) -> None:
    inputs = [EstimateInput("estimate", tmp_path / "missing.csv", 1.0)]

    estimates, diagnostics = build_track5_uncertainty_ensemble(
        inputs,
        template=_template().iloc[0:0],
        fallback_sigma_m="30",
        sigma_min_m=np.array(1.0),
        sigma_max_m=np.float64(100.0),
    )

    assert estimates.empty
    assert diagnostics.empty
