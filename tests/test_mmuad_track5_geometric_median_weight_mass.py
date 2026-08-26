from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_geometric_median_ensemble import (
    build_track5_geometric_median_ensemble,
    write_track5_geometric_median_outputs,
)


def test_build_rejects_missing_estimate_inputs_before_empty_template() -> None:
    with pytest.raises(ValueError, match="at least one estimate input is required"):
        build_track5_geometric_median_ensemble([], pd.DataFrame())


@pytest.mark.parametrize("weights", [[0.0], [0.0, 0.0]])
def test_build_rejects_zero_weight_mass_before_empty_template(
    weights: list[float],
) -> None:
    inputs = [
        (f"candidate-{index}", pd.DataFrame(), weight)
        for index, weight in enumerate(weights)
    ]

    with pytest.raises(
        ValueError,
        match="estimate weights must have positive finite mass",
    ):
        build_track5_geometric_median_ensemble(inputs, pd.DataFrame())


def test_build_allows_zero_weight_members_when_total_mass_is_positive() -> None:
    estimates, diagnostics = build_track5_geometric_median_ensemble(
        [
            ("disabled", pd.DataFrame(), 0.0),
            ("enabled", pd.DataFrame(), 1.0),
        ],
        pd.DataFrame(),
    )

    assert estimates.empty
    assert diagnostics.empty


def test_writer_rejects_missing_inputs_before_output_creation(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="at least one estimate input is required"):
        write_track5_geometric_median_outputs(
            estimate_inputs=[],
            template=pd.DataFrame(),
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_writer_rejects_zero_weight_mass_before_file_io(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    missing_input = tmp_path / "missing.csv"

    with pytest.raises(
        ValueError,
        match="estimate weights must have positive finite mass",
    ):
        write_track5_geometric_median_outputs(
            estimate_inputs=[EstimateInput("missing", missing_input, 0.0)],
            template=pd.DataFrame(),
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not missing_input.exists()
