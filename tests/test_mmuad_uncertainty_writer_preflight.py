from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_ensemble import write_track5_uncertainty_ensemble_outputs


def _template() -> pd.DataFrame:
    return pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]})


def test_uncertainty_writer_rejects_empty_inputs_before_creating_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="at least one estimate input"):
        write_track5_uncertainty_ensemble_outputs(
            estimate_inputs=[],
            template=_template(),
            output_dir=output,
        )

    assert not output.exists()


def test_uncertainty_writer_reads_inputs_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "out"
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        write_track5_uncertainty_ensemble_outputs(
            estimate_inputs=[EstimateInput("missing", missing, 1.0)],
            template=_template(),
            output_dir=output,
        )

    assert not output.exists()
