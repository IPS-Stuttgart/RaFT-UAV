from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_forward_backward import (
    CandidateForwardBackwardConfig,
    attach_forward_backward_candidate_prior,
    write_forward_backward_outputs,
)
from raft_uav.mmuad.schema import CandidateFrame


@pytest.mark.parametrize("invalid_config", [False, 0, "", {}, []])
def test_forward_backward_rejects_falsy_invalid_config_before_empty_fast_path(
    invalid_config: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="config must be a CandidateForwardBackwardConfig or None",
    ):
        attach_forward_backward_candidate_prior(
            pd.DataFrame(),
            config=invalid_config,
        )


@pytest.mark.parametrize("invalid_config", [False, 0, "", {}, []])
def test_forward_backward_writer_rejects_falsy_invalid_config_before_writing(
    tmp_path: Path,
    invalid_config: object,
) -> None:
    output_csv = tmp_path / "forward-backward.csv"

    with pytest.raises(
        TypeError,
        match="config must be a CandidateForwardBackwardConfig or None",
    ):
        write_forward_backward_outputs(
            CandidateFrame(pd.DataFrame()),
            output_csv=output_csv,
            config=invalid_config,
        )

    assert not output_csv.exists()


def test_forward_backward_accepts_explicit_config_on_empty_input() -> None:
    result = attach_forward_backward_candidate_prior(
        pd.DataFrame(),
        config=CandidateForwardBackwardConfig(),
    )

    assert result.rows.empty
