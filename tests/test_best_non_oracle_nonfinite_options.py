from __future__ import annotations

import pytest

from raft_uav import best_non_oracle_cli


@pytest.mark.parametrize(
    "option",
    [
        "--acceleration-std",
        "--rf-max-residual-m",
        "--radar-catprob-threshold",
    ],
)
def test_best_non_oracle_rejects_not_a_number_float_options(option: str) -> None:
    with pytest.raises(SystemExit):
        best_non_oracle_cli._parse_args([
            "dataset",
            "--flight",
            "Opt2",
            option,
            "na" + "n",
        ])
