"""Run tracklet-Viterbi with learned heteroscedastic RF/radar covariance.

This entry point combines two existing wrappers: the canonical
``tracklet-viterbi`` CLI and the heteroscedastic covariance hooks.  It is meant
for leakage-safe leave-one-flight-out benchmark rows where the uncertainty
model is trained on the other flights and consumed by the held-out run.
"""

from __future__ import annotations

from collections.abc import Sequence
import sys

from raft_uav import tracklet_viterbi_cli
from raft_uav.heteroscedastic_cli import (
    _extract_uncertainty_model_arg,
    heteroscedastic_covariance_hooks,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``run-baseline`` with learned covariance and tracklet-Viterbi enabled."""

    uncertainty_model_path, delegated_argv = _extract_uncertainty_model_arg(
        sys.argv[1:] if argv is None else argv
    )
    if not delegated_argv or delegated_argv[0] != "run-baseline":
        raise SystemExit(
            "raft-uav-heteroscedastic-tracklet-viterbi wraps only the "
            "'run-baseline' subcommand. Example: "
            "raft-uav-heteroscedastic-tracklet-viterbi run-baseline DATA "
            "--flight FLIGHT --uncertainty-model MODEL.json "
            "--radar-association tracklet-viterbi --tracklet-variant "
            "range-covariance"
        )
    with heteroscedastic_covariance_hooks(uncertainty_model_path):
        return tracklet_viterbi_cli.main(list(delegated_argv))


if __name__ == "__main__":
    raise SystemExit(main())
