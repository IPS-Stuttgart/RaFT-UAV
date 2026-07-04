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

_TRACKLET_ASSOCIATION_FLAG = "--radar-association"
_TRACKLET_ASSOCIATION_MODE = "tracklet-viterbi"
_HELP_FLAGS = {"-h", "--help"}


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``run-baseline`` with learned covariance and tracklet-Viterbi enabled."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if _is_help_request(raw_argv):
        return tracklet_viterbi_cli.main(_ensure_tracklet_viterbi_association(raw_argv))

    uncertainty_model_path, delegated_argv = _extract_uncertainty_model_arg(raw_argv)
    if not delegated_argv or delegated_argv[0] != "run-baseline":
        raise SystemExit(
            "raft-uav-heteroscedastic-tracklet-viterbi wraps only the "
            "'run-baseline' subcommand. Example: "
            "raft-uav-heteroscedastic-tracklet-viterbi run-baseline DATA "
            "--flight FLIGHT --uncertainty-model MODEL.json "
            "--radar-association tracklet-viterbi --tracklet-variant "
            "range-covariance"
        )
    delegated_argv = _ensure_tracklet_viterbi_association(delegated_argv)
    with heteroscedastic_covariance_hooks(uncertainty_model_path):
        return tracklet_viterbi_cli.main(list(delegated_argv))


def _is_help_request(argv: Sequence[str]) -> bool:
    """Return whether the invocation only needs parser help, not a model file."""

    return any(arg in _HELP_FLAGS for arg in argv)


def _ensure_tracklet_viterbi_association(argv: list[str]) -> list[str]:
    """Default the dedicated wrapper to tracklet-Viterbi association.

    The base ``run-baseline`` parser defaults to ``catprob``.  For this wrapper,
    that is surprising because the command name promises tracklet-Viterbi; still,
    preserve an explicit ``--radar-association`` so callers can run intentional
    comparison rows through the same learned-covariance hooks.
    """

    for index, arg in enumerate(argv):
        if arg == "--":
            return [
                *argv[:index],
                _TRACKLET_ASSOCIATION_FLAG,
                _TRACKLET_ASSOCIATION_MODE,
                *argv[index:],
            ]
        if arg == _TRACKLET_ASSOCIATION_FLAG or arg.startswith(
            f"{_TRACKLET_ASSOCIATION_FLAG}="
        ):
            return argv
    return [*argv, _TRACKLET_ASSOCIATION_FLAG, _TRACKLET_ASSOCIATION_MODE]


if __name__ == "__main__":
    raise SystemExit(main())
