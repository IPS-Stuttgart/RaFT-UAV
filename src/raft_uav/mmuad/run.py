"""Convenience entrypoint for MMUAD sequence-root runs."""

from __future__ import annotations

import argparse
import sys

from raft_uav.mmuad.cli import main as track_main

_HELP_FLAGS = {"-h", "--help"}
_FLAG_ONLY_OPTIONS = {
    "--evaluation-require-complete-track5",
    "--infer-ug2-class-map-from-candidates",
    "--inspect-layout-only",
    "--native-ros-auto-topic-map",
    "--cluster-ranker-keep-confidence",
    "--no-apply-calibration",
    "--trajectory-completion-no-infer-grid",
    "--trajectory-completion-no-truth-timestamps",
    "--ug2-official-complete-to-sequence-timestamps",
    "--ug2-official-validate-on-write",
}
_LONG_VALUE_OPTION_SUFFIXES = (
    "-age-s",
    "-blend",
    "-classification",
    "-classifier",
    "-confidence",
    "-convention",
    "-csv",
    "-deg",
    "-delta-s",
    "-dir",
    "-extrapolation",
    "-file",
    "-fraction",
    "-frames",
    "-gap-s",
    "-gate-m",
    "-glob",
    "-id",
    "-json",
    "-m",
    "-manifest",
    "-mode",
    "-mps",
    "-mps2",
    "-name",
    "-path",
    "-penalty",
    "-points",
    "-policy",
    "-protocol",
    "-replacement",
    "-root",
    "-s",
    "-scale",
    "-source",
    "-std-deg",
    "-std-m",
    "-submission",
    "-tolerance-s",
    "-unit",
    "-voxels",
    "-weight",
    "-window-s",
    "-zip",
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in _HELP_FLAGS for arg in args):
        return track_main(args)
    if _has_explicit_sequence_root(args):
        return track_main(args)

    sequence_root_index = _sequence_root_index(args)
    if sequence_root_index is None:
        parser = argparse.ArgumentParser(
            prog="raft-uav-mmuad-run",
            description="run the MMUAD tracker on a sequence root",
            add_help=False,
        )
        parser.error("the following arguments are required: sequence_root")

    sequence_root = args[sequence_root_index]
    remainder = _forwarding_remainder(args, sequence_root_index=sequence_root_index)
    forwarded = ["--sequence-root", sequence_root, *remainder]
    return track_main(forwarded)


def _forwarding_remainder(args: list[str], *, sequence_root_index: int) -> list[str]:
    delimiter_index = sequence_root_index - 1
    return [
        arg
        for index, arg in enumerate(args)
        if index != sequence_root_index
        and not (index == delimiter_index and arg == "--")
    ]


def _has_explicit_sequence_root(args: list[str]) -> bool:
    for arg in args:
        if arg == "--":
            return False
        if arg == "--sequence-root" or arg.startswith("--sequence-root="):
            return True
    return False


def _sequence_root_index(args: list[str]) -> int | None:
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            return index + 1 if index + 1 < len(args) else None
        if _option_consumes_next(arg):
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        return index
    return None


def _option_consumes_next(arg: str) -> bool:
    if arg in _HELP_FLAGS or not arg.startswith("-") or arg == "--":
        return False
    if "=" in arg:
        return False
    if arg.startswith("--no-"):
        return False
    if arg in _FLAG_ONLY_OPTIONS:
        return False
    if arg.startswith("--"):
        return arg.endswith(_LONG_VALUE_OPTION_SUFFIXES)
    return True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
