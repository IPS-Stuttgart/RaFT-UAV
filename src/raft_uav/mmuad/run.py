"""Convenience entrypoint for MMUAD sequence-root runs."""

from __future__ import annotations

import argparse
import sys

from raft_uav.mmuad.cli import main as track_main

_HELP_FLAGS = {"-h", "--help"}
# Options from ``raft_uav.mmuad.cli`` that consume the following argv token.  The
# sequence-root convenience wrapper has to skip these values before it can safely
# identify the first positional sequence root.
_VALUE_FLAGS = {
    "--acceleration-std-mps2",
    "--calibration-file",
    "--calibration-json",
    "--camera-calibration-file",
    "--camera-detections-csv",
    "--camera-detections-file",
    "--camera-fixed-depth-m",
    "--camera-source",
    "--camera-std-xy-m",
    "--camera-std-z-m",
    "--candidate-csv",
    "--candidate-file",
    "--class-map-csv",
    "--classification-min-confidence",
    "--cluster-ranker-cross-sensor-distance-gate-m",
    "--cluster-ranker-cross-sensor-time-window-s",
    "--cluster-ranker-image-evidence-csv",
    "--cluster-ranker-merged-candidates-csv",
    "--cluster-ranker-model-json",
    "--cluster-ranker-previous-states-csv",
    "--cluster-ranker-score-features-csv",
    "--cluster-ranker-scored-candidates-csv",
    "--complete-results-to-truth-csv",
    "--complete-results-to-truth-file",
    "--completed-results-csv",
    "--completed-results-diagnostics-csv",
    "--completed-ug2-codabench-zip",
    "--completion-extrapolation",
    "--completion-max-interpolation-gap-s",
    "--dynamic-background-min-frame-fraction",
    "--dynamic-background-min-frames",
    "--dynamic-background-neighbor-radius-voxels",
    "--dynamic-background-voxel-size-m",
    "--evaluate-results-csv",
    "--evaluate-results-zip",
    "--evaluate-submission-csv",
    "--evaluate-truth-csv",
    "--evaluate-truth-file",
    "--evaluation-class-map-csv",
    "--evaluation-class-map-file",
    "--evaluation-json",
    "--evaluation-max-time-delta-s",
    "--evaluation-protocol",
    "--evaluation-rows-csv",
    "--evaluation-timestamp-tolerance-s",
    "--inferred-class-map-csv",
    "--inspect-root",
    "--layout-report-csv",
    "--layout-report-json",
    "--min-cluster-points",
    "--mot-max-association-distance-m",
    "--mot-max-track-age-s",
    "--native-ros-auto-topic-map-dir",
    "--native-ros-extract-output-dir",
    "--normalize-ug2-official-submission",
    "--normalized-ug2-official-codabench-zip",
    "--normalized-ug2-official-results-csv",
    "--official-normalization-json",
    "--official-upload-manifest-json",
    "--official-upload-manifest-verification-json",
    "--official-validation-json",
    "--official-validation-rows-csv",
    "--official-validation-template-csv",
    "--official-validation-template-file",
    "--official-validation-timestamp-tolerance-s",
    "--output-dir",
    "--point-cloud-csv",
    "--point-cloud-file",
    "--point-extraction-mode",
    "--point-source",
    "--radar-angle-unit",
    "--radar-azimuth-convention",
    "--radar-polar-angle-std-deg",
    "--radar-polar-csv",
    "--radar-polar-file",
    "--radar-polar-range-std-m",
    "--radar-polar-source",
    "--radar-polar-z-std-m",
    "--rosbag-path",
    "--rosbag-report-json",
    "--secondary-covariance-scale",
    "--selection-confidence-weight",
    "--selection-mobility-weight",
    "--selection-motion-weight",
    "--selection-source-priority-weight",
    "--selection-speed-scale-mps",
    "--sequence-classifier",
    "--sequence-classifier-feature-report",
    "--sequence-classifier-predictions-csv",
    "--sequence-classifier-provenance-json",
    "--sequence-glob",
    "--sequence-root",
    "--sequence-root-archive-extract-dir",
    "--soft-anchor-cap-m",
    "--split-file",
    "--split-name",
    "--submission-csv",
    "--submission-json",
    "--submission-track-id",
    "--submission-zip",
    "--topic-map-base-dir",
    "--topic-map-file",
    "--topic-map-json",
    "--topic-map-template-json",
    "--topic-map-template-mode",
    "--tracker-mode",
    "--trajectory-completion-max-gap-s",
    "--trajectory-completion-mode",
    "--trajectory-outlier-replacement",
    "--trajectory-outlier-replacement-max-gap-s",
    "--trajectory-smoothing-blend",
    "--trajectory-smoothing-lag-s",
    "--trajectory-speed-gate-mps",
    "--truth-csv",
    "--truth-file",
    "--ug2-class-map-csv",
    "--ug2-class-map-file",
    "--ug2-class-name",
    "--ug2-codabench-zip",
    "--ug2-official-classification",
    "--ug2-official-codabench-zip",
    "--ug2-official-invalid-row-policy",
    "--ug2-official-results-csv",
    "--ug2-official-timestamp-source",
    "--ug2-results-csv",
    "--validate-ug2-official-codabench-zip",
    "--verify-official-upload-manifest",
    "--voxel-size-m",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if _has_explicit_sequence_root(args):
        return track_main(args)

    sequence_root_index = _sequence_root_index(args)
    if sequence_root_index is None:
        if any(arg in _HELP_FLAGS for arg in args):
            return track_main(args)
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
    return "--sequence-root" in args or any(arg.startswith("--sequence-root=") for arg in args)


def _sequence_root_index(args: list[str]) -> int | None:
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            return index + 1 if index + 1 < len(args) else None
        if arg in _VALUE_FLAGS:
            skip_next = True
            continue
        if _is_value_flag_assignment(arg):
            continue
        if arg.startswith("-"):
            continue
        return index
    return None


def _is_value_flag_assignment(arg: str) -> bool:
    if "=" not in arg:
        return False
    flag, _value = arg.split("=", 1)
    return flag in _VALUE_FLAGS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
