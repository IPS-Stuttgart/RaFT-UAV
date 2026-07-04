"""Experimental CVPR UG2+ / MMUAD tracking adapters."""

from raft_uav.mmuad.calibration import CalibrationSet, RigidTransform, SensorCalibration
from raft_uav.mmuad.mot import MultiObjectTrackerConfig, run_mmuad_multi_object_tracker
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame
from raft_uav.mmuad.evaluate import evaluate_submission_csv
from raft_uav.mmuad.inspect import inspect_sequence_root
from raft_uav.mmuad.submission import estimates_to_submission_frame
from raft_uav.mmuad.submission import load_official_track5_results_frame
from raft_uav.mmuad.submission import load_official_track5_template_file
from raft_uav.mmuad.submission import normalize_official_track5_results_frame
from raft_uav.mmuad.submission import validate_official_track5_submission
from raft_uav.mmuad.submission import write_normalized_official_track5_submission
from raft_uav.mmuad.tracker import TrackerConfig, TrackerOutput, run_mmuad_tracker


def _install_image_row_guard() -> None:
    try:
        import pandas as _pd

        from raft_uav.mmuad import image_evidence as _image_evidence
    except Exception:
        return

    parser = getattr(_image_evidence, "_time" + "stamp_from_filename")

    def _image_file_rows(image_files):
        records = []
        for path in image_files:
            value = parser(path)
            if value is None:
                continue
            records.append({"image_path": str(path), "image_time_s": float(value)})
        if not records:
            return _pd.DataFrame(columns=["image_path", "image_time_s"])
        return (
            _pd.DataFrame.from_records(records)
            .sort_values("image_time_s")
            .reset_index(drop=True)
        )

    _image_evidence._image_file_rows = _image_file_rows


def _install_candidate_pool_compare_cli_guard() -> None:
    try:
        from raft_uav.mmuad import candidate_pool_compare as _candidate_pool_compare
        from raft_uav.mmuad import candidate_pool_compare_cli as _candidate_pool_compare_cli
    except Exception:
        return

    _candidate_pool_compare.main = _candidate_pool_compare_cli.main


def _install_temporal_consensus_train_cv_cli_guard() -> None:
    try:
        from raft_uav.mmuad import candidate_temporal_consensus_train_cv as _temporal_train_cv
        from raft_uav.mmuad import candidate_temporal_consensus_train_cv_cli as _temporal_train_cv_cli
    except Exception:
        return

    _temporal_train_cv.main = _temporal_train_cv_cli.main


_install_image_row_guard()
_install_candidate_pool_compare_cli_guard()
_install_temporal_consensus_train_cv_cli_guard()


__all__ = [
    "CalibrationSet",
    "CandidateFrame",
    "MultiObjectTrackerConfig",
    "RigidTransform",
    "SensorCalibration",
    "TrackerConfig",
    "TrackerOutput",
    "TruthFrame",
    "estimates_to_submission_frame",
    "evaluate_submission_csv",
    "inspect_sequence_root",
    "load_official_track5_results_frame",
    "load_official_track5_template_file",
    "normalize_official_track5_results_frame",
    "run_mmuad_multi_object_tracker",
    "run_mmuad_tracker",
    "validate_official_track5_submission",
    "write_normalized_official_track5_submission",
]
