"""Experimental CVPR UG2+ / MMUAD tracking adapters."""

from raft_uav.mmuad.calibration import CalibrationSet, RigidTransform, SensorCalibration
from raft_uav.mmuad.mot import MultiObjectTrackerConfig, run_mmuad_multi_object_tracker
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame
from raft_uav.mmuad.evaluate import evaluate_submission_csv
from raft_uav.mmuad.inspect import inspect_sequence_root
from raft_uav.mmuad.submission import estimates_to_submission_frame
from raft_uav.mmuad import _submission_validation_guard as _submission_validation_guard
from raft_uav.mmuad.submission import validate_official_track5_submission
from raft_uav.mmuad.tracker import TrackerConfig, TrackerOutput, run_mmuad_tracker

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
    "run_mmuad_multi_object_tracker",
    "run_mmuad_tracker",
    "validate_official_track5_submission",
]
