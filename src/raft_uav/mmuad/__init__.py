"""Experimental CVPR UG2+ / MMUAD tracking adapters."""

from raft_uav.mmuad.calibration import CalibrationSet, RigidTransform, SensorCalibration
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame
from raft_uav.mmuad.submission import estimates_to_submission_frame
from raft_uav.mmuad.tracker import TrackerConfig, TrackerOutput, run_mmuad_tracker

__all__ = [
    "CalibrationSet",
    "CandidateFrame",
    "RigidTransform",
    "SensorCalibration",
    "TrackerConfig",
    "TrackerOutput",
    "TruthFrame",
    "estimates_to_submission_frame",
    "run_mmuad_tracker",
]
