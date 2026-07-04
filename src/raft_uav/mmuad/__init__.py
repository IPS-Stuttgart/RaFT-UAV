"""Experimental CVPR UG2+ / MMUAD tracking adapters."""

from raft_uav.mmuad.calibration import CalibrationSet, RigidTransform, SensorCalibration
from raft_uav.mmuad.mot import MultiObjectTrackerConfig, run_mmuad_multi_object_tracker
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame
from raft_uav.mmuad.evaluate import evaluate_submission_csv
from raft_uav.mmuad.inspect import inspect_sequence_root
from raft_uav.mmuad import submission as _submission
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



def _install_track5_validation_class_domain_guard() -> None:
    original = _submission._official_track5_row_diagnostics
    allowed = set(_submission.OFFICIAL_TRACK5_CLASS_IDS)

    def _official_track5_row_diagnostics(frame):
        diagnostics, normalized = original(frame)
        if normalized.empty or "classification" not in normalized.columns:
            return diagnostics, normalized
        invalid = ~normalized["classification"].isin(allowed)
        if not invalid.any():
            return diagnostics, normalized
        invalid_indices = normalized.loc[invalid, "row_index"].astype(int).tolist()
        mask = diagnostics["row_index"].isin(invalid_indices) & diagnostics["row_type"].eq("prediction")
        diagnostics = diagnostics.copy()
        diagnostics.loc[mask, "status"] = "invalid_classification"
        diagnostics.loc[mask, "reason"] = (
            "official MMUAD Classification values must be one of {0, 1, 2, 3}"
        )
        normalized = normalized.loc[~invalid].reset_index(drop=True)
        return diagnostics, normalized

    _submission._official_track5_row_diagnostics = _official_track5_row_diagnostics


_install_image_row_guard()
_install_track5_validation_class_domain_guard()

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
