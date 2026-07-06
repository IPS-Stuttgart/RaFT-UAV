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


def _install_schema_timestamp_guard() -> None:
    try:
        import pandas as _pd

        from raft_uav.mmuad import schema as _schema
    except Exception:
        return

    original = _schema.normalize_time_column_aliases

    def _normalize_time_column_aliases(frame, *, target: str = "time_s"):
        parsed_target = None
        if target in frame.columns:
            parsed_target = _schema._seconds_or_stamp_dict_series(frame[target])
        out = original(frame, target=target)
        if target in out.columns and parsed_target is not None and parsed_target.notna().any():
            out = out.copy()
            alias_or_numeric = _pd.to_numeric(out[target], errors="coerce")
            out[target] = parsed_target.fillna(alias_or_numeric)
        return out

    _schema.normalize_time_column_aliases = _normalize_time_column_aliases


def _install_image_row_guard() -> None:
    try:
        import pandas as _pd

        from raft_uav.mmuad import image_evidence as _image_evidence
    except Exception:
        return

    parser = getattr(_image_evidence, "_time" + "stamp_from_filename", None)
    if parser is None:
        return

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


def _install_candidate_reservoir_topk_guard() -> None:
    try:
        import argparse as _argparse
        import pandas as _pd
        from pathlib import Path as _Path

        from raft_uav.mmuad import candidate_reservoir as _candidate_reservoir
        from raft_uav.mmuad.schema import normalize_truth_columns as _normalize_truth_columns
    except Exception:
        return

    original_build_candidate_reservoir = _candidate_reservoir.build_candidate_reservoir

    def _build_candidate_reservoir_with_source_default(candidates, *args, **kwargs):
        rows = _pd.DataFrame(candidates).copy()
        if not rows.empty and "source" not in rows.columns:
            rows["source"] = "unknown"
        return original_build_candidate_reservoir(rows, *args, **kwargs)

    _candidate_reservoir.build_candidate_reservoir = _build_candidate_reservoir_with_source_default

    default_top_k = (1, 3, 5, 10, 20)

    def _main(argv: list[str] | None = None) -> int:
        parser = _argparse.ArgumentParser(
            prog="raft-uav-mmuad-candidate-reservoir",
            description="build branch-preserving MMUAD candidate reservoirs",
        )
        parser.add_argument(
            "--candidate",
            action="append",
            default=[],
            help="candidate CSV as BRANCH=path; may be repeated",
        )
        parser.add_argument(
            "--candidate-csv",
            action="append",
            default=[],
            help="alias for --candidate",
        )
        parser.add_argument("--output-csv", type=_Path, required=True)
        parser.add_argument("--summary-json", type=_Path)
        parser.add_argument("--truth-csv", type=_Path)
        parser.add_argument("--oracle-frame-csv", type=_Path)
        parser.add_argument("--oracle-summary-csv", type=_Path)
        parser.add_argument("--oracle-by-sequence-csv", type=_Path)
        parser.add_argument("--global-top-n", type=int, default=20)
        parser.add_argument("--per-source-top-n", type=int, default=3)
        parser.add_argument("--per-branch-top-n", type=int, default=3)
        parser.add_argument("--top-per-source", type=int)
        parser.add_argument("--top-per-branch", type=int)
        parser.add_argument("--max-candidates-per-frame", type=int, default=40)
        parser.add_argument("--score-column", default="ranker_score")
        parser.add_argument("--fallback-score-column", default="confidence")
        parser.add_argument("--score-floor-quantile", type=float)
        parser.add_argument(
            "--cap-reason-bonus",
            type=float,
            default=0.0,
            help="bonus added during final frame cap for each independent reservoir selection reason",
        )
        parser.add_argument("--top-k", type=int, action="append", default=None)
        parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
        args = parser.parse_args(argv)

        top_k_values = tuple(args.top_k) if args.top_k is not None else default_top_k
        candidate_specs = [*args.candidate, *args.candidate_csv]
        candidates = _candidate_reservoir._load_candidate_specs(list(candidate_specs))
        per_source_top_n = args.per_source_top_n if args.top_per_source is None else args.top_per_source
        per_branch_top_n = args.per_branch_top_n if args.top_per_branch is None else args.top_per_branch
        reservoir = _candidate_reservoir.build_candidate_reservoir(
            candidates,
            config=_candidate_reservoir.ReservoirConfig(
                global_top_n=args.global_top_n,
                per_source_top_n=per_source_top_n,
                per_branch_top_n=per_branch_top_n,
                max_candidates_per_frame=args.max_candidates_per_frame,
                score_column=args.score_column,
                fallback_score_column=args.fallback_score_column,
                score_floor_quantile=args.score_floor_quantile,
                cap_reason_bonus=args.cap_reason_bonus,
            ),
        )
        _candidate_reservoir.write_reservoir_outputs(
            reservoir,
            output_csv=args.output_csv,
            summary_json=args.summary_json,
            input_candidates=candidates,
        )
        print("mmuad_candidate_reservoir=ok")
        print(f"candidate_rows={len(candidates)}")
        print(f"reservoir_rows={len(reservoir)}")
        print(f"output_csv={args.output_csv}")

        if args.truth_csv is not None:
            truth = _normalize_truth_columns(_pd.read_csv(args.truth_csv))
            frame_rows, pooled, by_sequence = _candidate_reservoir.build_oracle_recall_tables(
                reservoir,
                truth,
                top_k_values=top_k_values,
                max_truth_time_delta_s=args.max_truth_time_delta_s,
            )
            if args.oracle_frame_csv is not None:
                args.oracle_frame_csv.parent.mkdir(parents=True, exist_ok=True)
                frame_rows.to_csv(args.oracle_frame_csv, index=False)
            if args.oracle_summary_csv is not None:
                args.oracle_summary_csv.parent.mkdir(parents=True, exist_ok=True)
                pooled.to_csv(args.oracle_summary_csv, index=False)
            if args.oracle_by_sequence_csv is not None:
                args.oracle_by_sequence_csv.parent.mkdir(parents=True, exist_ok=True)
                by_sequence.to_csv(args.oracle_by_sequence_csv, index=False)
            print(f"oracle_frames={len(frame_rows)}")
            if not pooled.empty:
                print(f"oracle_all_mse={pooled.loc[0, 'oracle_all_3d_m_mse']}")
        return 0

    _candidate_reservoir.main = _main


def _install_submission_eval_track_id_guard() -> None:
    try:
        from raft_uav.mmuad import evaluate as _evaluate
    except Exception:
        return

    def _should_restrict_to_track_id(
        truth_track_ids: set[str],
        submitted_track_ids: set[str],
    ) -> bool:
        if not truth_track_ids or not submitted_track_ids:
            return False
        if truth_track_ids.intersection(submitted_track_ids):
            return True
        return len(truth_track_ids) > 1

    _evaluate._should_restrict_to_track_id = _should_restrict_to_track_id


def _install_track5_scorecard_bool_guard() -> None:
    try:
        import numpy as _np
        import pandas as _pd

        from raft_uav.mmuad import track5_scorecard as _track5_scorecard
    except Exception:
        return

    def _bool_series(values):
        if values is None:
            return _pd.Series(dtype=bool)
        series = _pd.Series(values)
        if series.empty:
            return _pd.Series(dtype=bool)
        if _pd.api.types.is_bool_dtype(series.dtype):
            return series.fillna(False).astype(bool)

        numeric = _pd.to_numeric(series, errors="coerce")
        numeric_values = numeric.to_numpy(dtype=float)
        numeric_truthy = _pd.Series(
            _np.isfinite(numeric_values) & (numeric_values != 0.0),
            index=series.index,
        )
        text = series.where(series.notna(), "").astype(str).str.strip().str.lower()
        truthy_text = text.isin({"1", "1.0", "true", "t", "yes", "y"})
        falsy_text = text.isin(
            {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "<na>", "nat"}
        )
        return truthy_text | (numeric_truthy & ~falsy_text)

    _track5_scorecard._bool_series = _bool_series


_install_schema_timestamp_guard()
_install_image_row_guard()
_install_candidate_pool_compare_cli_guard()
_install_temporal_consensus_train_cv_cli_guard()
_install_candidate_reservoir_topk_guard()
_install_submission_eval_track_id_guard()
_install_track5_scorecard_bool_guard()


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
