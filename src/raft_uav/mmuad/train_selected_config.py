"""Train-only MMUAD pipeline hyperparameter selection."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.source_calibration import SOURCE_CALIBRATION_MODES


TRAIN_SELECTED_CONFIG_SCHEMA = "raft-uav-mmuad-train-selected-config-v2"

CONFIG_FIELDS = (
    "source_calibration_mode",
    "source_translation_alpha",
    "point_extraction_mode",
    "ranker_model_type",
    "ranker_target_column",
    "mmuad_selection_mode",
    "viterbi_motion_weight",
    "viterbi_ranker_weight",
    "viterbi_source_switch_penalty",
    "viterbi_max_speed_mps",
    "viterbi_gap_penalty",
    "candidate_reservoir_global_top_n",
    "candidate_reservoir_per_source_top_n",
    "candidate_reservoir_per_branch_top_n",
    "candidate_reservoir_max_candidates_per_frame",
    "candidate_reservoir_score_column",
    "candidate_reservoir_score_floor_quantile",
    "candidate_reservoir_cap_reason_bonus",
    "candidate_mixture_score_column",
    "candidate_mixture_sigma_column",
    "candidate_mixture_score_weight",
    "candidate_mixture_temperature",
    "candidate_mixture_sigma_log_weight",
    "candidate_mixture_huber_delta",
    "candidate_mixture_smoothness_weight",
    "candidate_mixture_uniform_weight_floor",
    "candidate_mixture_branch_balance",
    "candidate_mixture_source_balance",
    "candidate_mixture_responsibility_floor",
    "candidate_mixture_sigma_min_m",
    "candidate_mixture_sigma_max_m",
    "smoothing_mode",
    "smoothing_speed_gate_mps",
    "smoothing_blend",
    "classifier_method",
    "image_nonimage_fusion_weight",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "source_calibration_mode": "identity",
    "source_translation_alpha": 1.0,
    "point_extraction_mode": "static",
    "ranker_model_type": "random-forest-classifier",
    "ranker_target_column": "good_cluster_5m",
    "mmuad_selection_mode": "greedy",
    "viterbi_motion_weight": 1.0,
    "viterbi_ranker_weight": 1.0,
    "viterbi_source_switch_penalty": 0.0,
    "viterbi_max_speed_mps": 60.0,
    "viterbi_gap_penalty": 0.0,
    "candidate_reservoir_global_top_n": 20,
    "candidate_reservoir_per_source_top_n": 3,
    "candidate_reservoir_per_branch_top_n": 3,
    "candidate_reservoir_max_candidates_per_frame": 40,
    "candidate_reservoir_score_column": "candidate_reservoir_grid_score",
    "candidate_reservoir_score_floor_quantile": None,
    "candidate_reservoir_cap_reason_bonus": 0.0,
    "candidate_mixture_score_column": "candidate_reservoir_score",
    "candidate_mixture_sigma_column": "predicted_sigma_m",
    "candidate_mixture_score_weight": 1.0,
    "candidate_mixture_temperature": 1.0,
    "candidate_mixture_sigma_log_weight": 3.0,
    "candidate_mixture_huber_delta": 1.0,
    "candidate_mixture_smoothness_weight": 7200.0,
    "candidate_mixture_uniform_weight_floor": 0.0,
    "candidate_mixture_branch_balance": 0.0,
    "candidate_mixture_source_balance": 0.0,
    "candidate_mixture_responsibility_floor": 0.0,
    "candidate_mixture_sigma_min_m": 1.0,
    "candidate_mixture_sigma_max_m": 30.0,
    "smoothing_mode": "none",
    "smoothing_speed_gate_mps": 0.0,
    "smoothing_blend": 1.0,
    "classifier_method": "random-forest",
    "image_nonimage_fusion_weight": 0.0,
}

LOWER_IS_BETTER_COLUMNS = (
    "train_cv_pose_mse_loss_m2",
    "loso_pose_mse_loss_m2",
    "pose_mse_loss_m2",
    "pose_mse",
    "mse",
    "weighted_mse",
    "cv_mse",
    "source_translation_alpha_cv_mse",
    "after_mean_m",
)
HIGHER_IS_BETTER_COLUMNS = (
    "train_cv_classification_accuracy",
    "loso_classification_accuracy",
    "classification_accuracy",
    "uav_type_accuracy",
    "timestamp_accuracy",
    "sequence_accuracy",
    "accuracy",
)
POINT_EXTRACTION_MODES = (
    "static",
    "dynamic",
    "dynamic-object",
    "static-plus-dynamic",
    "static_dynamic_union",
)


def load_train_selected_config(path: Path) -> dict[str, Any]:
    """Load and validate a frozen MMUAD train-selected config."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a config object")
    out = dict(DEFAULT_CONFIG)
    for field in CONFIG_FIELDS:
        if field in config and config[field] is not None:
            out[field] = config[field]
    return validate_train_selected_config(out)


def validate_train_selected_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized train-selected config or raise ``ValueError``."""

    out = dict(DEFAULT_CONFIG)
    out.update({key: value for key, value in config.items() if key in CONFIG_FIELDS})
    out["source_calibration_mode"] = _choice(
        out["source_calibration_mode"],
        SOURCE_CALIBRATION_MODES,
        "source_calibration_mode",
    )
    out["source_translation_alpha"] = float(np.clip(_float(out["source_translation_alpha"]), 0.0, 1.0))
    out["point_extraction_mode"] = _choice(
        out["point_extraction_mode"],
        POINT_EXTRACTION_MODES,
        "point_extraction_mode",
    )
    out["mmuad_selection_mode"] = _choice(
        out["mmuad_selection_mode"],
        ("greedy", "viterbi"),
        "mmuad_selection_mode",
    )
    out["smoothing_mode"] = _choice(
        out["smoothing_mode"],
        ("none", "gap-interpolation", "fixed-lag", "constant-velocity", "constant-acceleration"),
        "smoothing_mode",
    )
    for field in (
        "candidate_reservoir_global_top_n",
        "candidate_reservoir_per_source_top_n",
        "candidate_reservoir_per_branch_top_n",
        "candidate_reservoir_max_candidates_per_frame",
    ):
        out[field] = _nonnegative_int(out[field], field)
    for field in (
        "viterbi_motion_weight",
        "viterbi_ranker_weight",
        "viterbi_source_switch_penalty",
        "viterbi_max_speed_mps",
        "viterbi_gap_penalty",
        "candidate_reservoir_cap_reason_bonus",
        "candidate_mixture_score_weight",
        "candidate_mixture_temperature",
        "candidate_mixture_sigma_log_weight",
        "candidate_mixture_huber_delta",
        "candidate_mixture_smoothness_weight",
        "candidate_mixture_uniform_weight_floor",
        "candidate_mixture_branch_balance",
        "candidate_mixture_source_balance",
        "candidate_mixture_responsibility_floor",
        "candidate_mixture_sigma_min_m",
        "candidate_mixture_sigma_max_m",
        "smoothing_speed_gate_mps",
        "smoothing_blend",
        "image_nonimage_fusion_weight",
    ):
        out[field] = _float(out[field])
    out["candidate_reservoir_score_floor_quantile"] = _optional_quantile(
        out["candidate_reservoir_score_floor_quantile"],
        "candidate_reservoir_score_floor_quantile",
    )
    for field in (
        "candidate_mixture_uniform_weight_floor",
        "candidate_mixture_branch_balance",
        "candidate_mixture_source_balance",
        "candidate_mixture_responsibility_floor",
    ):
        if not 0.0 <= float(out[field]) <= 1.0:
            raise ValueError(f"{field} must be within [0, 1]")
    if not 0.0 <= float(out["candidate_mixture_uniform_weight_floor"]) < 1.0:
        raise ValueError("candidate_mixture_uniform_weight_floor must be in [0, 1)")
    if out["candidate_mixture_temperature"] <= 0.0:
        raise ValueError("candidate_mixture_temperature must be positive")
    if out["candidate_mixture_huber_delta"] <= 0.0:
        raise ValueError("candidate_mixture_huber_delta must be positive")
    if out["candidate_mixture_sigma_min_m"] <= 0.0:
        raise ValueError("candidate_mixture_sigma_min_m must be positive")
    if out["candidate_mixture_sigma_min_m"] > out["candidate_mixture_sigma_max_m"]:
        raise ValueError("candidate mixture sigma bounds must satisfy min <= max")
    out["ranker_model_type"] = str(out["ranker_model_type"])
    out["ranker_target_column"] = str(out["ranker_target_column"])
    out["candidate_reservoir_score_column"] = str(out["candidate_reservoir_score_column"])
    out["candidate_mixture_score_column"] = str(out["candidate_mixture_score_column"])
    out["candidate_mixture_sigma_column"] = str(out["candidate_mixture_sigma_column"])
    out["classifier_method"] = str(out["classifier_method"])
    return out


def build_train_selected_config(
    *,
    source_calibration_summary_csv: Path | None = None,
    ranker_summary_csv: Path | None = None,
    reservoir_summary_csv: Path | None = None,
    mixture_summary_csv: Path | None = None,
    viterbi_summary_csv: Path | None = None,
    smoothing_summary_csv: Path | None = None,
    classifier_summary_csv: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select final MMUAD settings from train-only CV/LOSO summaries."""

    config = dict(DEFAULT_CONFIG)
    records: list[dict[str, Any]] = []
    _select_component(
        config,
        records,
        component="source_calibration",
        csv_path=source_calibration_summary_csv,
        mappings={
            "source_calibration_mode": ("source_calibration_mode", "mode"),
            "source_translation_alpha": ("source_translation_alpha", "alpha"),
        },
        metric_columns=LOWER_IS_BETTER_COLUMNS,
        maximize=False,
    )
    _select_component(
        config,
        records,
        component="ranker",
        csv_path=ranker_summary_csv,
        mappings={
            "ranker_model_type": ("ranker_model_type", "model_type"),
            "ranker_target_column": ("ranker_target_column", "target_column"),
            "point_extraction_mode": ("point_extraction_mode", "extraction_mode"),
        },
        metric_columns=LOWER_IS_BETTER_COLUMNS,
        maximize=False,
    )
    _select_component(
        config,
        records,
        component="candidate_reservoir",
        csv_path=reservoir_summary_csv,
        mappings={
            "candidate_reservoir_global_top_n": ("global_top_n", "reservoir_global_top_n"),
            "candidate_reservoir_per_source_top_n": (
                "per_source_top_n",
                "reservoir_per_source_top_n",
            ),
            "candidate_reservoir_per_branch_top_n": (
                "per_branch_top_n",
                "reservoir_per_branch_top_n",
            ),
            "candidate_reservoir_max_candidates_per_frame": (
                "max_candidates_per_frame",
                "reservoir_max_candidates_per_frame",
            ),
            "candidate_reservoir_score_floor_quantile": (
                "score_floor_quantile",
                "reservoir_score_floor_quantile",
            ),
            "candidate_reservoir_cap_reason_bonus": (
                "cap_reason_bonus",
                "reservoir_cap_reason_bonus",
            ),
        },
        metric_columns=LOWER_IS_BETTER_COLUMNS,
        maximize=False,
    )
    _select_component(
        config,
        records,
        component="candidate_mixture",
        csv_path=mixture_summary_csv,
        mappings={
            "candidate_mixture_score_weight": ("score_weight", "mixture_score_weight"),
            "candidate_mixture_temperature": ("temperature", "mixture_temperature"),
            "candidate_mixture_sigma_log_weight": (
                "sigma_log_weight",
                "mixture_sigma_log_weight",
            ),
            "candidate_mixture_huber_delta": ("huber_delta", "loss_scale_m"),
            "candidate_mixture_smoothness_weight": (
                "smoothness_weight",
                "mixture_smoothness_weight",
            ),
            "candidate_mixture_uniform_weight_floor": (
                "uniform_weight_floor",
                "mixture_uniform_weight_floor",
            ),
            "candidate_mixture_branch_balance": ("branch_balance", "mixture_branch_balance"),
            "candidate_mixture_source_balance": ("source_balance", "mixture_source_balance"),
            "candidate_mixture_responsibility_floor": (
                "responsibility_floor",
                "mixture_responsibility_floor",
            ),
            "candidate_mixture_sigma_min_m": ("sigma_min_m", "mixture_sigma_min_m"),
            "candidate_mixture_sigma_max_m": ("sigma_max_m", "mixture_sigma_max_m"),
        },
        metric_columns=LOWER_IS_BETTER_COLUMNS,
        maximize=False,
    )
    _select_component(
        config,
        records,
        component="viterbi",
        csv_path=viterbi_summary_csv,
        mappings={
            "mmuad_selection_mode": ("mmuad_selection_mode", "selection_mode"),
            "viterbi_motion_weight": ("viterbi_motion_weight", "motion_weight"),
            "viterbi_ranker_weight": ("viterbi_ranker_weight", "ranker_weight"),
            "viterbi_source_switch_penalty": (
                "viterbi_source_switch_penalty",
                "source_switch_penalty",
            ),
            "viterbi_max_speed_mps": ("viterbi_max_speed_mps", "max_speed_mps"),
            "viterbi_gap_penalty": ("viterbi_gap_penalty", "gap_penalty"),
        },
        metric_columns=LOWER_IS_BETTER_COLUMNS,
        maximize=False,
        fixed_updates={"mmuad_selection_mode": "viterbi"},
    )
    _select_component(
        config,
        records,
        component="smoothing",
        csv_path=smoothing_summary_csv,
        mappings={
            "smoothing_mode": ("smoothing_mode", "mode"),
            "smoothing_speed_gate_mps": ("smoothing_speed_gate_mps", "speed_gate_mps"),
            "smoothing_blend": ("smoothing_blend", "blend"),
        },
        metric_columns=LOWER_IS_BETTER_COLUMNS,
        maximize=False,
    )
    _select_component(
        config,
        records,
        component="classifier",
        csv_path=classifier_summary_csv,
        mappings={
            "classifier_method": ("classifier_method", "method"),
            "image_nonimage_fusion_weight": (
                "image_nonimage_fusion_weight",
                "fusion_weight",
                "image_weight",
            ),
        },
        metric_columns=HIGHER_IS_BETTER_COLUMNS,
        maximize=True,
    )
    if overrides:
        for field, value in overrides.items():
            if value is not None and field in CONFIG_FIELDS:
                config[field] = value
    return validate_train_selected_config(config), records


def write_train_selected_config(
    config: dict[str, Any],
    *,
    output_json: Path,
    summary_csv: Path,
    selection_records: list[dict[str, Any]] | None = None,
    selection_inputs: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Write the frozen selected-config JSON and one-row summary CSV."""

    config = validate_train_selected_config(config)
    payload = {
        "schema": TRAIN_SELECTED_CONFIG_SCHEMA,
        "protocol": "train_only_hyperparameter_selection_then_single_public_validation_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **config,
        "config": config,
        "selection_inputs": selection_inputs or {},
        "selection_records": selection_records or [],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        **config,
        "selected_config_json": str(output_json),
        "selection_record_count": len(selection_records or []),
    }
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-select-train-config",
        description=(
            "freeze MMUAD pipeline settings from train-only CV/LOSO summaries "
            "for a single public-validation run"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mmuad_train_selected_config"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--source-calibration-summary-csv", type=Path)
    parser.add_argument("--ranker-summary-csv", type=Path)
    parser.add_argument("--reservoir-summary-csv", type=Path)
    parser.add_argument("--mixture-summary-csv", type=Path)
    parser.add_argument("--viterbi-summary-csv", type=Path)
    parser.add_argument("--smoothing-summary-csv", type=Path)
    parser.add_argument("--classifier-summary-csv", type=Path)
    parser.add_argument("--source-calibration-mode", choices=SOURCE_CALIBRATION_MODES)
    parser.add_argument("--source-translation-alpha", type=float)
    parser.add_argument("--point-extraction-mode", choices=POINT_EXTRACTION_MODES)
    parser.add_argument("--ranker-model-type")
    parser.add_argument("--ranker-target-column")
    parser.add_argument("--mmuad-selection-mode", choices=("greedy", "viterbi"))
    parser.add_argument("--viterbi-motion-weight", type=float)
    parser.add_argument("--viterbi-ranker-weight", type=float)
    parser.add_argument("--viterbi-source-switch-penalty", type=float)
    parser.add_argument("--viterbi-max-speed-mps", type=float)
    parser.add_argument("--viterbi-gap-penalty", type=float)
    parser.add_argument("--candidate-reservoir-global-top-n", type=int)
    parser.add_argument("--candidate-reservoir-per-source-top-n", type=int)
    parser.add_argument("--candidate-reservoir-per-branch-top-n", type=int)
    parser.add_argument("--candidate-reservoir-max-candidates-per-frame", type=int)
    parser.add_argument("--candidate-reservoir-score-column")
    parser.add_argument("--candidate-reservoir-score-floor-quantile", type=float)
    parser.add_argument("--candidate-reservoir-cap-reason-bonus", type=float)
    parser.add_argument("--candidate-mixture-score-column")
    parser.add_argument("--candidate-mixture-sigma-column")
    parser.add_argument("--candidate-mixture-score-weight", type=float)
    parser.add_argument("--candidate-mixture-temperature", type=float)
    parser.add_argument("--candidate-mixture-sigma-log-weight", type=float)
    parser.add_argument("--candidate-mixture-huber-delta", type=float)
    parser.add_argument("--candidate-mixture-smoothness-weight", type=float)
    parser.add_argument("--candidate-mixture-uniform-weight-floor", type=float)
    parser.add_argument("--candidate-mixture-branch-balance", type=float)
    parser.add_argument("--candidate-mixture-source-balance", type=float)
    parser.add_argument("--candidate-mixture-responsibility-floor", type=float)
    parser.add_argument("--candidate-mixture-sigma-min-m", type=float)
    parser.add_argument("--candidate-mixture-sigma-max-m", type=float)
    parser.add_argument(
        "--smoothing-mode",
        choices=("none", "gap-interpolation", "fixed-lag", "constant-velocity", "constant-acceleration"),
    )
    parser.add_argument("--smoothing-speed-gate-mps", type=float)
    parser.add_argument("--smoothing-blend", type=float)
    parser.add_argument("--classifier-method")
    parser.add_argument("--image-nonimage-fusion-weight", type=float)
    args = parser.parse_args(argv)

    output_json = args.output_json or args.output_dir / "mmuad_train_selected_config.json"
    summary_csv = args.summary_csv or args.output_dir / "mmuad_train_selected_config_summary.csv"
    inputs = {
        "source_calibration_summary_csv": _path_or_none(args.source_calibration_summary_csv),
        "ranker_summary_csv": _path_or_none(args.ranker_summary_csv),
        "reservoir_summary_csv": _path_or_none(args.reservoir_summary_csv),
        "mixture_summary_csv": _path_or_none(args.mixture_summary_csv),
        "viterbi_summary_csv": _path_or_none(args.viterbi_summary_csv),
        "smoothing_summary_csv": _path_or_none(args.smoothing_summary_csv),
        "classifier_summary_csv": _path_or_none(args.classifier_summary_csv),
    }
    config, records = build_train_selected_config(
        source_calibration_summary_csv=args.source_calibration_summary_csv,
        ranker_summary_csv=args.ranker_summary_csv,
        reservoir_summary_csv=args.reservoir_summary_csv,
        mixture_summary_csv=args.mixture_summary_csv,
        viterbi_summary_csv=args.viterbi_summary_csv,
        smoothing_summary_csv=args.smoothing_summary_csv,
        classifier_summary_csv=args.classifier_summary_csv,
        overrides={field: getattr(args, field) for field in CONFIG_FIELDS if hasattr(args, field)},
    )
    write_train_selected_config(
        config,
        output_json=output_json,
        summary_csv=summary_csv,
        selection_records=records,
        selection_inputs=inputs,
    )
    print("mmuad_train_selected_config=ok")
    print(f"selected_config_json={output_json}")
    print(f"selected_config_summary_csv={summary_csv}")
    return 0


def _select_component(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    component: str,
    csv_path: Path | None,
    mappings: dict[str, tuple[str, ...]],
    metric_columns: tuple[str, ...],
    maximize: bool,
    fixed_updates: dict[str, Any] | None = None,
) -> None:
    if csv_path is None:
        return
    row, metric_column, metric_value, row_index = _best_row(csv_path, metric_columns, maximize=maximize)
    updates = dict(fixed_updates or {})
    for field, columns in mappings.items():
        value = _first_present(row, columns)
        if value is not None and not _is_nan(value):
            updates[field] = value
    config.update(updates)
    records.append(
        {
            "component": component,
            "input_csv": str(csv_path),
            "selected_row_index": int(row_index),
            "selection_metric": metric_column,
            "selection_metric_value": metric_value,
            **{key: _jsonable(value) for key, value in updates.items()},
        }
    )


def _best_row(
    csv_path: Path,
    metric_columns: tuple[str, ...],
    *,
    maximize: bool,
) -> tuple[pd.Series, str | None, float | None, int]:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"{csv_path} has no rows")
    for column in metric_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        index = int(finite.idxmax() if maximize else finite.idxmin())
        return frame.loc[index], column, float(values.loc[index]), index
    return frame.iloc[0], None, None, int(frame.index[0])


def _first_present(row: pd.Series, columns: tuple[str, ...]) -> Any:
    for column in columns:
        if column in row.index:
            return row[column]
    return None


def _choice(value: Any, choices: tuple[str, ...], field: str) -> str:
    normalized = str(value).strip()
    if normalized not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"{field} must be one of {allowed}; got {value!r}")
    return normalized


def _float(value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"expected finite float, got {value!r}")
    return number


def _nonnegative_int(value: Any, field: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _optional_quantile(value: Any, field: str) -> float | None:
    if value is None or _is_nan(value):
        return None
    number = _float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return number


def _is_nan(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _path_or_none(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
