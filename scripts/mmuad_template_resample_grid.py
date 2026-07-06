#!/usr/bin/env python
"""Sweep MMUAD Track 5 template-resampling packaging settings.

The Codabench Track 5 upload must contain exactly one row for each official
``Sequence``/``Timestamp``. The single-run template-resample helper already
creates upload-ready artifacts; this script runs a small truth-free or
truth-scored grid over resampling method, interpolation-gap fallback, and
classification preservation policy so the final packaging choice can be frozen
from artifacts instead of hand-edited commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.submission import (  # noqa: E402
    load_official_track5_template_file,
    load_sequence_class_map,
)
from raft_uav.mmuad.track5_scorecard import (  # noqa: E402
    build_track5_scorecard,
    scorecard_summary_frame,
    write_track5_scorecard,
)
from raft_uav.mmuad.track5_template_resample import (  # noqa: E402
    CLASSIFICATION_POLICIES,
    RESAMPLE_METHODS,
    write_track5_template_resample_outputs,
)

SUMMARY_CSV = "mmuad_template_resample_grid_summary.csv"
SUMMARY_JSON = "mmuad_template_resample_grid_summary.json"


def run_template_resample_grid(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    template_path: Path | None = None,
    truth_path: Path | None = None,
    class_map_path: Path | None = None,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    resample_methods: tuple[str, ...] = ("linear", "nearest"),
    max_interpolation_gaps_s: tuple[float | None, ...] = (None,),
    classification_policies: tuple[str, ...] = ("sequence-mode",),
    max_nearest_time_delta_s: float | None = None,
    require_leaderboard_ready: bool = False,
) -> pd.DataFrame:
    """Run the template-resample grid and return a sorted summary table."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    methods = _normalized_choices(resample_methods, RESAMPLE_METHODS, "resample_method")
    class_policies = _normalized_choices(
        classification_policies,
        CLASSIFICATION_POLICIES,
        "classification_policy",
    )
    rows: list[dict[str, Any]] = []
    for method in methods:
        for gap_s in max_interpolation_gaps_s:
            for class_policy in class_policies:
                variant = _variant_label(method, gap_s, class_policy)
                variant_dir = output / variant
                paths = write_track5_template_resample_outputs(
                    estimates=estimates,
                    template=template,
                    output_dir=variant_dir,
                    class_map=class_map,
                    default_classification=default_classification,
                    max_nearest_time_delta_s=max_nearest_time_delta_s,
                    resample_method=method,  # type: ignore[arg-type]
                    max_interpolation_gap_s=gap_s,
                    classification_policy=class_policy,  # type: ignore[arg-type]
                )
                validation_summary = _read_json(paths["validation_json"])
                manifest = _read_json(paths["manifest_json"])
                record = _base_summary_record(
                    variant=variant,
                    method=method,
                    gap_s=gap_s,
                    class_policy=class_policy,
                    paths=paths,
                    manifest=manifest,
                    validation_summary=validation_summary,
                )
                if truth_path is not None and template_path is not None:
                    record.update(
                        _write_variant_scorecard(
                            variant_dir=variant_dir,
                            results_path=paths["official_zip"],
                            truth_path=truth_path,
                            template_path=template_path,
                            class_map_path=class_map_path,
                        ),
                    )
                rows.append(record)
    summary = _sort_summary(pd.DataFrame.from_records(rows))
    summary.to_csv(output / SUMMARY_CSV, index=False)
    (output / SUMMARY_JSON).write_text(
        json.dumps({"rows": _jsonable(summary.to_dict(orient="records"))}, indent=2),
        encoding="utf-8",
    )
    if require_leaderboard_ready and not _has_ready_row(summary):
        raise SystemExit("no template-resample grid row is leaderboard-ready")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates-csv", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth", type=Path, help="optional public-validation truth for scoring")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument(
        "--resample-method",
        action="append",
        default=["linear,nearest"],
        help="method list; comma-separated and repeatable",
    )
    parser.add_argument(
        "--max-interpolation-gap-s",
        action="append",
        default=["none"],
        help="gap fallback list in seconds; use none/off/null for no fallback",
    )
    parser.add_argument(
        "--classification-policy",
        action="append",
        default=["sequence-mode"],
        help="classification policy list; comma-separated and repeatable",
    )
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    estimates = pd.read_csv(args.estimates_csv)
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    summary = run_template_resample_grid(
        estimates=estimates,
        template=template,
        output_dir=args.output_dir,
        template_path=args.template,
        truth_path=args.truth,
        class_map_path=args.class_map,
        class_map=class_map,
        default_classification=args.default_classification,
        resample_methods=_parse_text_list(args.resample_method),
        max_interpolation_gaps_s=_parse_optional_float_list(args.max_interpolation_gap_s),
        classification_policies=_parse_text_list(args.classification_policy),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        require_leaderboard_ready=args.require_leaderboard_ready,
    )
    print("mmuad_template_resample_grid=ok")
    print(f"summary_csv={args.output_dir / SUMMARY_CSV}")
    print(f"summary_json={args.output_dir / SUMMARY_JSON}")
    if not summary.empty:
        best = summary.iloc[0]
        print(f"best_variant={best['variant']}")
        print(f"best_codabench_upload_ready={best.get('codabench_upload_ready')}")
        if pd.notna(best.get("pose_mse_loss_m2")):
            print(f"best_pose_mse_loss_m2={best.get('pose_mse_loss_m2')}")
    return 0


def _write_variant_scorecard(
    *,
    variant_dir: Path,
    results_path: Path,
    truth_path: Path,
    template_path: Path,
    class_map_path: Path | None,
) -> dict[str, Any]:
    scorecard = build_track5_scorecard(
        results_path=results_path,
        truth_path=truth_path,
        template_path=template_path,
        class_map_path=class_map_path,
        require_zip=True,
    )
    scorecard_paths = write_track5_scorecard(
        scorecard,
        summary_json=variant_dir / "track5_scorecard.json",
        summary_csv=variant_dir / "track5_scorecard.csv",
        validation_rows_csv=variant_dir / "track5_scorecard_validation_rows.csv",
        public_evaluation_rows_csv=variant_dir / "track5_scorecard_public_rows.csv",
        nearest_time_rows_csv=variant_dir / "track5_scorecard_nearest_rows.csv",
        pose_by_sequence_csv=variant_dir / "mmuad_pose_by_sequence.csv",
    )
    values = _scorecard_summary_values(scorecard.summary)
    values.update({f"scorecard_{key}": value for key, value in scorecard_paths.items()})
    return values


def _base_summary_record(
    *,
    variant: str,
    method: str,
    gap_s: float | None,
    class_policy: str,
    paths: dict[str, Path],
    manifest: dict[str, Any],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant,
        "resample_method": method,
        "max_interpolation_gap_s": gap_s,
        "classification_policy": class_policy,
        "leaderboard_ready": validation_summary.get("leaderboard_ready"),
        "codabench_upload_ready": validation_summary.get("codabench_upload_ready"),
        "leaderboard_blocking_reasons": ";".join(
            str(item) for item in validation_summary.get("leaderboard_blocking_reasons", [])
        ),
        "row_count": manifest.get("row_count"),
        "template_row_count": manifest.get("template_row_count"),
        "valid_resampled_rows": manifest.get("valid_resampled_rows"),
        "invalid_resampled_rows": manifest.get("invalid_resampled_rows"),
        "extrapolated_rows": manifest.get("extrapolated_rows"),
        "large_gap_fallback_rows": manifest.get("large_gap_fallback_rows"),
        "resampled_classification_rows": manifest.get("resampled_classification_rows"),
        "official_zip": str(paths["official_zip"]),
        "official_results_csv": str(paths["official_results_csv"]),
        "manifest_json": str(paths["manifest_json"]),
        "validation_json": str(paths["validation_json"]),
    }


def _scorecard_summary_values(summary: dict[str, Any]) -> dict[str, Any]:
    frame = scorecard_summary_frame(summary)
    if frame.empty:
        return {}
    row = frame.iloc[0].to_dict()
    return {
        "pose_mse_loss_m2": row.get("pose_mse_loss_m2"),
        "public_rmse_3d_m": row.get("public_rmse_3d_m"),
        "public_p95_3d_m": row.get("public_p95_3d_m"),
        "public_max_3d_m": row.get("public_max_3d_m"),
        "uav_type_accuracy": row.get("uav_type_accuracy"),
        "classification_accuracy": row.get("classification_accuracy"),
        "scorecard_leaderboard_ready": row.get("scorecard_leaderboard_ready"),
        "public_missing_prediction_count": row.get("missing_prediction_count"),
        "public_extra_prediction_count": row.get("extra_prediction_count"),
        "public_duplicate_prediction_count": row.get("duplicate_prediction_count"),
    }


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    rows = summary.copy()
    has_pose = "pose_mse_loss_m2" in rows.columns and rows["pose_mse_loss_m2"].notna().any()
    if has_pose:
        rows["_sort_pose_mse"] = _numeric_column(rows, "pose_mse_loss_m2").fillna(np.inf)
        rows["_sort_p95"] = _numeric_column(rows, "public_p95_3d_m").fillna(np.inf)
        rows = rows.sort_values(["_sort_pose_mse", "_sort_p95", "variant"]).drop(
            columns=["_sort_pose_mse", "_sort_p95"]
        )
    else:
        rows["_ready"] = _bool_column(rows, "codabench_upload_ready")
        rows["_invalid"] = _numeric_column(rows, "invalid_resampled_rows").fillna(np.inf)
        rows = rows.sort_values(
            ["_ready", "_invalid", "variant"],
            ascending=[False, True, True],
        ).drop(columns=["_ready", "_invalid"])
    return rows.reset_index(drop=True)


def _has_ready_row(summary: pd.DataFrame) -> bool:
    if summary.empty:
        return False
    if "scorecard_leaderboard_ready" in summary.columns:
        column = "scorecard_leaderboard_ready"
    else:
        column = "leaderboard_ready"
    return bool(_bool_column(summary, column).any())


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series([np.nan] * len(rows), index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _bool_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series([False] * len(rows), index=rows.index, dtype=bool)
    return pd.Series(rows[column]).fillna(False).astype(bool)


def _parse_text_list(values: list[str]) -> tuple[str, ...]:
    parsed: list[str] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return tuple(parsed)


def _parse_optional_float_list(values: list[str]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            text = item.strip().lower()
            if not text:
                continue
            if text in {"none", "null", "off", "inf"}:
                parsed.append(None)
            else:
                parsed.append(float(text))
    return tuple(parsed or [None])


def _normalized_choices(
    values: tuple[str, ...],
    allowed: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    normalized = tuple(str(value).strip().lower() for value in values if str(value).strip())
    bad = [value for value in normalized if value not in allowed]
    if bad:
        raise ValueError(f"{name} values must be in {allowed}; got {bad}")
    return normalized


def _variant_label(method: str, gap_s: float | None, class_policy: str) -> str:
    gap_label = "gap_none" if gap_s is None else f"gap_{_number_label(gap_s)}s"
    return f"{method}_{gap_label}_{class_policy.replace('-', '_')}"


def _number_label(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
