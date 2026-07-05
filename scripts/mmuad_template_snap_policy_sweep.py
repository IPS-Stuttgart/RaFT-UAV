#!/usr/bin/env python
"""Sweep MMUAD Track 5 template-snapping policies.

Dense tracker outputs can contain extra sensor-time predictions.  This helper
runs the existing template snapper over several interpolation/classification
policies and records which variants produce upload-ready Codabench ZIPs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.submission import (  # noqa: E402
    load_official_track5_results_frame,
    load_official_track5_template_file,
)
from raft_uav.mmuad.template_snap_utils import CLASSIFICATION_POLICIES, RESAMPLE_METHODS  # noqa: E402
from raft_uav.mmuad.template_snap_write import write_template_snapped_submission  # noqa: E402

SUMMARY_CSV = "mmuad_template_snap_policy_sweep_summary.csv"
SUMMARY_JSON = "mmuad_template_snap_policy_sweep_summary.json"


def run_template_snap_policy_sweep(
    *,
    results_path: Path,
    template_path: Path,
    output_dir: Path,
    resample_methods: Iterable[str] = ("linear", "nearest"),
    max_interpolation_gap_s_values: Iterable[float | None] = (None,),
    classification_policies: Iterable[str] = ("sequence-mode",),
    missing_position_policy: str = "zero",
) -> pd.DataFrame:
    """Write one snapped submission bundle per policy and return a summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_official_track5_results_frame(results_path)
    template = load_official_track5_template_file(template_path)
    records: list[dict[str, Any]] = []
    for method in _parse_resample_methods(resample_methods):
        for gap_s in _normalize_gap_values(max_interpolation_gap_s_values):
            for policy in _parse_classification_policies(classification_policies):
                label = _variant_label(method, gap_s, policy)
                variant_dir = output_dir / label
                paths = write_template_snapped_submission(
                    results=results,
                    template=template,
                    output_dir=variant_dir,
                    resample_method=method,
                    max_interpolation_gap_s=gap_s,
                    classification_policy=policy,
                    missing_position_policy=missing_position_policy,
                )
                manifest = _load_json(paths["manifest_json"])
                validation = _load_json(paths["validation_json"])
                records.append(
                    _summary_record(
                        label=label,
                        method=method,
                        gap_s=gap_s,
                        policy=policy,
                        missing_position_policy=missing_position_policy,
                        paths=paths,
                        manifest=manifest,
                        validation=validation,
                    )
                )
    summary = _sort_summary(pd.DataFrame.from_records(records))
    summary.to_csv(output_dir / SUMMARY_CSV, index=False)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps({"rows": _jsonable(summary.to_dict(orient="records"))}, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resample-methods", default="linear,nearest")
    parser.add_argument("--max-interpolation-gap-s", default="none,1,2,5")
    parser.add_argument("--classification-policies", default="sequence-mode,nearest")
    parser.add_argument("--missing-position-policy", choices=("zero", "raise"), default="zero")
    parser.add_argument("--require-any-upload-ready", action="store_true")
    args = parser.parse_args(argv)

    summary = run_template_snap_policy_sweep(
        results_path=args.results,
        template_path=args.template,
        output_dir=args.output_dir,
        resample_methods=_split_csv(args.resample_methods),
        max_interpolation_gap_s_values=_parse_gap_list(args.max_interpolation_gap_s),
        classification_policies=_split_csv(args.classification_policies),
        missing_position_policy=args.missing_position_policy,
    )
    ready_count = int(summary.get("codabench_upload_ready", pd.Series(dtype=bool)).astype(bool).sum())
    print("mmuad_template_snap_policy_sweep=ok")
    print(f"summary_csv={args.output_dir / SUMMARY_CSV}")
    print(f"variant_count={len(summary)}")
    print(f"codabench_upload_ready_count={ready_count}")
    if args.require_any_upload_ready and ready_count == 0:
        raise SystemExit("no swept policy produced an upload-ready ZIP")
    return 0


def _summary_record(
    *,
    label: str,
    method: str,
    gap_s: float | None,
    policy: str,
    missing_position_policy: str,
    paths: dict[str, Path],
    manifest: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant_label": label,
        "resample_method": method,
        "max_interpolation_gap_s": gap_s,
        "classification_policy": policy,
        "missing_position_policy": missing_position_policy,
        "leaderboard_ready": bool(validation.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.get("codabench_upload_ready", False)),
        "leaderboard_blocking_reasons": ";".join(
            str(item) for item in validation.get("leaderboard_blocking_reasons", []) or []
        ),
        "row_count": int(manifest.get("row_count", 0)),
        "template_row_count": int(manifest.get("template_row_count", 0)),
        "valid_snapped_rows": int(manifest.get("valid_snapped_rows", 0)),
        "invalid_snapped_rows": int(manifest.get("invalid_snapped_rows", 0)),
        "extrapolated_rows": int(manifest.get("extrapolated_rows", 0)),
        "large_gap_fallback_rows": int(manifest.get("large_gap_fallback_rows", 0)),
        "official_zip": str(paths["official_zip"]),
        "official_results_csv": str(paths["official_results_csv"]),
        "manifest_json": str(paths["manifest_json"]),
        "validation_json": str(paths["validation_json"]),
        "diagnostics_csv": str(paths["diagnostics_csv"]),
    }


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return summary.sort_values(
        ["codabench_upload_ready", "invalid_snapped_rows", "large_gap_fallback_rows", "variant_label"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _variant_label(method: str, gap_s: float | None, policy: str) -> str:
    gap_text = "none" if gap_s is None else f"{float(gap_s):.6g}s"
    text = f"{method}_gap_{gap_text}_{policy}"
    return text.replace("/", "_").replace("\\", "_").replace(".", "p").replace("-", "_")


def _parse_resample_methods(values: Iterable[str]) -> tuple[str, ...]:
    methods = tuple(_split_values(values))
    unknown = sorted(set(methods).difference(RESAMPLE_METHODS))
    if unknown:
        raise ValueError(f"unsupported resample methods: {unknown}")
    return methods or tuple(RESAMPLE_METHODS)


def _parse_classification_policies(values: Iterable[str]) -> tuple[str, ...]:
    policies = tuple(_split_values(values))
    unknown = sorted(set(policies).difference(CLASSIFICATION_POLICIES))
    if unknown:
        raise ValueError(f"unsupported classification policies: {unknown}")
    return policies or tuple(CLASSIFICATION_POLICIES)


def _parse_gap_list(value: str) -> tuple[float | None, ...]:
    return _normalize_gap_values(_split_csv(value))


def _normalize_gap_values(values: Iterable[float | str | None]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    seen: set[str] = set()
    for value in values:
        if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
            item = None
            key = "none"
        else:
            item = float(value)
            if item < 0.0:
                raise ValueError("max-interpolation-gap values must be non-negative")
            key = f"{item:.12g}"
        if key not in seen:
            seen.add(key)
            parsed.append(item)
    return tuple(parsed or [None])


def _split_values(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(_split_csv(value))
    return out


def _split_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).replace(";", ",").split(",") if item.strip()]


def _load_json(path: Path) -> dict[str, Any]:
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
