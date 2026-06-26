"""Copy Track 5 classification labels onto another official submission."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    load_official_track5_results_frame,
    validate_official_track5_submission,
)


RELABEL_RESULTS_CSV = "mmaud_results_relabelled.csv"
RELABEL_ZIP = "ug2_submission_relabelled.zip"
RELABEL_DIAGNOSTICS_CSV = "mmuad_track5_classification_relabel_diagnostics.csv"
RELABEL_MANIFEST_JSON = "mmuad_track5_classification_relabel_manifest.json"
RELABEL_VALIDATION_JSON = "mmuad_track5_classification_relabel_validation.json"
RelabelMode = Literal["by-key", "by-sequence-majority"]


@dataclass(frozen=True)
class ClassificationRelabelResult:
    """Official relabelled rows plus diagnostics."""

    rows: pd.DataFrame
    diagnostics: pd.DataFrame
    manifest: dict[str, Any]


def relabel_track5_classification(
    pose_submission: pd.DataFrame,
    classification_submission: pd.DataFrame,
    *,
    mode: RelabelMode = "by-key",
) -> ClassificationRelabelResult:
    """Return pose rows with labels copied from another official submission."""

    pose = _normalize_frame(pose_submission, name="pose_submission")
    source = _normalize_frame(classification_submission, name="classification_submission")
    if mode == "by-key":
        labels = source[["Sequence", "Timestamp", "Classification"]].rename(
            columns={"Classification": "source_classification"}
        )
        merged = pose.merge(labels, on=["Sequence", "Timestamp"], how="left", validate="one_to_one")
    elif mode == "by-sequence-majority":
        labels = (
            source.groupby("Sequence", sort=True)["Classification"]
            .agg(_majority_class)
            .rename("source_classification")
            .reset_index()
        )
        merged = pose.merge(labels, on="Sequence", how="left", validate="many_to_one")
    else:
        raise ValueError("classification relabel mode must be 'by-key' or 'by-sequence-majority'")
    if merged["source_classification"].isna().any():
        missing = merged.loc[merged["source_classification"].isna(), ["Sequence", "Timestamp"]]
        raise ValueError(f"classification source is missing {len(missing)} pose rows")
    diagnostics = merged[
        ["Sequence", "Timestamp", "Classification", "source_classification"]
    ].copy()
    diagnostics.rename(columns={"Classification": "pose_classification"}, inplace=True)
    diagnostics["relabelled_classification"] = diagnostics["source_classification"].astype(int)
    diagnostics["classification_changed"] = (
        diagnostics["pose_classification"].astype(int)
        != diagnostics["relabelled_classification"].astype(int)
    )
    out = pose.copy()
    out["Classification"] = merged["source_classification"].astype(int)
    manifest = {
        "schema": "raft-uav-mmuad-track5-classification-relabel-v1",
        "mode": str(mode),
        "row_count": int(len(out)),
        "sequence_count": int(out["Sequence"].nunique()) if not out.empty else 0,
        "changed_row_count": int(diagnostics["classification_changed"].sum()),
        "changed_sequence_count": int(
            diagnostics.loc[diagnostics["classification_changed"], "Sequence"].nunique()
        ),
    }
    return ClassificationRelabelResult(
        rows=out[["Sequence", "Timestamp", "Position", "Classification"]],
        diagnostics=diagnostics,
        manifest=manifest,
    )


def write_track5_classification_relabel_outputs(
    *,
    result: ClassificationRelabelResult,
    output_dir: Path,
    pose_submission_path: Path,
    classification_submission_path: Path,
    template: pd.DataFrame | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write relabelled official CSV/ZIP plus manifest and validation."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_csv": output / RELABEL_RESULTS_CSV,
        "zip": output / RELABEL_ZIP,
        "diagnostics_csv": output / RELABEL_DIAGNOSTICS_CSV,
        "manifest_json": output / RELABEL_MANIFEST_JSON,
    }
    result.rows.to_csv(paths["results_csv"], index=False)
    with ZipFile(paths["zip"], "w", compression=ZIP_DEFLATED) as archive:
        archive.write(paths["results_csv"], arcname="mmaud_results.csv")
    result.diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    validation_summary: dict[str, Any] | None = None
    if template is not None:
        validation = validate_official_track5_submission(
            paths["zip"],
            template=template,
            require_zip=True,
        )
        validation_summary = _jsonable(validation.summary)
        paths["validation_json"] = output / RELABEL_VALIDATION_JSON
        paths["validation_json"].write_text(
            json.dumps(validation_summary, indent=2),
            encoding="utf-8",
        )
        if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", []))
            raise SystemExit(f"relabelled submission is not leaderboard-ready: {reasons}")
    manifest = dict(result.manifest)
    manifest.update(
        {
            "pose_submission": str(pose_submission_path),
            "classification_submission": str(classification_submission_path),
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-classification-relabel",
        description=__doc__,
    )
    parser.add_argument("--pose-submission", type=Path, required=True)
    parser.add_argument("--classification-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("by-key", "by-sequence-majority"),
        default="by-key",
    )
    parser.add_argument("--template", type=Path)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    result = relabel_track5_classification(
        load_official_track5_results_frame(args.pose_submission),
        load_official_track5_results_frame(args.classification_submission),
        mode=args.mode,
    )
    template = None if args.template is None else pd.read_csv(args.template)
    paths = write_track5_classification_relabel_outputs(
        result=result,
        output_dir=args.output_dir,
        pose_submission_path=args.pose_submission,
        classification_submission_path=args.classification_submission,
        template=template,
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_classification_relabel=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    validation = manifest.get("validation") or {}
    if validation:
        print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
        print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    return 0


def _normalize_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = {"Sequence", "Timestamp", "Position", "Classification"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing official columns: {sorted(missing)}")
    out = frame[["Sequence", "Timestamp", "Position", "Classification"]].copy()
    out["Sequence"] = out["Sequence"].astype(str)
    out["Timestamp"] = pd.to_numeric(out["Timestamp"], errors="coerce")
    out["Classification"] = pd.to_numeric(out["Classification"], errors="coerce")
    if not np.isfinite(out[["Timestamp", "Classification"]].to_numpy(float)).all():
        raise ValueError(f"{name} contains non-finite Timestamp or Classification")
    return out.sort_values(["Sequence", "Timestamp"]).reset_index(drop=True)


def _majority_class(values: pd.Series) -> int:
    counts = values.astype(int).value_counts()
    if counts.empty:
        raise ValueError("cannot compute majority class for empty sequence")
    max_count = counts.max()
    return int(counts.loc[counts == max_count].sort_index().index[0])


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


if __name__ == "__main__":
    raise SystemExit(main())
