#!/usr/bin/env python
"""Run and package the upstream Multi-UAV LTS YOLOv12-BoT-SORT baseline.

This script intentionally treats YOLOv12-BoT-SORT as an external dependency. It
coordinates the already-cloned upstream checkout, first-frame test labels, and
RaFT-UAV's lightweight LTS submission validator/packager.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.multi_uav_lts.cli import (  # noqa: E402
    _write_file_summary_csv,
    package_submission,
)


DEFAULT_WORK_ROOT = Path("/mnt/lexar4tb/multi_uav_lts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--botsort-root", type=Path)
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--first-frame-label-dir", type=Path)
    parser.add_argument("--template-zip", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--submission-zip", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--file-summary-csv", type=Path)
    parser.add_argument("--weights", default="./yolov12/weights/ViA_yolov12n.pt")
    parser.add_argument("--img-size", type=int, default=1600)
    parser.add_argument("--track-buffer", type=int, default=60)
    parser.add_argument("--device", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SHARD_INDEX", "0")))
    parser.add_argument("--shard-count", type=int, default=int(os.environ.get("SHARD_COUNT", "1")))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--fast-reid-config", default="logs/sbs_S50/config.yaml")
    parser.add_argument("--fast-reid-weights", default="logs/sbs_S50/model_0016.pth")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--sort-rows", action="store_true", default=True)
    parser.add_argument("--no-sort-rows", dest="sort_rows", action="store_false")
    args = parser.parse_args(argv)
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")

    paths = _resolve_paths(args)
    _validate_inputs(paths, require_botsort=not args.package_only)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["predictions_dir"].mkdir(parents=True, exist_ok=True)
    (paths["output_dir"] / "logs").mkdir(exist_ok=True)
    (paths["output_dir"] / "runs").mkdir(exist_ok=True)

    records: list[dict[str, Any]] = []
    if not args.package_only:
        records = _run_sequences(args, paths)

    validation_payload = None
    if not args.dry_run or args.package_only:
        validation = package_submission(
            paths["predictions_dir"],
            paths["submission_zip"],
            template_zip=paths["template_zip"],
            normalize=args.normalize,
            sort_rows=args.sort_rows,
        )
        _write_file_summary_csv(validation, paths["file_summary_csv"])
        validation_payload = asdict(validation)
    summary = {
        "schema": "raft-uav-multi-uav-lts-official-baseline-v1",
        "work_root": str(paths["work_root"]),
        "botsort_root": str(paths["botsort_root"]),
        "sequence_root": str(paths["sequence_root"]),
        "first_frame_label_dir": str(paths["first_frame_label_dir"]),
        "template_zip": str(paths["template_zip"]),
        "output_dir": str(paths["output_dir"]),
        "predictions_dir": str(paths["predictions_dir"]),
        "submission_zip": str(paths["submission_zip"]),
        "normalize": args.normalize,
        "sort_rows": args.sort_rows,
        "package_only": args.package_only,
        "dry_run": args.dry_run,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "records": records,
        "submission_validation": validation_payload,
    }
    _write_json(summary, paths["summary_json"])
    print(f"multi_uav_lts_summary_json={paths['summary_json']}")
    print(f"multi_uav_lts_submission_zip={paths['submission_zip']}")
    print(
        "multi_uav_lts_valid="
        + ("not_run_dry_run" if validation_payload is None else str(validation_payload["valid"]))
    )
    if validation_payload is not None and not validation_payload["valid"]:
        return 1
    return 0


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    work_root = args.work_root
    output_dir = args.output_dir or work_root / "outputs/official_baseline_via_first_init"
    return {
        "work_root": work_root,
        "botsort_root": args.botsort_root
        or work_root / "repos/YOLOv12-BoT-SORT-ReID/BoT-SORT",
        "sequence_root": args.sequence_root or work_root / "extracted/TestImages",
        "first_frame_label_dir": args.first_frame_label_dir
        or work_root / "extracted/TestLabels_FirstFrameOnly",
        "template_zip": args.template_zip or work_root / "downloads/submission.zip",
        "output_dir": output_dir,
        "predictions_dir": args.predictions_dir or output_dir / "predictions",
        "submission_zip": args.submission_zip or output_dir / "submission.zip",
        "summary_json": args.summary_json or output_dir / "multi_uav_lts_run_summary.json",
        "file_summary_csv": args.file_summary_csv or output_dir / "submission_file_summary.csv",
    }


def _validate_inputs(paths: dict[str, Path], *, require_botsort: bool) -> None:
    for key in ("sequence_root", "first_frame_label_dir", "template_zip"):
        if not paths[key].exists():
            raise FileNotFoundError(f"{key} does not exist: {paths[key]}")
    if require_botsort and not (paths["botsort_root"] / "tools/inference.py").exists():
        raise FileNotFoundError(f"missing YOLOv12-BoT-SORT inference.py under {paths['botsort_root']}")


def _run_sequences(args: argparse.Namespace, paths: dict[str, Path]) -> list[dict[str, Any]]:
    sequences = [path for path in sorted(paths["sequence_root"].iterdir()) if path.is_dir()]
    records: list[dict[str, Any]] = []
    env = os.environ.copy()
    botsort_root = paths["botsort_root"]
    env["PYTHONPATH"] = ":".join(
        str(path)
        for path in (botsort_root / "yolov12", botsort_root)
        if path.exists()
    ) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    for index, sequence_dir in enumerate(sequences):
        if index % args.shard_count != args.shard_index:
            continue
        sequence = sequence_dir.name
        prediction_path = paths["predictions_dir"] / f"{sequence}.txt"
        log_path = paths["output_dir"] / "logs" / f"{sequence}.log"
        if prediction_path.exists() and prediction_path.stat().st_size > 0 and not args.overwrite:
            records.append({"sequence": sequence, "status": "skipped_existing", "prediction": str(prediction_path)})
            continue
        command = _inference_command(args, paths, sequence_dir)
        record = {
            "sequence": sequence,
            "status": "dry_run" if args.dry_run else "planned",
            "command": " ".join(shlex.quote(part) for part in command),
            "prediction": str(prediction_path),
            "log": str(log_path),
        }
        if not args.dry_run:
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=paths["botsort_root"],
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record["returncode"] = completed.returncode
            record["status"] = "ok" if completed.returncode == 0 else "failed"
            if completed.returncode != 0:
                records.append(record)
                raise RuntimeError(f"{sequence} failed; see {log_path}")
        records.append(record)
    return records


def _inference_command(
    args: argparse.Namespace,
    paths: dict[str, Path],
    sequence_dir: Path,
) -> list[str]:
    sequence = sequence_dir.name
    command = [
        args.python,
        "tools/inference.py",
        "--weights",
        args.weights,
        "--source",
        str(sequence_dir),
        "--with-initial-positions",
        "--initial-position-config",
        str(paths["first_frame_label_dir"] / f"{sequence}.txt"),
        "--img-size",
        str(args.img_size),
        "--track_buffer",
        str(args.track_buffer),
        "--device",
        str(args.device),
        "--agnostic-nms",
        "--save_path_answer",
        str(paths["predictions_dir"]),
        "--project",
        str(paths["output_dir"] / "runs"),
        "--hide-labels-name",
    ]
    if not args.no_reid:
        command.extend(
            [
                "--with-reid",
                "--fast-reid-config",
                args.fast_reid_config,
                "--fast-reid-weights",
                args.fast_reid_weights,
            ]
        )
    return command


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
