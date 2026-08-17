#!/usr/bin/env python3
"""Regenerate and validate the public Multi-UAV LTS proposal-graph evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "raft-uav-multi-uav-lts-public-evidence-v1"
EXPECTED_INPUT_SCHEMA = "raft-uav-multi-uav-lts-public-inputs-v1"
PINNED_UPSTREAM_REVISION = "1dfce32f90842b3233e0bbe7a14a082f2386f933"
PINNED_REID_SHA256 = (
    "7f6e20a9082eb5fe100e005bd5d6f2bf"
    "9fde7d9aeff5575546ff3c530baf6363"
)
DEFAULT_SEQUENCE_COUNT = 102
DEFAULT_FRAME_COUNT = 77_293

CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("graph_default", ()),
    (
        "graph_conservative",
        (
            "--anchor-max-cost",
            "1.0",
            "--anchor-min-margin",
            "0.25",
            "--max-link-gap",
            "15",
            "--max-link-cost",
            "1.75",
            "--birth-min-hits",
            "4",
            "--birth-min-span",
            "3",
        ),
    ),
    (
        "graph_permissive",
        (
            "--anchor-max-cost",
            "1.5",
            "--anchor-min-margin",
            "0.05",
            "--max-link-gap",
            "45",
            "--max-link-cost",
            "2.75",
        ),
    ),
    (
        "graph_seed_only",
        (
            "--birth-min-hits",
            "1000000",
            "--birth-min-span",
            "1000000",
        ),
    ),
    ("graph_no_global", ("--no-global-links",)),
    ("graph_no_border", ("--border-gap-discount", "1.0")),
    (
        "graph_strict_birth",
        (
            "--birth-min-hits",
            "5",
            "--birth-min-span",
            "4",
            "--birth-min-mean-confidence",
            "0.01",
        ),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _directory_digest(directory: Path) -> tuple[str, int, int]:
    hasher = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in sorted(directory.glob("*.txt"), key=lambda item: item.name):
        data = path.read_bytes()
        encoded_name = path.name.encode("utf-8")
        hasher.update(len(encoded_name).to_bytes(8, "big"))
        hasher.update(encoded_name)
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)
        total_bytes += len(data)
        file_count += 1
    return hasher.hexdigest(), total_bytes, file_count


def _command_text(command: Sequence[object]) -> str:
    return " ".join(subprocess.list2cmdline([str(part)]) for part in command)


def _run(
    command: Sequence[object],
    *,
    log_path: Path,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    normalized = [str(part) for part in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ {_command_text(normalized)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"+ {_command_text(normalized)}\n")
        log.flush()
        process = subprocess.Popen(
            normalized,
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("subprocess stdout pipe was not created")
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, normalized)


def _path(payload: Mapping[str, Any], *keys: str) -> Path:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"missing public-inputs field: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid path field: {'.'.join(keys)}")
    return Path(value).expanduser().resolve()


def _truth_frame_one_lines(path: Path) -> tuple[str, ...]:
    selected: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        try:
            frame_id = int(float(fields[0]))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid frame id in {path}: {line!r}") from exc
        if frame_id == 1:
            selected.append(line)
    return tuple(selected)


def _validate_inputs(
    payload: Mapping[str, Any],
    *,
    run_dir: Path,
    expected_sequences: int,
    expected_frames: int,
) -> dict[str, Any]:
    if payload.get("schema") != EXPECTED_INPUT_SCHEMA:
        raise ValueError(
            f"public-inputs schema {payload.get('schema')!r} "
            f"!= {EXPECTED_INPUT_SCHEMA!r}"
        )

    truth_dir = _path(payload, "dataset", "truth_dir")
    seed_dir = _path(payload, "dataset", "first_frame_label_dir")
    image_root = _path(payload, "dataset", "train_sequence_root")
    checkout = _path(payload, "upstream", "checkout")
    botsort_root = checkout / "BoT-SORT"
    detector = botsort_root / "yolov12" / "weights" / "ViA_yolov12n.pt"
    reid_model = botsort_root / "logs" / "sbs_S50" / "model_0016.pth"
    reid_config = botsort_root / "logs" / "sbs_S50" / "config.yaml"
    inference = botsort_root / "tools" / "inference.py"

    for name, path in (
        ("truth_dir", truth_dir),
        ("first_frame_label_dir", seed_dir),
        ("train_sequence_root", image_root),
        ("upstream checkout", checkout),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} is not a directory: {path}")
    for name, path in (
        ("detector weights", detector),
        ("ReID model", reid_model),
        ("ReID config", reid_config),
        ("upstream inference entry point", inference),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is not a file: {path}")

    truth_files = sorted(truth_dir.glob("*.txt"), key=lambda item: item.name)
    seed_files = sorted(seed_dir.glob("*.txt"), key=lambda item: item.name)
    image_sequences = sorted(
        (path for path in image_root.iterdir() if path.is_dir()),
        key=lambda item: item.name,
    )
    truth_names = {path.stem for path in truth_files}
    seed_names = {path.stem for path in seed_files}
    image_names = {path.name for path in image_sequences}
    if len(truth_files) != expected_sequences:
        raise ValueError(
            f"truth sequence count {len(truth_files)} != {expected_sequences}"
        )
    if len(seed_files) != expected_sequences:
        raise ValueError(
            f"seed sequence count {len(seed_files)} != {expected_sequences}"
        )
    if len(image_sequences) != expected_sequences:
        raise ValueError(
            f"image sequence count {len(image_sequences)} != {expected_sequences}"
        )
    if not (truth_names == seed_names == image_names):
        raise ValueError(
            "truth, seed, and image sequence manifests differ: "
            f"truth_only={sorted(truth_names - seed_names - image_names)[:10]}, "
            f"seed_only={sorted(seed_names - truth_names - image_names)[:10]}, "
            f"image_only={sorted(image_names - truth_names - seed_names)[:10]}"
        )

    frame_count = 0
    empty_sequences: list[str] = []
    for sequence_dir in image_sequences:
        images = [
            path
            for path in sequence_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        ]
        frame_count += len(images)
        if not images:
            empty_sequences.append(sequence_dir.name)
    if frame_count != expected_frames:
        raise ValueError(f"frame count {frame_count} != {expected_frames}")
    if empty_sequences:
        raise ValueError(
            "image sequences contain no JPEG frames: " + ", ".join(empty_sequences[:10])
        )

    seed_rows = 0
    for truth_path in truth_files:
        expected_lines = _truth_frame_one_lines(truth_path)
        if not expected_lines:
            raise ValueError(f"{truth_path.name} has no frame-1 truth rows")
        seed_path = seed_dir / truth_path.name
        actual_lines = tuple(
            line.strip()
            for line in seed_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if actual_lines != expected_lines:
            raise ValueError(
                f"derived seed labels do not match frame-1 truth: {truth_path.name}"
            )
        seed_rows += len(actual_lines)

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        text=True,
    ).strip()
    declared_revision = payload.get("upstream", {}).get("revision")
    if revision != PINNED_UPSTREAM_REVISION or declared_revision != revision:
        raise ValueError(
            "upstream revision mismatch: "
            f"checkout={revision!r}, declared={declared_revision!r}, "
            f"pinned={PINNED_UPSTREAM_REVISION!r}"
        )

    model_sha256 = _sha256(reid_model)
    declared_model_sha256 = payload.get("reid", {}).get("model_sha256")
    if model_sha256 != PINNED_REID_SHA256 or declared_model_sha256 != model_sha256:
        raise ValueError(
            "ReID model digest mismatch: "
            f"actual={model_sha256!r}, declared={declared_model_sha256!r}"
        )
    if detector.stat().st_size < 1_000_000:
        raise ValueError(f"detector weights are implausibly small: {detector}")
    detector_sha256 = _sha256(detector)

    probe = {
        "schema": "raft-uav-multi-uav-lts-public-evidence-probe-v2",
        "generated_at_utc": _utc_now(),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "truth_sequence_count": len(truth_files),
        "seed_sequence_count": len(seed_files),
        "image_sequence_count": len(image_sequences),
        "frame_count": frame_count,
        "seed_row_count": seed_rows,
        "upstream_revision": revision,
        "detector_path": str(detector),
        "detector_size": detector.stat().st_size,
        "detector_sha256": detector_sha256,
        "reid_model_path": str(reid_model),
        "reid_model_size": reid_model.stat().st_size,
        "reid_model_sha256": model_sha256,
        "reid_config_path": str(reid_config),
        "inference_path": str(inference),
    }
    _write_json(run_dir / "probe.json", probe)
    return {
        "truth_dir": truth_dir,
        "seed_dir": seed_dir,
        "image_root": image_root,
        "checkout": checkout,
        "botsort_root": botsort_root,
        "detector_sha256": detector_sha256,
        "reid_model_sha256": model_sha256,
        "probe": probe,
    }


def _baseline_cache_key(
    payload: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_sequences: int,
    expected_frames: int,
    device: str,
) -> str:
    hasher = hashlib.sha256()
    settings = {
        "schema": "raft-uav-multi-uav-lts-proposal-source-key-v1",
        "upstream_revision": payload["upstream"]["revision"],
        "archive_md5": payload["dataset"]["archive_md5"],
        "detector_sha256": validated["detector_sha256"],
        "reid_model_sha256": validated["reid_model_sha256"],
        "expected_sequences": expected_sequences,
        "expected_frames": expected_frames,
        "device": device,
        "img_size": 1920,
        "proposal_conf_thres": 0.001,
        "proposal_iou_thres": 0.95,
        "device_independent": True,
    }
    hasher.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    source_paths = [
        repo_root / "scripts" / "run_multi_uav_lts_official_baseline.py",
        *sorted((repo_root / "src" / "raft_uav" / "multi_uav_lts").glob("*.py")),
    ]
    for path in source_paths:
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _valid_baseline_cache(
    baseline_root: Path,
    *,
    expected_sequences: int,
    cache_key: str,
) -> bool:
    manifest_path = baseline_root / "source_manifest.json"
    proposal_dir = baseline_root / "proposals"
    prediction_dir = baseline_root / "predictions"
    if not manifest_path.is_file() or not proposal_dir.is_dir() or not prediction_dir.is_dir():
        return False
    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if (
        manifest.get("cache_key") != cache_key
        or manifest.get("expected_sequence_count") != expected_sequences
    ):
        return False
    for directory, prefix in (
        (proposal_dir, "proposal"),
        (prediction_dir, "prediction"),
    ):
        digest, total_bytes, count = _directory_digest(directory)
        if count != expected_sequences or total_bytes <= 0:
            return False
        if manifest.get(f"{prefix}_content_sha256") != digest:
            return False
        if manifest.get(f"{prefix}_content_bytes") != total_bytes:
            return False
    return True


def _prepare_baseline(
    payload: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    run_dir: Path,
    device: str,
    expected_sequences: int,
    expected_frames: int,
) -> tuple[Path, Path, dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    work_root = _path(payload, "work_root")
    cache_key = _baseline_cache_key(
        payload,
        validated,
        repo_root=repo_root,
        expected_sequences=expected_sequences,
        expected_frames=expected_frames,
        device=device,
    )
    baseline_root = work_root / "baseline-cache" / cache_key
    reused = _valid_baseline_cache(
        baseline_root,
        expected_sequences=expected_sequences,
        cache_key=cache_key,
    )
    if reused:
        (run_dir / "proposal-baseline-console.txt").write_text(
            f"Reusing verified proposal source cache: {baseline_root}\n",
            encoding="utf-8",
        )
    else:
        shutil.rmtree(baseline_root, ignore_errors=True)
        baseline_root.mkdir(parents=True)
        _run(
            [
                sys.executable,
                "-m",
                "raft_uav.multi_uav_lts.proposal_baseline",
                "--work-root",
                work_root,
                "--botsort-root",
                validated["botsort_root"],
                "--sequence-root",
                validated["image_root"],
                "--first-frame-label-dir",
                validated["seed_dir"],
                "--output-dir",
                baseline_root,
                "--proposal-output-dir",
                baseline_root / "proposals",
                "--no-template",
                "--python",
                sys.executable,
                "--device",
                device,
                "--img-size",
                "1920",
                "--proposal-conf-thres",
                "0.001",
                "--proposal-iou-thres",
                "0.95",
                "--overwrite",
            ],
            log_path=run_dir / "proposal-baseline-console.txt",
        )

    proposal_dir = baseline_root / "proposals"
    prediction_dir = baseline_root / "predictions"
    proposal_digest, proposal_bytes, proposal_count = _directory_digest(proposal_dir)
    prediction_digest, prediction_bytes, prediction_count = _directory_digest(
        prediction_dir
    )
    if proposal_count != expected_sequences:
        raise ValueError(
            f"proposal bank covers {proposal_count} sequences, expected {expected_sequences}"
        )
    if prediction_count != expected_sequences:
        raise ValueError(
            f"raw control covers {prediction_count} sequences, expected {expected_sequences}"
        )
    if proposal_bytes <= 0 or prediction_bytes <= 0:
        raise ValueError("proposal bank or raw control is empty")

    manifest = {
        "schema": "raft-uav-multi-uav-lts-proposal-source-v1",
        "generated_at_utc": _utc_now(),
        "cache_key": cache_key,
        "cache_reused": reused,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "expected_sequence_count": expected_sequences,
        "expected_frame_count": expected_frames,
        "baseline_root": str(baseline_root),
        "proposal_dir": str(proposal_dir),
        "proposal_file_count": proposal_count,
        "proposal_content_bytes": proposal_bytes,
        "proposal_content_sha256": proposal_digest,
        "prediction_dir": str(prediction_dir),
        "prediction_file_count": prediction_count,
        "prediction_content_bytes": prediction_bytes,
        "prediction_content_sha256": prediction_digest,
        "device": device,
        "settings": {
            "img_size": 1920,
            "proposal_conf_thres": 0.001,
            "proposal_iou_thres": 0.95,
        },
    }
    _write_json(baseline_root / "source_manifest.json", manifest)
    _write_json(run_dir / "proposal-source-manifest.json", manifest)
    return proposal_dir, prediction_dir, manifest


def _run_candidates(
    proposal_dir: Path,
    seed_dir: Path,
    *,
    run_dir: Path,
    expected_sequences: int,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for name, extra_arguments in CANDIDATES:
        root = run_dir / name
        output_dir = root / "predictions"
        _run(
            [
                sys.executable,
                "-m",
                "raft_uav.multi_uav_lts.proposal_graph_tracker",
                proposal_dir,
                "--first-frame-label-dir",
                seed_dir,
                "--output-dir",
                output_dir,
                "--output-json",
                root / "summary.json",
                "--image-width",
                "1920",
                "--image-height",
                "1080",
                *extra_arguments,
            ],
            log_path=root / "console.txt",
        )
        _, total_bytes, count = _directory_digest(output_dir)
        if count != expected_sequences:
            raise ValueError(
                f"{name} covers {count} sequences, expected {expected_sequences}"
            )
        if total_bytes <= 0:
            raise ValueError(f"{name} produced no prediction rows")
        outputs[name] = output_dir
    return outputs


def _score(
    name: str,
    prediction_dir: Path,
    truth_dir: Path,
    *,
    run_dir: Path,
    expected_sequences: int,
) -> dict[str, Any]:
    root = run_dir / name
    metrics_path = root / "metrics.json"
    _run(
        [
            sys.executable,
            "-m",
            "raft_uav.multi_uav_lts.metrics",
            prediction_dir,
            "--truth-dir",
            truth_dir,
            "--output-json",
            metrics_path,
            "--sequence-summary-csv",
            root / "metrics_by_sequence.csv",
            "--alpha-summary-csv",
            root / "metrics_by_alpha.csv",
        ],
        log_path=root / "metrics-console.txt",
    )
    metrics = _load_json(metrics_path)
    if metrics.get("sequence_count") != expected_sequences:
        raise ValueError(
            f"{name} metrics cover {metrics.get('sequence_count')} sequences, "
            f"expected {expected_sequences}"
        )
    return metrics


def _run_tournament(
    raw_dir: Path,
    candidates: Mapping[str, Path],
    truth_dir: Path,
    *,
    run_dir: Path,
    expected_sequences: int,
    require_improvement: bool,
) -> None:
    command: list[object] = [
        sys.executable,
        "-m",
        "raft_uav.multi_uav_lts.tournament",
        raw_dir,
    ]
    for name, path in candidates.items():
        command.extend(["--candidate", f"{name}={path}"])
    command.extend(
        [
            "--truth-dir",
            truth_dir,
            "--output-dir",
            run_dir / "tournament",
            "--fold-count",
            "5",
            "--seed",
            "0",
            "--expected-sequence-count",
            str(expected_sequences),
            "--bootstrap-samples",
            "5000",
            "--min-mean-hota-gain",
            "0.001",
            "--min-ci-hota-gain",
            "0.0",
            "--max-mean-mota-drop",
            "0.002",
            "--max-mean-idf1-drop",
            "0.002",
            "--max-worst-scenario-hota-drop",
            "0.01",
            "--no-copy-selected",
        ]
    )
    if require_improvement:
        command.append("--require-improvement")
    _run(command, log_path=run_dir / "tournament-console.txt")


def _available_results(run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in ("raw", *(candidate[0] for candidate in CANDIDATES)):
        path = run_dir / name / "metrics.json"
        if path.is_file():
            try:
                metrics[name] = _load_json(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                metrics[name] = {"read_error": f"{type(exc).__name__}: {exc}"}
    tournament_path = run_dir / "tournament" / "tournament_summary.json"
    tournament: dict[str, Any] | None = None
    if tournament_path.is_file():
        try:
            tournament = _load_json(tournament_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            tournament = {"read_error": f"{type(exc).__name__}: {exc}"}
    return {"metrics": metrics, "tournament": tournament}


def _write_progress(run_dir: Path, stage: str, *, status: str = "running") -> None:
    _write_json(
        run_dir / "progress.json",
        {
            "schema": "raft-uav-multi-uav-lts-evidence-progress-v1",
            "updated_at_utc": _utc_now(),
            "git_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "stage": stage,
            "status": status,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--expected-sequence-count",
        type=int,
        default=DEFAULT_SEQUENCE_COUNT,
    )
    parser.add_argument(
        "--expected-frame-count",
        type=int,
        default=DEFAULT_FRAME_COUNT,
    )
    parser.add_argument("--require-improvement", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    stage = "load-public-inputs"
    try:
        _write_progress(run_dir, stage)
        payload = _load_json(args.inputs_json.expanduser().resolve())

        stage = "validate-public-inputs"
        _write_progress(run_dir, stage)
        validated = _validate_inputs(
            payload,
            run_dir=run_dir,
            expected_sequences=args.expected_sequence_count,
            expected_frames=args.expected_frame_count,
        )

        stage = "regenerate-proposal-source"
        _write_progress(run_dir, stage)
        proposal_dir, raw_dir, source_manifest = _prepare_baseline(
            payload,
            validated,
            run_dir=run_dir,
            device=str(args.device),
            expected_sequences=args.expected_sequence_count,
            expected_frames=args.expected_frame_count,
        )

        stage = "generate-graph-candidates"
        _write_progress(run_dir, stage)
        candidates = _run_candidates(
            proposal_dir,
            validated["seed_dir"],
            run_dir=run_dir,
            expected_sequences=args.expected_sequence_count,
        )

        stage = "score-raw-control"
        _write_progress(run_dir, stage)
        metrics = {
            "raw": _score(
                "raw",
                raw_dir,
                validated["truth_dir"],
                run_dir=run_dir,
                expected_sequences=args.expected_sequence_count,
            )
        }
        for name, path in candidates.items():
            stage = f"score-{name}"
            _write_progress(run_dir, stage)
            metrics[name] = _score(
                name,
                path,
                validated["truth_dir"],
                run_dir=run_dir,
                expected_sequences=args.expected_sequence_count,
            )

        stage = "guarded-tournament"
        _write_progress(run_dir, stage)
        _run_tournament(
            raw_dir,
            candidates,
            validated["truth_dir"],
            run_dir=run_dir,
            expected_sequences=args.expected_sequence_count,
            require_improvement=args.require_improvement,
        )
        tournament = _load_json(run_dir / "tournament" / "tournament_summary.json")
        selected_candidate = tournament.get("selected_candidate")
        if not isinstance(selected_candidate, str) or not selected_candidate:
            raise ValueError("tournament did not record a selected candidate")
        if args.require_improvement and selected_candidate == "raw":
            raise RuntimeError("guarded tournament selected the raw fallback")

        stage = "complete"
        summary = {
            "schema": SCHEMA,
            "status": "success",
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "git_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "device": str(args.device),
            "expected_sequence_count": args.expected_sequence_count,
            "expected_frame_count": args.expected_frame_count,
            "require_improvement": args.require_improvement,
            "public_inputs": payload,
            "probe": validated["probe"],
            "proposal_source": source_manifest,
            "selected_candidate": selected_candidate,
            "selection_status": tournament.get("selection_status"),
            "metrics": metrics,
            "tournament": tournament,
        }
        _write_json(run_dir / "evidence-summary.json", summary)
        _write_progress(run_dir, stage, status="success")
        print(f"selected_candidate={selected_candidate}")
        print(f"selection_status={tournament.get('selection_status')}")
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "failure",
            "started_at_utc": started_at,
            "failed_at_utc": _utc_now(),
            "stage": stage,
            "git_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "device": str(args.device),
            "expected_sequence_count": args.expected_sequence_count,
            "expected_frame_count": args.expected_frame_count,
            "require_improvement": args.require_improvement,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            **_available_results(run_dir),
        }
        _write_json(run_dir / "evidence-failure.json", failure)
        _write_progress(run_dir, stage, status="failure")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
