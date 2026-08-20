#!/usr/bin/env python3
"""Run the public evidence gate with guarded association-improvement candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_multi_uav_lts_public_evidence as evidence

IMPROVED_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "graph_common_motion",
        (
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
        ),
    ),
    (
        "graph_delayed_path_cover",
        (
            "--enable-delayed-path-cover",
            "--delayed-max-gap",
            "0",
            "--delayed-lookahead-frames",
            "2",
            "--delayed-successors-per-frame",
            "3",
            "--delayed-continuation-weight",
            "0.75",
        ),
    ),
    (
        "graph_delayed_common_motion",
        (
            "--enable-delayed-path-cover",
            "--delayed-max-gap",
            "0",
            "--delayed-lookahead-frames",
            "2",
            "--delayed-successors-per-frame",
            "3",
            "--delayed-continuation-weight",
            "0.75",
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
        ),
    ),
    (
        "graph_interpolate_one",
        ("--interpolate-max-gap", "1"),
    ),
    (
        "graph_common_motion_interpolate",
        (
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
            "--interpolate-max-gap",
            "1",
        ),
    ),
    (
        "graph_guarded_border_birth",
        (
            "--birth-require-border-entry",
            "--birth-min-inward-motion",
            "0.0",
        ),
    ),
    (
        "graph_inward_border_birth",
        (
            "--birth-require-border-entry",
            "--birth-min-inward-motion",
            "0.25",
        ),
    ),
    (
        "graph_common_motion_guarded_birth",
        (
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
            "--birth-require-border-entry",
            "--birth-min-inward-motion",
            "0.0",
        ),
    ),
    (
        "graph_seed_calibrated",
        (
            "--enable-seed-calibration",
            "--seed-calibration-min-pairs",
            "2",
        ),
    ),
    (
        "graph_seed_calibrated_common_motion",
        (
            "--enable-seed-calibration",
            "--seed-calibration-min-pairs",
            "2",
            "--enable-common-motion",
            "--common-motion-min-pairs",
            "4",
            "--common-motion-max-normalized-step",
            "8.0",
            "--common-motion-max-normalized-residual",
            "1.5",
        ),
    ),
    (
        "graph_seed_calibrated_interpolate",
        (
            "--enable-seed-calibration",
            "--seed-calibration-min-pairs",
            "2",
            "--interpolate-max-gap",
            "1",
        ),
    ),
)

_JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)
_JPEG_STANDALONE_MARKERS = frozenset({0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)})
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def _proposal_source_cache_key(
    payload: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_sequences: int,
    expected_frames: int,
    device: str,
) -> str:
    """Hash detector-generation inputs without invalidating on graph-only edits."""
    settings = {
        "schema": "raft-uav-multi-uav-lts-proposal-source-key-v3",
        "upstream_revision": payload["upstream"]["revision"],
        "archive_md5": payload["dataset"]["archive_md5"],
        "detector_sha256": validated["detector_sha256"],
        "reid_model_sha256": validated["reid_model_sha256"],
        "cython_bbox_version": "0.1.5",
        "faiss_cpu_version": "1.8.0.post1",
        "expected_sequences": expected_sequences,
        "expected_frames": expected_frames,
        "device": device,
        "img_size": 1920,
        "proposal_conf_thres": 0.001,
        "proposal_iou_thres": 0.95,
        "device_independent": False,
    }
    source_root = repo_root / "src" / "raft_uav"
    package_root = source_root / "multi_uav_lts"
    source_paths = [
        repo_root / "scripts" / "run_multi_uav_lts_official_baseline.py",
        source_root / "__init__.py",
        *sorted((source_root / "numeric").rglob("*.py")),
        package_root / "__init__.py",
        package_root / "_records.py",
        package_root / "cli.py",
        package_root / "improved_baseline.py",
        package_root / "proposal_baseline.py",
        package_root / "proposal_export.py",
        package_root / "upstream_fixes.py",
        *sorted((package_root / "_records").rglob("*.py")),
        *sorted((package_root / "cli").rglob("*.py")),
    ]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"proposal cache source missing: {rendered}")

    hasher = hashlib.sha256()
    hasher.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for path in source_paths:
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError(f"{path}: missing JPEG SOI marker")
        while True:
            marker_prefix = handle.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b"\xff":
                continue
            marker_byte = handle.read(1)
            while marker_byte == b"\xff":
                marker_byte = handle.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]
            if marker == 0x00:
                continue
            if marker in _JPEG_STANDALONE_MARKERS:
                if marker == 0xD9:
                    break
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                raise ValueError(f"{path}: truncated JPEG segment length")
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                raise ValueError(f"{path}: invalid JPEG segment length")
            payload_length = segment_length - 2
            if marker in _JPEG_SOF_MARKERS:
                payload = handle.read(payload_length)
                if len(payload) != payload_length or len(payload) < 5:
                    raise ValueError(f"{path}: truncated JPEG SOF segment")
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                if width <= 0 or height <= 0:
                    raise ValueError(f"{path}: invalid JPEG dimensions {width}x{height}")
                return width, height
            handle.seek(payload_length, os.SEEK_CUR)
    raise ValueError(f"{path}: JPEG dimensions were not found")


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: invalid PNG header")
    if header[12:16] != b"IHDR":
        raise ValueError(f"{path}: PNG IHDR chunk is missing")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: invalid PNG dimensions {width}x{height}")
    return width, height


def _image_dimensions(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_dimensions(path)
    if suffix == ".png":
        return _png_dimensions(path)
    raise ValueError(f"unsupported image format: {path}")


def _sequence_resolution_groups(
    image_root: Path,
    seed_dir: Path,
) -> tuple[tuple[tuple[int, int], tuple[str, ...]], ...]:
    seed_sequences = tuple(sorted(path.stem for path in seed_dir.glob("*.txt")))
    if not seed_sequences:
        raise ValueError(f"seed directory contains no sequence files: {seed_dir}")
    groups: dict[tuple[int, int], list[str]] = {}
    for sequence in seed_sequences:
        sequence_dir = image_root / sequence
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"sequence image directory is missing: {sequence_dir}")
        frames = sorted(
            path
            for path in sequence_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not frames:
            raise ValueError(f"sequence contains no supported images: {sequence_dir}")
        dimensions = _image_dimensions(frames[0])
        if len(frames) > 1:
            final_dimensions = _image_dimensions(frames[-1])
            if final_dimensions != dimensions:
                raise ValueError(
                    f"{sequence}: frame dimensions change from "
                    f"{dimensions[0]}x{dimensions[1]} to "
                    f"{final_dimensions[0]}x{final_dimensions[1]}"
                )
        groups.setdefault(dimensions, []).append(sequence)
    return tuple(
        (dimensions, tuple(sequences))
        for dimensions, sequences in sorted(groups.items())
    )


def _run_candidates_with_native_dimensions(
    proposal_dir: Path,
    seed_dir: Path,
    *,
    image_root: Path,
    run_dir: Path,
    expected_sequences: int,
) -> dict[str, Path]:
    groups = _sequence_resolution_groups(image_root, seed_dir)
    covered = sum(len(sequences) for _dimensions, sequences in groups)
    if covered != expected_sequences:
        raise ValueError(
            f"resolution inventory covers {covered} sequences, "
            f"expected {expected_sequences}"
        )

    outputs: dict[str, Path] = {}
    for name, extra_arguments in evidence.CANDIDATES:
        root = run_dir / name
        output_dir = root / "predictions"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)
        group_records: list[dict[str, Any]] = []
        copied_names: set[str] = set()
        for (width, height), sequences in groups:
            group_name = f"{width}x{height}"
            group_root = root / "resolution-groups" / group_name
            group_output = group_root / "predictions"
            evidence._run(
                [
                    sys.executable,
                    "-m",
                    "raft_uav.multi_uav_lts.experimental_proposal_graph_tracker",
                    proposal_dir,
                    "--first-frame-label-dir",
                    seed_dir,
                    "--output-dir",
                    group_output,
                    "--output-json",
                    group_root / "summary.json",
                    "--image-width",
                    str(width),
                    "--image-height",
                    str(height),
                    *extra_arguments,
                    "--sequences",
                    *sequences,
                ],
                log_path=root / f"{group_name}-console.txt",
            )
            _digest, total_bytes, count = evidence._directory_digest(group_output)
            if count != len(sequences):
                raise ValueError(
                    f"{name}/{group_name} covers {count} sequences, "
                    f"expected {len(sequences)}"
                )
            if total_bytes <= 0:
                raise ValueError(f"{name}/{group_name} produced no prediction rows")
            for source in sorted(group_output.glob("*.txt")):
                if source.name in copied_names:
                    raise ValueError(
                        f"{name}: duplicate sequence produced by resolution groups: "
                        f"{source.name}"
                    )
                shutil.copy2(source, output_dir / source.name)
                copied_names.add(source.name)
            group_records.append(
                {
                    "width": width,
                    "height": height,
                    "sequence_count": len(sequences),
                    "sequences": list(sequences),
                    "prediction_content_bytes": total_bytes,
                }
            )

        digest, total_bytes, count = evidence._directory_digest(output_dir)
        if count != expected_sequences:
            raise ValueError(
                f"{name} covers {count} sequences, expected {expected_sequences}"
            )
        if total_bytes <= 0:
            raise ValueError(f"{name} produced no prediction rows")
        evidence._write_json(
            root / "native-resolution-summary.json",
            {
                "schema": "raft-uav-multi-uav-lts-native-resolution-candidate-v1",
                "candidate": name,
                "image_root": str(image_root),
                "sequence_count": count,
                "prediction_content_bytes": total_bytes,
                "prediction_content_sha256": digest,
                "resolution_groups": group_records,
            },
        )
        outputs[name] = output_dir
    return outputs


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _retain_safe_baseline_prefix(
    baseline_root: Path,
    image_root: Path,
) -> tuple[int, int]:
    """Retain completed-looking prefix files and discard the uncertain tail."""
    proposal_dir = baseline_root / "proposals"
    prediction_dir = baseline_root / "predictions"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    sequences = tuple(
        path.name for path in sorted(image_root.iterdir()) if path.is_dir()
    )
    prefix: list[str] = []
    for sequence in sequences:
        proposal = proposal_dir / f"{sequence}.txt"
        prediction = prediction_dir / f"{sequence}.txt"
        if _nonempty_file(proposal) and _nonempty_file(prediction):
            prefix.append(sequence)
        else:
            break

    # A cancelled inference can leave nonempty but incomplete files for its current
    # sequence. Sacrifice the last apparent success because no upstream done marker
    # exists; every earlier sequence was processed sequentially and is safe to keep.
    safe_prefix = tuple(prefix[:-1]) if prefix else ()
    retained = set(safe_prefix)
    removed = 0
    expected_names = {f"{sequence}.txt" for sequence in sequences}
    for directory in (proposal_dir, prediction_dir):
        for path in directory.glob("*.txt"):
            if path.name not in expected_names or path.stem not in retained:
                path.unlink()
                removed += 1
    return len(safe_prefix), removed


def _prepare_baseline_resumable(
    payload: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    run_dir: Path,
    device: str,
    expected_sequences: int,
    expected_frames: int,
) -> tuple[Path, Path, dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    work_root = evidence._path(payload, "work_root")
    cache_key = _proposal_source_cache_key(
        payload,
        validated,
        repo_root=repo_root,
        expected_sequences=expected_sequences,
        expected_frames=expected_frames,
        device=device,
    )
    baseline_root = work_root / "baseline-cache" / cache_key
    reused = evidence._valid_baseline_cache(
        baseline_root,
        expected_sequences=expected_sequences,
        cache_key=cache_key,
    )
    resumed_sequences = expected_sequences if reused else 0
    removed_files = 0
    if reused:
        (run_dir / "proposal-baseline-console.txt").write_text(
            f"Reusing verified proposal source cache: {baseline_root}\n",
            encoding="utf-8",
        )
    else:
        baseline_root.mkdir(parents=True, exist_ok=True)
        (baseline_root / "source_manifest.json").unlink(missing_ok=True)
        resumed_sequences, removed_files = _retain_safe_baseline_prefix(
            baseline_root,
            validated["image_root"],
        )
        evidence._run(
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
            ],
            log_path=run_dir / "proposal-baseline-console.txt",
        )

    proposal_dir = baseline_root / "proposals"
    prediction_dir = baseline_root / "predictions"
    proposal_digest, proposal_bytes, proposal_count = evidence._directory_digest(
        proposal_dir
    )
    prediction_digest, prediction_bytes, prediction_count = (
        evidence._directory_digest(prediction_dir)
    )
    if proposal_count != expected_sequences:
        raise ValueError(
            f"proposal bank covers {proposal_count} sequences, "
            f"expected {expected_sequences}"
        )
    if prediction_count != expected_sequences:
        raise ValueError(
            f"raw control covers {prediction_count} sequences, "
            f"expected {expected_sequences}"
        )
    if proposal_bytes <= 0 or prediction_bytes <= 0:
        raise ValueError("proposal bank or raw control is empty")

    manifest = {
        "schema": "raft-uav-multi-uav-lts-proposal-source-v2",
        "generated_at_utc": evidence._utc_now(),
        "cache_key": cache_key,
        "cache_reused": reused,
        "cache_resume_prefix_sequences": resumed_sequences,
        "cache_discarded_tail_files": removed_files,
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
    evidence._write_json(baseline_root / "source_manifest.json", manifest)
    evidence._write_json(run_dir / "proposal-source-manifest.json", manifest)
    return proposal_dir, prediction_dir, manifest


def _inputs_json_path(arguments: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parsed, _unknown = parser.parse_known_args(arguments)
    return parsed.inputs_json.expanduser().resolve()


def _image_root_from_inputs(arguments: Sequence[str]) -> Path:
    payload_path = _inputs_json_path(arguments)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    try:
        raw_path = payload["dataset"]["train_sequence_root"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{payload_path}: missing dataset.train_sequence_root"
        ) from exc
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            f"{payload_path}: invalid dataset.train_sequence_root"
        )
    image_root = Path(raw_path).expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"train sequence root is not a directory: {image_root}")
    return image_root


def main() -> int:
    evidence.CANDIDATES = (*evidence.CANDIDATES, *IMPROVED_CANDIDATES)
    evidence._baseline_cache_key = _proposal_source_cache_key
    evidence._prepare_baseline = _prepare_baseline_resumable
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        return evidence.main()
    image_root = _image_root_from_inputs(sys.argv[1:])

    def run_candidates(
        proposal_dir: Path,
        seed_dir: Path,
        *,
        run_dir: Path,
        expected_sequences: int,
    ) -> dict[str, Path]:
        return _run_candidates_with_native_dimensions(
            proposal_dir,
            seed_dir,
            image_root=image_root,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )

    evidence._run_candidates = run_candidates
    return evidence.main()


if __name__ == "__main__":
    raise SystemExit(main())
