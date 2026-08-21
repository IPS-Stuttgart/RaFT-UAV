#!/usr/bin/env python3
"""Prepare reproducible public inputs for the Multi-UAV LTS evidence workflow.

The script deliberately pins every external revision and verifies the large binary
artifacts before exposing paths to the workflow. It extracts only the image and
label portions of the official training archive and derives the allowed
first-frame seed labels deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ZENODO_RECORD_ID = 15_853_476
ZENODO_ARCHIVE_NAME = "Train.zip"
ZENODO_ARCHIVE_MD5 = "d888235326ea6c939bbdb65258b8b407"
EXPECTED_FRAME_COUNT = 77_293
EXPECTED_SEQUENCE_COUNT = 102

UPSTREAM_REPOSITORY = "https://github.com/wish44165/YOLOv12-BoT-SORT-ReID.git"
UPSTREAM_REVISION = "1dfce32f90842b3233e0bbe7a14a082f2386f933"

HF_REPOSITORY = "wish44165/YOLOv12-BoT-SORT-ReID"
HF_REVISION = "e677d81dac9909ddeabb6bc70ded5510ff4872aa"
HF_REID_CONFIG = "config.yaml"
HF_REID_MODEL = "model_0016.pth"
HF_REID_MODEL_SHA256 = "7f6e20a9082eb5fe100e005bd5d6f2bf9fde7d9aeff5575546ff3c530baf6363"
HF_REID_MODEL_SIZE = 411_018_623


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "RaFT-UAV-evidence-workflow/1.0"},
    )
    for attempt in range(1, 9):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == 8:
                raise
            time.sleep(min(5 * attempt, 30))
    raise AssertionError("unreachable")


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _verified(
    path: Path,
    *,
    algorithm: str,
    expected_digest: str,
    expected_size: int | None = None,
) -> bool:
    if not path.is_file():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return _digest(path, algorithm) == expected_digest


def _download(
    url: str,
    destination: Path,
    *,
    algorithm: str | None = None,
    expected_digest: str | None = None,
    expected_size: int | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        algorithm is not None
        and expected_digest is not None
        and _verified(
            destination,
            algorithm=algorithm,
            expected_digest=expected_digest,
            expected_size=expected_size,
        )
    ):
        print(f"Reusing verified artifact: {destination}", flush=True)
        return

    partial = destination.with_name(destination.name + ".part")
    if destination.exists():
        destination.unlink()
    if (
        partial.exists()
        and expected_size is not None
        and partial.stat().st_size >= expected_size
    ):
        partial.unlink()

    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable public artifact downloads")
    _run(
        [
            curl,
            "--location",
            "--fail",
            "--show-error",
            "--silent",
            "--retry",
            "8",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ]
    )
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"downloaded size mismatch for {destination.name}: "
            f"{partial.stat().st_size} != {expected_size}"
        )
    if algorithm is not None and expected_digest is not None:
        actual = _digest(partial, algorithm)
        if actual != expected_digest:
            raise RuntimeError(
                f"{algorithm} mismatch for {destination.name}: "
                f"{actual} != {expected_digest}"
            )
    partial.replace(destination)


def _archive_member_suffix(name: str, component: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    try:
        index = path.parts.index(component)
    except ValueError:
        return None
    suffix = PurePosixPath(*path.parts[index:])
    if any(part in {"", ".", ".."} for part in suffix.parts):
        raise RuntimeError(f"unsafe ZIP member: {name!r}")
    return suffix


def _count_public_inputs(extracted_root: Path) -> tuple[int, int, int]:
    labels = extracted_root / "TrainLabels"
    images = extracted_root / "TrainImages"
    label_count = (
        sum(
            path.is_file() and path.suffix.lower() == ".txt"
            for path in labels.iterdir()
        )
        if labels.is_dir()
        else 0
    )
    sequence_count = (
        sum(
            path.is_dir()
            and any(
                child.is_file() and child.suffix.lower() == ".jpg"
                for child in path.iterdir()
            )
            for path in images.iterdir()
        )
        if images.is_dir()
        else 0
    )
    frame_count = (
        sum(
            child.is_file() and child.suffix.lower() == ".jpg"
            for path in images.iterdir()
            if path.is_dir()
            for child in path.iterdir()
        )
        if images.is_dir()
        else 0
    )
    return label_count, sequence_count, frame_count


def _valid_public_inputs(
    extracted_root: Path,
    *,
    expected_sequences: int,
    expected_frames: int,
) -> bool:
    return _count_public_inputs(extracted_root) == (
        expected_sequences,
        expected_sequences,
        expected_frames,
    )


def _extract_public_inputs(
    archive: Path,
    extracted_root: Path,
    *,
    expected_sequences: int,
    expected_frames: int,
) -> None:
    if _valid_public_inputs(
        extracted_root,
        expected_sequences=expected_sequences,
        expected_frames=expected_frames,
    ):
        print(f"Reusing complete extracted dataset: {extracted_root}", flush=True)
        return

    temporary = extracted_root.with_name(extracted_root.name + f".tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)

    selected = 0
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if info.is_dir():
                continue
            suffix = _archive_member_suffix(info.filename, "TrainImages")
            if suffix is None:
                suffix = _archive_member_suffix(info.filename, "TrainLabels")
            if suffix is None:
                continue
            if suffix.parts[0] == "TrainImages" and suffix.suffix.lower() != ".jpg":
                continue
            if suffix.parts[0] == "TrainLabels" and suffix.suffix.lower() != ".txt":
                continue

            destination = temporary.joinpath(*suffix.parts)
            resolved = destination.resolve()
            if temporary.resolve() not in resolved.parents:
                raise RuntimeError(f"unsafe extraction target: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as reader, destination.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            selected += 1

    if selected == 0:
        raise RuntimeError("official archive contained no TrainImages/TrainLabels entries")
    if not _valid_public_inputs(
        temporary,
        expected_sequences=expected_sequences,
        expected_frames=expected_frames,
    ):
        counts = _count_public_inputs(temporary)
        raise RuntimeError(
            "extracted dataset failed integrity gate: "
            f"labels={counts[0]}, sequences={counts[1]}, frames={counts[2]}"
        )

    shutil.rmtree(extracted_root, ignore_errors=True)
    temporary.replace(extracted_root)


def _derive_first_frame_labels(
    truth_dir: Path,
    seed_dir: Path,
    *,
    expected_sequences: int,
) -> None:
    temporary = seed_dir.with_name(seed_dir.name + f".tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)

    truth_files = sorted(truth_dir.glob("*.txt"))
    if len(truth_files) != expected_sequences:
        raise RuntimeError(
            f"truth sequence count {len(truth_files)} != {expected_sequences}"
        )
    for source in truth_files:
        first_frame: list[str] = []
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split(",")]
            if not fields:
                continue
            try:
                frame_id = int(float(fields[0]))
            except ValueError as exc:
                raise RuntimeError(f"invalid frame id in {source}: {fields[0]!r}") from exc
            if frame_id == 1:
                first_frame.append(line)
        if not first_frame:
            raise RuntimeError(f"{source.name} has no frame-1 seed annotations")
        (temporary / source.name).write_text(
            "\n".join(first_frame) + "\n",
            encoding="utf-8",
        )

    shutil.rmtree(seed_dir, ignore_errors=True)
    temporary.replace(seed_dir)


def _prepare_upstream_checkout(work_root: Path) -> Path:
    checkout = work_root / "repos" / "YOLOv12-BoT-SORT-ReID"
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").is_dir():
        shutil.rmtree(checkout, ignore_errors=True)
        checkout.mkdir()
        _run(["git", "init"], cwd=checkout)
        _run(["git", "remote", "add", "origin", UPSTREAM_REPOSITORY], cwd=checkout)
    else:
        _run(["git", "remote", "set-url", "origin", UPSTREAM_REPOSITORY], cwd=checkout)

    _run(["git", "fetch", "--depth=1", "origin", UPSTREAM_REVISION], cwd=checkout)
    _run(["git", "checkout", "--detach", "--force", "FETCH_HEAD"], cwd=checkout)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        text=True,
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"upstream revision mismatch: {revision} != {UPSTREAM_REVISION}")
    return checkout


def _prepare_reid_assets(work_root: Path, upstream_checkout: Path) -> tuple[Path, Path]:
    asset_root = work_root / "assets" / f"hf-{HF_REVISION}"
    config_asset = asset_root / HF_REID_CONFIG
    model_asset = asset_root / HF_REID_MODEL
    base = f"https://huggingface.co/{HF_REPOSITORY}/resolve/{HF_REVISION}"

    _download(f"{base}/{HF_REID_CONFIG}?download=true", config_asset)
    if not config_asset.is_file() or config_asset.stat().st_size < 100:
        raise RuntimeError("downloaded ReID config is missing or implausibly small")
    _download(
        f"{base}/{HF_REID_MODEL}?download=true",
        model_asset,
        algorithm="sha256",
        expected_digest=HF_REID_MODEL_SHA256,
        expected_size=HF_REID_MODEL_SIZE,
    )

    log_dir = upstream_checkout / "BoT-SORT" / "logs" / "sbs_S50"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_link = log_dir / "config.yaml"
    model_link = log_dir / "model_0016.pth"
    for link, target in ((config_link, config_asset), (model_link, model_asset)):
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.resolve())
    return config_asset, model_asset


def _zenodo_archive(work_root: Path) -> tuple[Path, str, int]:
    metadata = _request_json(f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}")
    matches = [
        item for item in metadata.get("files", []) if item.get("key") == ZENODO_ARCHIVE_NAME
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {ZENODO_ARCHIVE_NAME!r} file in Zenodo record, "
            f"got {len(matches)}"
        )
    item = matches[0]
    checksum = str(item.get("checksum", ""))
    if checksum != f"md5:{ZENODO_ARCHIVE_MD5}":
        raise RuntimeError(
            f"Zenodo checksum changed: {checksum!r} != md5:{ZENODO_ARCHIVE_MD5}"
        )
    size = int(item["size"])
    links = item.get("links", {})
    url = links.get("self") or links.get("content")
    if not isinstance(url, str) or not url:
        raise RuntimeError("Zenodo metadata did not provide a file content URL")

    archive = work_root / "archives" / ZENODO_ARCHIVE_NAME
    _download(
        url,
        archive,
        algorithm="md5",
        expected_digest=ZENODO_ARCHIVE_MD5,
        expected_size=size,
    )
    return archive, url, size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--expected-sequence-count",
        type=int,
        default=EXPECTED_SEQUENCE_COUNT,
    )
    parser.add_argument(
        "--expected-frame-count",
        type=int,
        default=EXPECTED_FRAME_COUNT,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    work_root = args.work_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    archive, archive_url, archive_size = _zenodo_archive(work_root)
    extracted = work_root / "extracted"
    _extract_public_inputs(
        archive,
        extracted,
        expected_sequences=args.expected_sequence_count,
        expected_frames=args.expected_frame_count,
    )

    truth_dir = extracted / "TrainLabels"
    image_root = extracted / "TrainImages"
    seed_dir = extracted / "TrainLabels_FirstFrameOnly"
    _derive_first_frame_labels(
        truth_dir,
        seed_dir,
        expected_sequences=args.expected_sequence_count,
    )

    upstream_checkout = _prepare_upstream_checkout(work_root)
    config_asset, model_asset = _prepare_reid_assets(work_root, upstream_checkout)

    label_count, sequence_count, frame_count = _count_public_inputs(extracted)
    payload: dict[str, Any] = {
        "schema": "raft-uav-multi-uav-lts-public-inputs-v1",
        "work_root": str(work_root),
        "dataset": {
            "zenodo_record_id": ZENODO_RECORD_ID,
            "archive_name": ZENODO_ARCHIVE_NAME,
            "archive_url": archive_url,
            "archive_path": str(archive),
            "archive_size": archive_size,
            "archive_md5": ZENODO_ARCHIVE_MD5,
            "label_sequence_count": label_count,
            "image_sequence_count": sequence_count,
            "frame_count": frame_count,
            "truth_dir": str(truth_dir),
            "first_frame_label_dir": str(seed_dir),
            "train_sequence_root": str(image_root),
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "checkout": str(upstream_checkout),
        },
        "reid": {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "config_path": str(config_asset),
            "model_path": str(model_asset),
            "model_sha256": HF_REID_MODEL_SHA256,
            "model_size": HF_REID_MODEL_SIZE,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"bootstrap failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
