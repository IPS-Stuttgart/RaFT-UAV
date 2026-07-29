"""Evaluate Multi-UAV LTS predictions with the official TrackEval metrics.

The competition files already use the MOTChallenge row layout.  This bridge
materializes TrackEval's expected directory structure, invokes a vendored
TrackEval checkout, and emits compact JSON containing HOTA, DetA, AssA, LocA,
MOTA, and IDF1 for every sequence and for the combined evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator
import zipfile

import numpy as np


@dataclass(frozen=True)
class LtsTrackEvalMetrics:
    hota: float
    deta: float
    assa: float
    loca: float
    mota: float
    idf1: float
    hota_by_alpha: tuple[float, ...]
    alpha: tuple[float, ...]


@dataclass(frozen=True)
class LtsTrackEvalReport:
    prediction_path: str
    truth_dir: str
    sequence_root: str | None
    trackeval_root: str
    tracker_name: str
    sequence_count: int
    combined: LtsTrackEvalMetrics
    sequences: dict[str, LtsTrackEvalMetrics]


class TrackEvalBridgeError(RuntimeError):
    """Raised when TrackEval inputs or outputs do not match the expected schema."""


def _parse_int_like(value: str, *, location: str) -> int:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise TrackEvalBridgeError(f"{location}: expected integer-like value") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise TrackEvalBridgeError(f"{location}: expected integer-like value")
    return int(parsed)


def _validate_mot_text(text: str, *, source: str) -> int:
    """Validate one LTS/MOT text file and return its maximum frame id."""

    maximum_frame = 0
    seen_keys: set[tuple[int, int]] = set()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        location = f"{source}:{line_number}"
        if len(parts) != 9:
            raise TrackEvalBridgeError(
                f"{location}: expected 9 MOT columns, got {len(parts)}"
            )
        frame_id = _parse_int_like(parts[0], location=f"{location}:frame_id")
        object_id = _parse_int_like(parts[1], location=f"{location}:object_id")
        class_id = _parse_int_like(parts[7], location=f"{location}:class_id")
        try:
            x1, y1, width, height, confidence, visibility = (
                float(parts[index]) for index in (2, 3, 4, 5, 6, 8)
            )
        except ValueError as exc:
            raise TrackEvalBridgeError(f"{location}: invalid numeric field") from exc
        if frame_id <= 0 or object_id <= 0:
            raise TrackEvalBridgeError(
                f"{location}: frame and object ids must be positive"
            )
        if class_id <= 0:
            raise TrackEvalBridgeError(f"{location}: class id must be positive")
        if not all(
            math.isfinite(value)
            for value in (x1, y1, width, height, confidence, visibility)
        ):
            raise TrackEvalBridgeError(f"{location}: numeric fields must be finite")
        if width <= 0.0 or height <= 0.0:
            raise TrackEvalBridgeError(f"{location}: box dimensions must be positive")
        key = (frame_id, object_id)
        if key in seen_keys:
            raise TrackEvalBridgeError(
                f"{location}: duplicate frame/object pair {key}"
            )
        seen_keys.add(key)
        maximum_frame = max(maximum_frame, frame_id)
    return maximum_frame


def _prediction_texts(prediction_path: Path) -> dict[str, str]:
    path = Path(prediction_path)
    if path.is_dir():
        return {
            file.name: file.read_text(encoding="utf-8")
            for file in sorted(path.glob("*.txt"))
        }
    if not path.is_file():
        raise FileNotFoundError(f"prediction path does not exist: {path}")
    if not zipfile.is_zipfile(path):
        raise TrackEvalBridgeError(
            f"prediction path must be a directory or ZIP archive: {path}"
        )
    with zipfile.ZipFile(path) as archive:
        physical_names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(physical_names) != len(set(physical_names)):
            raise TrackEvalBridgeError("prediction ZIP contains duplicate member names")
        nested = [name for name in physical_names if "/" in name or "\\" in name]
        if nested:
            raise TrackEvalBridgeError(
                "prediction ZIP must contain root-level files only: " + ", ".join(nested)
            )
        return {
            name: archive.read(name).decode("utf-8")
            for name in sorted(physical_names)
            if name.endswith(".txt")
        }


def _selected_truth(
    truth_dir: Path,
    sequences: list[str] | None,
) -> dict[str, tuple[str, int]]:
    root = Path(truth_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"truth directory does not exist: {root}")
    requested = set(sequences or [])
    truth: dict[str, tuple[str, int]] = {}
    for path in sorted(root.glob("*.txt")):
        sequence = path.stem
        if requested and sequence not in requested:
            continue
        text = path.read_text(encoding="utf-8")
        maximum_frame = _validate_mot_text(text, source=str(path))
        if maximum_frame <= 0:
            raise TrackEvalBridgeError(f"truth file contains no detections: {path}")
        truth[sequence] = (text, maximum_frame)
    missing = sorted(requested - set(truth))
    if missing:
        raise TrackEvalBridgeError(
            "requested truth sequences are missing: " + ", ".join(missing)
        )
    if not truth:
        raise TrackEvalBridgeError("no truth sequences selected")
    return truth


def _apply_sequence_frame_counts(
    truth: dict[str, tuple[str, int]],
    sequence_root: Path | None,
) -> dict[str, tuple[str, int]]:
    if sequence_root is None:
        return truth
    root = Path(sequence_root)
    if not root.is_dir():
        raise FileNotFoundError(f"sequence root does not exist: {root}")
    updated: dict[str, tuple[str, int]] = {}
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for sequence, (text, truth_maximum) in truth.items():
        sequence_dir = root / sequence
        if not sequence_dir.is_dir():
            raise TrackEvalBridgeError(
                f"sequence image directory is missing: {sequence_dir}"
            )
        frame_count = sum(
            1
            for path in sequence_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_suffixes
        )
        if frame_count <= 0:
            raise TrackEvalBridgeError(
                f"sequence image directory contains no frames: {sequence_dir}"
            )
        if frame_count < truth_maximum:
            raise TrackEvalBridgeError(
                f"{sequence}: image count {frame_count} is below truth frame "
                f"{truth_maximum}"
            )
        updated[sequence] = (text, frame_count)
    return updated


def _install_numpy_legacy_aliases() -> None:
    """Provide aliases required by pinned TrackEval revisions on modern NumPy."""

    aliases = {
        "bool": bool,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _safe_tracker_name(value: str) -> str:
    name = value.strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("tracker_name must be a non-empty path component")
    return name


def _materialize_trackeval_layout(
    *,
    truth: dict[str, tuple[str, int]],
    predictions: dict[str, str],
    root: Path,
    tracker_name: str,
) -> tuple[Path, Path, dict[str, int]]:
    gt_root = root / "gt"
    trackers_root = root / "trackers"
    tracker_data = trackers_root / tracker_name / "data"
    tracker_data.mkdir(parents=True, exist_ok=True)
    sequence_info: dict[str, int] = {}
    for sequence, (truth_text, frame_count) in truth.items():
        gt_path = gt_root / sequence / "gt" / "gt.txt"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(truth_text, encoding="utf-8")

        prediction_text = predictions.get(f"{sequence}.txt", "")
        prediction_maximum = _validate_mot_text(
            prediction_text,
            source=f"{sequence}.txt",
        )
        if prediction_maximum > frame_count:
            raise TrackEvalBridgeError(
                f"{sequence}: prediction frame {prediction_maximum} exceeds "
                f"truth frame {frame_count}"
            )
        (tracker_data / f"{sequence}.txt").write_text(
            prediction_text,
            encoding="utf-8",
        )
        sequence_info[sequence] = frame_count
    return gt_root, trackers_root, sequence_info


def _resolve_trackeval_root(path: Path) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "trackeval" / "__init__.py").is_file():
        return root
    if root.name == "trackeval" and (root / "__init__.py").is_file():
        return root.parent
    raise FileNotFoundError(
        f"TrackEval package not found under {root}; expected trackeval/__init__.py"
    )


@contextmanager
def _trackeval_import(root: Path) -> Iterator[Any]:
    root = _resolve_trackeval_root(root)
    _install_numpy_legacy_aliases()
    existing = sys.modules.get("trackeval")
    if existing is not None:
        module_path = Path(getattr(existing, "__file__", "")).resolve()
        if root not in module_path.parents:
            raise TrackEvalBridgeError(
                f"trackeval is already imported from a different checkout: {module_path}"
            )
        yield existing
        return

    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("trackeval")
        yield module
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


def _mean_array(value: Any, *, field: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise TrackEvalBridgeError(f"TrackEval returned malformed {field}")
    return float(np.mean(array))


def _finite_scalar(value: Any, *, field: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0 or not np.isfinite(array.item()):
        raise TrackEvalBridgeError(f"TrackEval returned malformed {field}")
    return float(array.item())


def _extract_metrics(node: dict[str, Any], alpha: np.ndarray) -> LtsTrackEvalMetrics:
    try:
        hota_node = node["HOTA"]
        clear_node = node["CLEAR"]
        identity_node = node["Identity"]
        hota_by_alpha = np.asarray(hota_node["HOTA"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackEvalBridgeError("TrackEval result schema is incomplete") from exc
    if hota_by_alpha.shape != alpha.shape or not np.all(np.isfinite(hota_by_alpha)):
        raise TrackEvalBridgeError("TrackEval HOTA alpha grid has an unexpected shape")
    return LtsTrackEvalMetrics(
        hota=float(np.mean(hota_by_alpha)),
        deta=_mean_array(hota_node["DetA"], field="DetA"),
        assa=_mean_array(hota_node["AssA"], field="AssA"),
        loca=_mean_array(hota_node["LocA"], field="LocA"),
        mota=_finite_scalar(clear_node["MOTA"], field="MOTA"),
        idf1=_finite_scalar(identity_node["IDF1"], field="IDF1"),
        hota_by_alpha=tuple(float(value) for value in hota_by_alpha),
        alpha=tuple(float(value) for value in alpha),
    )


def evaluate_lts_with_trackeval(
    prediction_path: Path,
    truth_dir: Path,
    *,
    trackeval_root: Path,
    tracker_name: str = "raft_uav",
    sequences: list[str] | None = None,
    sequence_root: Path | None = None,
    work_dir: Path | None = None,
    use_parallel: bool = False,
    num_parallel_cores: int = 8,
) -> LtsTrackEvalReport:
    """Run TrackEval and return combined and per-sequence competition metrics."""

    tracker_name = _safe_tracker_name(tracker_name)
    if num_parallel_cores <= 0:
        raise ValueError("num_parallel_cores must be positive")
    truth = _selected_truth(Path(truth_dir), sequences)
    truth = _apply_sequence_frame_counts(truth, sequence_root)
    predictions = _prediction_texts(Path(prediction_path))
    resolved_trackeval_root = _resolve_trackeval_root(Path(trackeval_root))

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="raft-uav-trackeval-")
        materialization_root = Path(temporary.name)
    else:
        materialization_root = Path(work_dir).expanduser().resolve()
        if materialization_root.exists() and any(materialization_root.iterdir()):
            raise TrackEvalBridgeError(
                f"work directory must be empty: {materialization_root}"
            )
        materialization_root.mkdir(parents=True, exist_ok=True)

    try:
        gt_root, trackers_root, sequence_info = _materialize_trackeval_layout(
            truth=truth,
            predictions=predictions,
            root=materialization_root,
            tracker_name=tracker_name,
        )
        with _trackeval_import(resolved_trackeval_root) as trackeval:
            eval_config = trackeval.Evaluator.get_default_eval_config()
            eval_config.update(
                {
                    "USE_PARALLEL": bool(use_parallel),
                    "NUM_PARALLEL_CORES": int(num_parallel_cores),
                    "BREAK_ON_ERROR": True,
                    "PRINT_RESULTS": False,
                    "PRINT_ONLY_COMBINED": False,
                    "PRINT_CONFIG": False,
                    "TIME_PROGRESS": False,
                    "DISPLAY_LESS_PROGRESS": True,
                    "OUTPUT_SUMMARY": False,
                    "OUTPUT_DETAILED": False,
                    "PLOT_CURVES": False,
                }
            )
            dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            dataset_config.update(
                {
                    "GT_FOLDER": str(gt_root),
                    "TRACKERS_FOLDER": str(trackers_root),
                    "OUTPUT_FOLDER": str(materialization_root / "output"),
                    "TRACKERS_TO_EVAL": [tracker_name],
                    "TRACKER_DISPLAY_NAMES": [tracker_name],
                    "CLASSES_TO_EVAL": ["pedestrian"],
                    "BENCHMARK": "MultiUAVLTS",
                    "SPLIT_TO_EVAL": "train",
                    "INPUT_AS_ZIP": False,
                    "DO_PREPROC": False,
                    "TRACKER_SUB_FOLDER": "data",
                    "OUTPUT_SUB_FOLDER": "",
                    "SEQ_INFO": sequence_info,
                    "SKIP_SPLIT_FOL": True,
                    "PRINT_CONFIG": False,
                }
            )
            metric_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
            metrics = [
                trackeval.metrics.HOTA(metric_config),
                trackeval.metrics.CLEAR(metric_config),
                trackeval.metrics.Identity(metric_config),
            ]
            dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
            evaluator = trackeval.Evaluator(eval_config)
            output_res, output_msg = evaluator.evaluate([dataset], metrics)

            dataset_name = dataset.get_name()
            message = output_msg.get(dataset_name, {}).get(tracker_name)
            if message != "Success":
                raise TrackEvalBridgeError(
                    f"TrackEval failed for {tracker_name}: {message or 'unknown error'}"
                )
            try:
                raw = output_res[dataset_name][tracker_name]
                combined_node = raw["COMBINED_SEQ"]["pedestrian"]
            except (KeyError, TypeError) as exc:
                raise TrackEvalBridgeError("TrackEval returned an incomplete result tree") from exc
            alpha = np.arange(0.05, 0.99, 0.05, dtype=float)
            per_sequence = {
                sequence: _extract_metrics(raw[sequence]["pedestrian"], alpha)
                for sequence in sorted(sequence_info)
            }
            combined = _extract_metrics(combined_node, alpha)
    finally:
        if temporary is not None:
            temporary.cleanup()

    return LtsTrackEvalReport(
        prediction_path=str(Path(prediction_path).expanduser().resolve()),
        truth_dir=str(Path(truth_dir).expanduser().resolve()),
        sequence_root=(
            None
            if sequence_root is None
            else str(Path(sequence_root).expanduser().resolve())
        ),
        trackeval_root=str(resolved_trackeval_root),
        tracker_name=tracker_name,
        sequence_count=len(per_sequence),
        combined=combined,
        sequences=per_sequence,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("truth_dir", type=Path)
    parser.add_argument(
        "--trackeval-root",
        type=Path,
        default=os.environ.get("RAFT_UAV_TRACKEVAL_ROOT"),
        required=os.environ.get("RAFT_UAV_TRACKEVAL_ROOT") is None,
    )
    parser.add_argument("--tracker-name", default="raft_uav")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument(
        "--sequence-root",
        type=Path,
        help="image-sequence root used to recover full sequence frame counts",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num-parallel-cores", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    report = evaluate_lts_with_trackeval(
        args.prediction_path,
        args.truth_dir,
        trackeval_root=args.trackeval_root,
        tracker_name=args.tracker_name,
        sequences=args.sequences,
        sequence_root=args.sequence_root,
        work_dir=args.work_dir,
        use_parallel=args.parallel,
        num_parallel_cores=args.num_parallel_cores,
    )
    payload = asdict(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["combined"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
