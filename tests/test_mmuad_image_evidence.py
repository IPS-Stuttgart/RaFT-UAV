from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

from raft_uav.mmuad.classification import (
    classify_sequences_from_features,
    sequence_features_from_files,
)
from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table
from raft_uav.mmuad.image_evidence import build_image_evidence, main as image_evidence_main
from raft_uav.mmuad.schema import CandidateFrame


def _write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width = pixels.shape[:2]

    def chunk(name: bytes, data: bytes) -> bytes:
        payload = name + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _toy_image_root(tmp_path: Path) -> Path:
    root = tmp_path / "mmuad"
    image_dir = root / "seq001" / "Image"
    flat = np.full((8, 8, 3), 64, dtype=np.uint8)
    checker = np.indices((8, 8)).sum(axis=0) % 2
    textured = np.where(checker[:, :, None] == 0, [20, 220, 40], [240, 30, 200]).astype(np.uint8)
    _write_png(image_dir / "0.0.png", flat)
    _write_png(image_dir / "1.0.png", textured)
    return root


def _truth_csv(tmp_path: Path) -> Path:
    path = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    ).to_csv(path, index=False)
    return path


def test_image_evidence_samples_official_timestamps_and_scores_frames(tmp_path: Path) -> None:
    result = build_image_evidence(
        _toy_image_root(tmp_path),
        truth_file=_truth_csv(tmp_path),
        max_frames_per_sequence=8,
        max_image_time_delta_s=0.1,
    )

    assert len(result.frame_features) == 2
    assert result.sequence_features.loc[0, "sequence_id"] == "seq001"
    assert result.sequence_features.loc[0, "image_sampled_frame_count"] == 2
    assert result.sequence_features.loc[0, "image_matched_target_fraction"] == 1.0
    assert result.sequence_features.loc[0, "image_feature_backend_resolved"] == "handcrafted"
    assert "image_center_objectness_score_mean" in result.sequence_features.columns
    assert "image_grad_orientation_bin_0_mean" in result.sequence_features.columns
    assert "image_embedding_r_bin_0_mean" in result.sequence_features.columns
    assert result.frame_features.sort_values("target_time_s")["image_objectness_score"].iloc[1] > (
        result.frame_features.sort_values("target_time_s")["image_objectness_score"].iloc[0]
    )


def test_image_evidence_auto_backend_falls_back_to_handcrafted_when_deep_stack_missing(
    tmp_path: Path,
) -> None:
    result = build_image_evidence(
        _toy_image_root(tmp_path),
        truth_file=_truth_csv(tmp_path),
        max_frames_per_sequence=8,
        max_image_time_delta_s=0.1,
        image_feature_backend="auto",
    )

    assert len(result.frame_features) == 2
    assert set(result.frame_features["image_feature_backend_requested"]) == {"auto"}
    assert set(result.frame_features["image_feature_backend_resolved"]) <= {
        "handcrafted",
        "torchvision-resnet18",
    }


def test_image_evidence_cli_writes_sequence_and_frame_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    status = image_evidence_main(
        [
            str(_toy_image_root(tmp_path)),
            "--truth-file",
            str(_truth_csv(tmp_path)),
            "--output-dir",
            str(output_dir),
            "--max-image-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_image_evidence.csv").exists()
    assert (output_dir / "mmuad_image_frame_evidence.csv").exists()


def test_sequence_classifier_consumes_image_evidence_feature_table(tmp_path: Path) -> None:
    train_features = tmp_path / "train_image_evidence.csv"
    predict_features = tmp_path / "predict_image_evidence.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "image_objectness_score_mean": [0.1, 0.9],
            "image_edge_score_mean": [0.05, 0.5],
        }
    ).to_csv(train_features, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqC"],
            "image_objectness_score_mean": [0.85],
            "image_edge_score_mean": [0.45],
        }
    ).to_csv(predict_features, index=False)

    result = classify_sequences_from_features(
        train_features=sequence_features_from_files([train_features]),
        train_labels={"seqA": "Mavic 3", "seqB": "M30"},
        predict_features=sequence_features_from_files([predict_features]),
        method="nearest-neighbor",
    )

    assert result.predictions.loc[0, "predicted_class"] == "M30"
    assert "image_objectness_score_mean" in result.metrics["feature_columns"]


def test_image_sequence_classifier_train_val_script_writes_fused_outputs(tmp_path: Path) -> None:
    script_main = _load_image_sequence_classifier_script_main()
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    _write_sequence_images(train_root, "seqTrain0", dark=True)
    _write_sequence_images(train_root, "seqTrain1", dark=False)
    _write_sequence_images(val_root, "seqVal0", dark=True)
    _write_sequence_images(val_root, "seqVal1", dark=False)
    train_labels = tmp_path / "train_labels.csv"
    val_labels = tmp_path / "val_labels.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seqTrain0", "seqTrain1"],
            "uav_type": ["0", "1"],
        }
    ).to_csv(train_labels, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqVal0", "seqVal1"],
            "uav_type": ["0", "1"],
        }
    ).to_csv(val_labels, index=False)
    nonimage_train = tmp_path / "nonimage_train.csv"
    nonimage_val = tmp_path / "nonimage_val.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seqTrain0", "seqTrain1"],
            "range_mean": [1.0, 9.0],
        }
    ).to_csv(nonimage_train, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqVal0", "seqVal1"],
            "range_mean": [1.2, 8.8],
        }
    ).to_csv(nonimage_val, index=False)
    output_dir = tmp_path / "out"

    status = script_main(
        [
            "--train-root",
            str(train_root),
            "--val-root",
            str(val_root),
            "--train-labels",
            str(train_labels),
            "--eval-labels",
            str(val_labels),
            "--output-dir",
            str(output_dir),
            "--non-image-train-feature-table",
            str(nonimage_train),
            "--non-image-val-feature-table",
            str(nonimage_val),
            "--method",
            "nearest-neighbor",
            "--max-frames-per-sequence",
            "2",
            "--max-image-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    summary = json.loads(
        (output_dir / "mmuad_image_sequence_classifier_summary.json").read_text(encoding="utf-8")
    )
    assert {row["model"] for row in summary["rows"]} == {"image", "nonimage", "fused"}
    fused = pd.read_csv(output_dir / "mmuad_fused_classifier_probabilities.csv")
    assert sorted(fused["sequence_id"].astype(str).tolist()) == ["seqVal0", "seqVal1"]
    assert {"predicted_probability_0", "predicted_probability_1", "predicted_class"}.issubset(
        fused.columns
    )
    assert (output_dir / "mmuad_image_classifier_probabilities.csv").exists()
    assert (output_dir / "mmuad_nonimage_classifier_probabilities.csv").exists()


def test_cluster_ranker_feature_table_uses_sequence_level_image_evidence() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqB"],
                "time_s": [0.0, 0.0],
                "source": ["lidar", "lidar"],
                "track_id": ["a", "b"],
                "x_m": [0.0, 10.0],
                "y_m": [0.0, 0.0],
                "z_m": [1.0, 1.0],
            }
        )
    )
    image_evidence = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "image_objectness_score_mean": [0.2, 0.8],
            "image_matched_target_fraction": [1.0, 0.5],
        }
    )

    features = build_cluster_feature_table(candidates, image_evidence=image_evidence)

    assert features["image_evidence_available"].tolist() == [1.0, 1.0]
    assert features["image_objectness_score_mean"].tolist() == [0.2, 0.8]


def _write_sequence_images(root: Path, sequence_id: str, *, dark: bool) -> None:
    image_dir = root / sequence_id / "Image"
    if dark:
        first = np.full((8, 8, 3), 32, dtype=np.uint8)
        second = np.full((8, 8, 3), 48, dtype=np.uint8)
    else:
        checker = np.indices((8, 8)).sum(axis=0) % 2
        first = np.where(checker[:, :, None] == 0, [220, 40, 40], [20, 210, 230]).astype(np.uint8)
        second = np.where(checker[:, :, None] == 0, [240, 220, 40], [30, 30, 220]).astype(
            np.uint8
        )
    _write_png(image_dir / "0.0.png", first)
    _write_png(image_dir / "1.0.png", second)


def _load_image_sequence_classifier_script_main():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "mmuad_image_sequence_classifier_train_val.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mmuad_image_sequence_classifier_train_val",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main
