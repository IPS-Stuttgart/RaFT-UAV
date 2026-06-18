from __future__ import annotations

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
    assert result.frame_features.sort_values("target_time_s")["image_objectness_score"].iloc[1] > (
        result.frame_features.sort_values("target_time_s")["image_objectness_score"].iloc[0]
    )


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
