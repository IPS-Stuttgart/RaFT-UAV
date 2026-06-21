from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "run_mmuad_train_to_val_experiment.py"
spec = importlib.util.spec_from_file_location("run_mmuad_train_to_val_experiment", MODULE_PATH)
assert spec is not None and spec.loader is not None
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def test_train_to_val_harness_dry_run_writes_manifest_without_running(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    train_root.mkdir()
    val_root.mkdir()
    train_reference = tmp_path / "train_reference.csv"
    val_reference = tmp_path / "val_reference.csv"
    train_reference.write_text("Sequence,Classification\nseq0001,1\n", encoding="utf-8")
    val_reference.write_text("Sequence,Classification\nseq0002,2\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    rc = harness.main(
        [
            "--train-root",
            str(train_root),
            "--val-root",
            str(val_root),
            "--train-reference",
            str(train_reference),
            "--val-reference",
            str(val_reference),
            "--output-dir",
            str(output_dir),
            "--python",
            "python",
            "--dry-run",
        ]
    )

    assert rc == 0
    manifest = output_dir / "mmuad_train_to_val_manifest.json"
    inventory = output_dir / "train_to_val_inventory.json"
    assert manifest.exists()
    assert inventory.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "train_sequence_classifier" in text
    assert "raft_uav.mmuad.cluster_ranker" in text
    assert "raft_uav.mmuad.run" in text
    assert "raft_uav.mmuad.track5_scorecard_cli" in text
    assert str(val_reference) in text


def test_train_to_val_harness_rejects_same_reference_file(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    train_root.mkdir()
    val_root.mkdir()
    reference = tmp_path / "reference.csv"
    reference.write_text("Sequence,Classification\nseq0001,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="train and validation references"):
        harness.main(
            [
                "--train-root",
                str(train_root),
                "--val-root",
                str(val_root),
                "--train-reference",
                str(reference),
                "--val-reference",
                str(reference),
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
