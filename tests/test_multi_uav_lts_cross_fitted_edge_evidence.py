from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _directory_digest(directory: Path) -> tuple[str, int, int]:
    hasher = hashlib.sha256()
    total = 0
    count = 0
    for path in sorted(directory.glob("*.txt")):
        data = path.read_bytes()
        name = path.name.encode()
        hasher.update(len(name).to_bytes(8, "big"))
        hasher.update(name)
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)
        total += len(data)
        count += 1
    return hasher.hexdigest(), total, count


def _load_script(monkeypatch: pytest.MonkeyPatch):
    improved = types.ModuleType("run_multi_uav_lts_improved_evidence")
    improved.IMPROVED_CANDIDATES = ()
    improved._run_candidates_with_native_dimensions = lambda *args, **kwargs: {}
    improved._sequence_resolution_groups = lambda *args, **kwargs: ()
    improved._inputs_json_path = lambda arguments: Path(arguments[0])
    improved.main = lambda: 0
    evidence = types.ModuleType("run_multi_uav_lts_public_evidence")
    evidence.CANDIDATES = ()
    evidence._load_json = lambda path: json.loads(path.read_text(encoding="utf-8"))
    evidence._write_json = lambda path, payload: (
        path.parent.mkdir(parents=True, exist_ok=True),
        path.write_text(json.dumps(payload), encoding="utf-8"),
    )[-1]
    evidence._sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    evidence._directory_digest = _directory_digest
    evidence._run = lambda *args, **kwargs: None
    evidence._available_results = lambda run_dir: {"metrics": {}, "tournament": None}

    fixed_population_cv = types.ModuleType(
        "raft_uav.multi_uav_lts.fixed_population_cv"
    )

    def build_stratified_folds(sequences, *, fold_count=5, seed=0):
        del seed
        folds = [[] for _ in range(fold_count)]
        by_prefix: dict[str, list[str]] = {}
        for sequence in sorted(sequences):
            by_prefix.setdefault(sequence.split("_", 1)[0], []).append(sequence)
        cursor = 0
        for group in by_prefix.values():
            for sequence in group:
                folds[cursor % fold_count].append(sequence)
                cursor += 1
        return tuple(tuple(sorted(fold)) for fold in folds)

    fixed_population_cv.build_stratified_folds = build_stratified_folds
    raft_uav = types.ModuleType("raft_uav")
    multi_uav_lts = types.ModuleType("raft_uav.multi_uav_lts")
    monkeypatch.setitem(sys.modules, improved.__name__, improved)
    monkeypatch.setitem(sys.modules, evidence.__name__, evidence)
    monkeypatch.setitem(sys.modules, raft_uav.__name__, raft_uav)
    monkeypatch.setitem(sys.modules, multi_uav_lts.__name__, multi_uav_lts)
    monkeypatch.setitem(sys.modules, fixed_population_cv.__name__, fixed_population_cv)

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_multi_uav_lts_cross_fitted_edge_evidence.py"
    )
    name = "run_multi_uav_lts_cross_fitted_edge_evidence_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_fold_definitions_are_complementary_and_cover_every_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(monkeypatch)
    sequences = (
        "BB_00",
        "BB_01",
        "C_00",
        "C_01",
        "T_00",
        "T_01",
    )

    folds = module._build_fold_definitions(sequences, fold_count=3, seed=7)

    heldout = [sequence for fold in folds for sequence in fold.heldout_sequences]
    assert sorted(heldout) == sorted(sequences)
    assert len(heldout) == len(set(heldout))
    for fold in folds:
        assert not set(fold.training_sequences) & set(fold.heldout_sequences)
        assert set(fold.training_sequences) | set(fold.heldout_sequences) == set(
            sequences
        )


def test_model_provenance_rejects_any_heldout_training_leakage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script(monkeypatch)
    fold = module.FoldDefinition(
        index=0,
        training_sequences=("C_00", "T_00"),
        heldout_sequences=("C_01",),
    )
    model_path = tmp_path / "model.json"
    summary_path = tmp_path / "summary.json"
    model_path.write_text(
        json.dumps(
            {
                "sequence_count": 2,
                "metadata": {"selected_sequences": ["C_00", "T_00"]},
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "selected_sequences": ["C_00", "T_00"],
                "selected_sequence_count": 2,
                "positive_candidate_edges": 5,
                "negative_candidate_edges": 7,
            }
        ),
        encoding="utf-8",
    )

    model = module._validate_model_provenance(model_path, summary_path, fold)
    assert model.training_sequences == fold.training_sequences

    model_path.write_text(
        json.dumps(
            {
                "sequence_count": 3,
                "metadata": {
                    "selected_sequences": ["C_00", "T_00", "C_01"]
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance"):
        module._validate_model_provenance(model_path, summary_path, fold)


def test_edge_model_placeholder_is_required_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script(monkeypatch)
    model_path = tmp_path / "edge.json"

    resolved = module._resolve_edge_model_arguments(
        ("--edge-model-json", module._EDGE_MODEL_TOKEN, "--edge-model-weight", "1"),
        model_path,
    )
    assert resolved[1] == str(model_path)

    with pytest.raises(ValueError, match="one edge-model placeholder"):
        module._resolve_edge_model_arguments(("--edge-model-weight", "1"), model_path)


def test_candidate_assembly_uses_fold_model_only_on_its_heldout_sequences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script(monkeypatch)
    folds = (
        module.FoldDefinition(0, ("T_00", "T_01"), ("C_00", "C_01")),
        module.FoldDefinition(1, ("C_00", "C_01"), ("T_00", "T_01")),
    )
    models = {}
    for fold in folds:
        model_path = tmp_path / f"model-{fold.index}.json"
        model_path.write_text("{}", encoding="utf-8")
        models[fold.index] = module.FoldModel(
            fold.index,
            model_path,
            tmp_path / f"summary-{fold.index}.json",
            f"digest-{fold.index}",
            fold.training_sequences,
            fold.heldout_sequences,
        )

    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def run_tracker_group(
        proposal_dir,
        seed_dir,
        *,
        output_dir,
        summary_path,
        log_path,
        dimensions,
        arguments,
        sequences,
    ):
        del proposal_dir, seed_dir, summary_path, log_path, dimensions
        output_dir.mkdir(parents=True, exist_ok=True)
        model_index = arguments.index("--edge-model-json") + 1
        calls.append((sequences, (arguments[model_index],)))
        for sequence in sequences:
            (output_dir / f"{sequence}.txt").write_text(
                "1,1,0,0,1,1,1,1,1\n", encoding="utf-8"
            )

    monkeypatch.setattr(module, "_run_tracker_group", run_tracker_group)
    output = module._materialize_out_of_fold_candidate(
        "graph_edge_oof",
        ("--edge-model-json", module._EDGE_MODEL_TOKEN),
        tmp_path / "proposals",
        tmp_path / "seeds",
        folds,
        models,
        (
            ((1920, 1080), ("C_00", "T_00")),
            ((1280, 720), ("C_01", "T_01")),
        ),
        run_dir=tmp_path / "run",
        expected_sequences=4,
    )

    assert sorted(path.stem for path in output.glob("*.txt")) == [
        "C_00",
        "C_01",
        "T_00",
        "T_01",
    ]
    used_models = {
        sequence: model_path
        for sequences, (model_path,) in calls
        for sequence in sequences
    }
    assert used_models["C_00"].endswith("model-0.json")
    assert used_models["C_01"].endswith("model-0.json")
    assert used_models["T_00"].endswith("model-1.json")
    assert used_models["T_01"].endswith("model-1.json")
    summary = json.loads(
        (output.parent / "cross-fitted-candidate-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["sequence_count"] == 4
    assert summary["fold_count"] == 2
