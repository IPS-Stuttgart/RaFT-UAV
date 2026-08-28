#!/usr/bin/env python3
"""Run leakage-safe full-stack Multi-UAV LTS evidence.

The runner trains the P2 specialist and thermal affinity model on complementary
scenario folds, assembles exactly-once held-out candidates, cross-fits the
observable expert gate, and delegates final selection to the repository's
existing guarded tournament.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from raft_uav.multi_uav_lts._full_stack_io import image_index, prediction_files, read_rows, write_json, write_rows
from raft_uav.multi_uav_lts.imm_trajectory import ImmConfig, smooth_directory
from raft_uav.multi_uav_lts.observable_expert_gate import apply_gate, fit_gate, read_score_csv, sequence_features
from raft_uav.multi_uav_lts.seeded_multiscan import MultiScanConfig, associate_sequence
from raft_uav.multi_uav_lts.thermal_edge_model import ThermalModel, make_affinity, train_model
from raft_uav.multi_uav_lts.tiny_p2_detector import predict_sequences, train_detector
from raft_uav.multi_uav_lts.track_conditioned_proposals import RoiConfig, generate_track_conditioned
from raft_uav.multi_uav_lts.temporal_roi_proposals import TemporalRoiConfig, generate_temporal_roi

def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}); see {log}: {' '.join(command)}")

def read_folds(path: Path) -> dict[str, int]:
    out = {}
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            out[row.get('sequence') or row.get('sequence_name')] = int(row['fold'])
    if not out:
        raise ValueError('empty fold assignment')
    return out

def write_names(path: Path, names: list[str]) -> None:
    path.write_text('\n'.join(names) + '\n', encoding='utf-8')

def merge_fold(source: Path, destination: Path, names: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        file = source / f'{name}.txt'
        if not file.is_file():
            raise FileNotFoundError(file)
        target = destination / file.name
        payload = file.read_bytes()
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError(f'conflicting held-out prediction: {target}')
            continue
        target.write_bytes(payload)

def score_candidate(python: str, candidate: Path, truth: Path, output: Path, log: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run([python, '-m', 'raft_uav.multi_uav_lts.metrics', str(candidate), '--truth-dir', str(truth), '--output-json', str(output.with_suffix('.json')), '--sequence-summary-csv', str(output)], log)

def normalize_score_csv(files: dict[str, Path], output: Path) -> None:
    rows = []
    for candidate, path in files.items():
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                sequence = row.get('sequence') or row.get('sequence_name') or row.get('SEQ') or row.get('seq')
                if not sequence or sequence.upper() == 'COMBINED_SEQ':
                    continue
                value = None
                for key in ('CODABENCH_HOTA', 'codabench_hota', 'HOTA(0)', 'hota_0', 'HOTA'):
                    if key in row and row[key] not in ('', None):
                        value = float(row[key])
                        break
                if value is None:
                    for key, cell in row.items():
                        normalized = key.lower().replace('_', '').replace('(', '').replace(')', '')
                        if 'hota' in normalized and cell not in ('', None):
                            value = float(cell)
                            break
                if value is None:
                    raise ValueError(f'cannot find HOTA field in {path}: {sorted(row)}')
                rows.append({'sequence': Path(sequence).stem, 'candidate': candidate, 'score': value})
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence', 'candidate', 'score'])
        writer.writeheader()
        writer.writerows(rows)

def digest_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.resolve()).encode())
        if path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--truth-dir', type=Path, required=True)
    p.add_argument('--seed-dir', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--base-proposal-dir', type=Path, required=True)
    p.add_argument('--fold-assignments', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--device', default='cpu')
    p.add_argument('--p2-epochs', type=int, default=20)
    p.add_argument('--fold-count', type=int, default=5)
    p.add_argument('--bootstrap-samples', type=int, default=5000)
    p.add_argument('--require-improvement', action='store_true')
    return p.parse_args(argv)

def main(argv=None) -> int:
    a = parse_args(argv)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        'format': 'raft-uav-full-stack-contract-v1',
        'image_root': str(a.image_root.resolve()),
        'truth_dir': str(a.truth_dir.resolve()),
        'seed_dir': str(a.seed_dir.resolve()),
        'raw_dir': str(a.raw_dir.resolve()),
        'base_proposal_dir': str(a.base_proposal_dir.resolve()),
        'fold_digest': hashlib.sha256(a.fold_assignments.read_bytes()).hexdigest(),
        'runner_digest': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fold_count': a.fold_count,
        'p2_epochs': a.p2_epochs,
    }
    contract_path = a.run_dir / 'experiment-contract.json'
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding='utf-8'))
        if previous != contract:
            raise ValueError('run directory belongs to a different full-stack contract')
    else:
        write_json(contract_path, contract)
    folds = read_folds(a.fold_assignments)
    all_names = sorted(folds)
    observed = set((p.stem for p in prediction_files(a.raw_dir)))
    if set(all_names) != observed:
        raise ValueError(f'fold/raw sequence mismatch: folds={len(all_names)} raw={len(observed)}')
    candidate_names = ('raw_imm', 'multiscan', 'multiscan_thermal', 'multiscan_thermal_imm')
    candidate_dirs = {name: a.run_dir / 'candidates' / name for name in candidate_names}
    for path in candidate_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    progress = {'stage': 'folds', 'completed_folds': []}
    write_json(a.run_dir / 'progress.json', progress)
    for fold in range(a.fold_count):
        held = sorted((name for name, value in folds.items() if value == fold))
        train = sorted(set(all_names) - set(held))
        fold_dir = a.run_dir / 'folds' / f'fold-{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        held_file = fold_dir / 'heldout.txt'
        train_file = fold_dir / 'train.txt'
        write_names(held_file, held)
        write_names(train_file, train)
        checkpoint = fold_dir / 'tiny-p2.pt'
        thermal_path = fold_dir / 'thermal-affinity.json'
        if not checkpoint.is_file():
            result = train_detector(a.image_root, a.truth_dir, checkpoint, train, epochs=a.p2_epochs, device=a.device)
            write_json(fold_dir / 'p2-train.json', result)
        if not thermal_path.is_file():
            write_json(fold_dir / 'thermal-train.json', train_model(a.image_root, a.truth_dir, thermal_path, train))
        p2 = fold_dir / 'p2-proposals'
        roi = fold_dir / 'roi-p2-proposals'
        temporal = fold_dir / 'temporal-proposals'
        fused = fold_dir / 'fused-proposals'
        if len(list(p2.glob('*.txt'))) < len(held):
            write_json(fold_dir / 'p2-predict.json', predict_sequences(checkpoint, a.image_root, p2, held, a.device, 0.003, 500))
        if len(list(roi.glob('*.txt'))) < len(held):
            write_json(fold_dir / 'roi-predict.json', generate_track_conditioned(checkpoint, a.image_root, a.raw_dir, roi, held, RoiConfig(), a.device, 0.002, 60))
        if len(list(temporal.glob('*.txt'))) < len(held):
            write_json(fold_dir / 'temporal.json', generate_temporal_roi(a.image_root, a.raw_dir, temporal, held, TemporalRoiConfig()))
        for name in held:
            rows = []
            next_id = 1
            for source in (a.base_proposal_dir, p2, roi, temporal):
                file = source / f'{name}.txt'
                if not file.is_file():
                    continue
                for row in read_rows(file):
                    rows.append(type(row)(row.frame_id, next_id, row.x1, row.y1, row.width, row.height, row.confidence, row.class_id, row.visibility))
                    next_id += 1
            write_rows(fused / f'{name}.txt', rows)
        plain = fold_dir / 'multiscan'
        thermal = fold_dir / 'multiscan-thermal'
        model = ThermalModel.load(thermal_path)
        diagnostics = {}
        for name in held:
            proposals = read_rows(fused / f'{name}.txt')
            seeds = read_rows(a.seed_dir / f'{name}.txt')
            image_paths = image_index(a.image_root / name)
            plain_rows, plain_diag = associate_sequence(proposals, seeds, MultiScanConfig())
            write_rows(plain / f'{name}.txt', plain_rows)
            affinity = make_affinity(model, image_paths)
            thermal_rows, thermal_diag = associate_sequence(proposals, seeds, MultiScanConfig(affinity_weight=0.12), affinity)
            write_rows(thermal / f'{name}.txt', thermal_rows)
            diagnostics[name] = {'plain': plain_diag, 'thermal': thermal_diag}
        write_json(fold_dir / 'multiscan.json', diagnostics)
        raw_subset = fold_dir / 'raw'
        raw_subset.mkdir(exist_ok=True)
        for name in held:
            (raw_subset / f'{name}.txt').write_bytes((a.raw_dir / f'{name}.txt').read_bytes())
        raw_imm = fold_dir / 'raw-imm'
        thermal_imm = fold_dir / 'multiscan-thermal-imm'
        smooth_directory(raw_subset, raw_imm, held, ImmConfig())
        smooth_directory(thermal, thermal_imm, held, ImmConfig())
        merge_fold(raw_imm, candidate_dirs['raw_imm'], held)
        merge_fold(plain, candidate_dirs['multiscan'], held)
        merge_fold(thermal, candidate_dirs['multiscan_thermal'], held)
        merge_fold(thermal_imm, candidate_dirs['multiscan_thermal_imm'], held)
        progress['completed_folds'].append(fold)
        write_json(a.run_dir / 'progress.json', progress)
    for name, path in candidate_dirs.items():
        if len(prediction_files(path)) != len(all_names):
            raise ValueError(f'incomplete {name}')
    scores = {'raw': a.run_dir / 'scores' / 'raw.csv'}
    score_candidate(a.python, a.raw_dir, a.truth_dir, scores['raw'], a.run_dir / 'logs' / 'score-raw.log')
    for name, path in candidate_dirs.items():
        scores[name] = a.run_dir / 'scores' / f'{name}.csv'
        score_candidate(a.python, path, a.truth_dir, scores[name], a.run_dir / 'logs' / f'score-{name}.log')
    normalized = a.run_dir / 'gate-scores.csv'
    normalize_score_csv(scores, normalized)
    score_map = read_score_csv(normalized)
    gate_dir = a.run_dir / 'candidates' / 'observable_gate'
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate_evidence = {}
    for fold in range(a.fold_count):
        held = sorted((n for n, v in folds.items() if v == fold))
        train = sorted(set(all_names) - set(held))
        features = {n: sequence_features(n, a.image_root, a.seed_dir, a.raw_dir) for n in train}
        training_scores = {n: score_map[n] for n in train}
        model = fit_gate(features, training_scores)
        model_path = a.run_dir / 'folds' / f'fold-{fold}' / 'expert-gate.json'
        model.save(model_path)
        fold_out = a.run_dir / 'folds' / f'fold-{fold}' / 'gate-output'
        result = apply_gate(model, a.image_root, a.seed_dir, a.raw_dir, candidate_dirs, fold_out, held)
        merge_fold(fold_out, gate_dir, held)
        gate_evidence[str(fold)] = result['counts']
    write_json(a.run_dir / 'gate-evidence.json', gate_evidence)
    gate_score = a.run_dir / 'scores' / 'observable_gate.csv'
    score_candidate(a.python, gate_dir, a.truth_dir, gate_score, a.run_dir / 'logs' / 'score-observable-gate.log')
    tournament = a.run_dir / 'tournament'
    command = [a.python, '-m', 'raft_uav.multi_uav_lts.tournament', str(a.raw_dir), '--truth-dir', str(a.truth_dir), '--output-dir', str(tournament), '--fold-count', str(a.fold_count), '--bootstrap-samples', str(a.bootstrap_samples), '--expected-sequence-count', str(len(all_names))]
    for name, path in [*candidate_dirs.items(), ('observable_gate', gate_dir)]:
        command.extend(['--candidate', f'{name}={path}'])
    if a.require_improvement:
        command.append('--require-improvement')
    run(command, a.run_dir / 'logs' / 'tournament.log')
    selected = (tournament / 'selected_candidate.txt').read_text().strip() if (tournament / 'selected_candidate.txt').is_file() else 'unknown'
    summary = {'selected_candidate': selected, 'candidate_dirs': {k: str(v) for k, v in {**candidate_dirs, 'observable_gate': gate_dir}.items()}, 'fold_count': a.fold_count, 'sequence_count': len(all_names), 'source_digest': digest_paths([a.fold_assignments])}
    write_json(a.run_dir / 'full-stack-summary.json', summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
