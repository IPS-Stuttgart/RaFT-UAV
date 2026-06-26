# MMUAD temporal-consensus train-CV selection

Temporal candidate consensus is truth-free when it is applied, but its speed
gate, distance scale, and score weights should not be tuned on public
validation or hidden-test truth.

`raft-uav-mmuad-temporal-consensus-train-cv` evaluates a grid on labeled
training sequences. For each leave-one-sequence-out fold it selects the best
configuration on the remaining sequences and reports performance on the held
out sequence. It then selects one final configuration using all supplied
training sequences.

```bash
raft-uav-mmuad-temporal-consensus-train-cv \
  --candidate-csv outputs/mmuad/train_candidates.csv \
  --truth-csv data/mmuad/train_truth.csv \
  --output-dir outputs/mmuad/temporal_consensus_train_cv \
  --max-speed-mps 40 \
  --max-speed-mps 70 \
  --distance-scale-m 3 \
  --distance-scale-m 8 \
  --base-score-weight 0 \
  --base-score-weight 0.25 \
  --bidirectional-bonus 0.5 \
  --bidirectional-bonus 1.0 \
  --write-selected-candidates
```

The selector writes:

- `mmuad_temporal_consensus_train_selected_config.json`
- `mmuad_temporal_consensus_train_cv_folds.csv`
- `mmuad_temporal_consensus_train_grid_summary.csv`
- optionally `mmuad_temporal_consensus_train_selected_candidates.csv`

The default selection metric is top-1 squared 3-D candidate error. Mean top-1
error and mean candidate regret are also available through
`--selection-metric`.

Apply the frozen configuration without target truth:

```bash
raft-uav-mmuad-apply-temporal-consensus-config \
  --candidate-csv outputs/mmuad/validation_candidates.csv \
  --config-json \
    outputs/mmuad/temporal_consensus_train_cv/mmuad_temporal_consensus_train_selected_config.json \
  --output-csv outputs/mmuad/validation_candidates_temporal.csv \
  --summary-json outputs/mmuad/validation_temporal_summary.json
```

Use the resulting `candidate_temporal_consensus_score` in the candidate
reservoir or stratified mixture-MAP pipeline. The LOSO tables are training
diagnostics; they are not a justification for reselecting the configuration on
validation or test labels.
