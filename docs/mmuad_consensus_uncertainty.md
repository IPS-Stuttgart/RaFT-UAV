# MMUAD consensus-conditioned candidate uncertainty

The MMUAD candidate-mixture pipeline already supports learned per-candidate
uncertainty and cross-sensor branch-consensus scoring. This command combines the
two stages without using validation/test truth at inference time.

The wrapper first computes truth-free cross-sensor agreement and raw/calibrated
sibling diagnostics. Numeric `branch_consensus_*` columns are then exposed in
the `candidate_reservoir_consensus_*` namespace consumed by the existing
candidate-uncertainty model.

## Train

```bash
raft-uav-mmuad-consensus-uncertainty train \
  --candidates-csv /path/train_candidates.csv \
  --truth-csv /path/train_reference.csv \
  --model-json outputs/mmuad_consensus_sigma/model.json \
  --features-csv outputs/mmuad_consensus_sigma/train_features.csv \
  --augmented-candidates-csv outputs/mmuad_consensus_sigma/train_candidates_consensus.csv \
  --summary-json outputs/mmuad_consensus_sigma/train_summary.json \
  --model-type hist-gradient-boosting \
  --consensus-time-window-s 0.05 \
  --consensus-distance-gate-m 5
```

## Apply

```bash
raft-uav-mmuad-consensus-uncertainty apply \
  --candidates-csv /path/validation_or_test_candidates.csv \
  --model-json outputs/mmuad_consensus_sigma/model.json \
  --output-csv outputs/mmuad_consensus_sigma/scored_candidates.csv
```

The output contains the original candidates, branch-consensus diagnostics,
`candidate_reservoir_consensus_*` model features, and `predicted_sigma_m` for
candidate-mixture MAP smoothing.

Fit the uncertainty model on training sequences only. Consensus computation and
model application are truth-free, so the saved model can be reused on public
validation or hidden test candidates.
