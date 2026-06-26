# MMUAD branch-consensus candidate scoring

The branch-preserving MMUAD pipeline can keep raw, dynamic, calibrated, and
merged candidate hypotheses alive until reservoir selection and trajectory
optimization. Those branches may carry different score distributions, so a
single global ranker ordering can bury a candidate that is independently
supported by another sensor.

`raft-uav-mmuad-candidate-branch-consensus` adds non-oracle features for this
case. For every candidate it finds nearby candidates from different sensor
sources, records spatial and temporal agreement, counts independent source and
branch support, and compares raw/calibrated siblings that share the same
`mmuad_calibration_origin_row`.

The command does not use truth:

```bash
raft-uav-mmuad-candidate-branch-consensus \
  --candidate-csv outputs/mmuad/source_calibration_branch_union.csv \
  --output-csv outputs/mmuad/source_calibration_branch_union_consensus.csv \
  --provenance-json outputs/mmuad/branch_consensus.json \
  --time-window-s 0.05 \
  --distance-gate-m 5 \
  --distance-scale-m 5
```

The principal output score is `branch_consensus_rank_score`. It combines a
within-source/branch normalized base score, cross-sensor agreement, and a small
raw-versus-calibrated sibling preference. Raw and calibrated copies from the
same sensor do not count as independent support.

Use the score directly in the branch-aware reservoir:

```bash
raft-uav-mmuad-candidate-reservoir \
  --candidate-csv consensus=outputs/mmuad/source_calibration_branch_union_consensus.csv \
  --output-csv outputs/mmuad/branch_consensus_reservoir.csv \
  --score-column branch_consensus_rank_score \
  --global-top-n 20 \
  --per-source-top-n 3 \
  --per-branch-top-n 3 \
  --max-candidates-per-frame 40
```

## Consensus-aware uncertainty

The consensus-conditioned uncertainty model can now use cross-sensor agreement
in two complementary ways:

1. numeric `branch_consensus_*` features are available to the train-fitted sigma
   model; and
2. an optional monotonic sigma shrinkage can make independently supported
   candidates more precise without replacing or reordering the ranker score.

Train the model on the training split as before, then apply the optional shrinkage
at validation/test time:

```bash
raft-uav-mmuad-consensus-uncertainty apply \
  --candidates-csv outputs/mmuad/validation_branch_union.csv \
  --model-json outputs/mmuad/consensus_uncertainty.json \
  --output-csv outputs/mmuad/validation_consensus_uncertainty.csv \
  --consensus-sigma-weight 0.5 \
  --consensus-sigma-min-factor 0.5
```

For a consensus score in `[0, 1]`, the multiplicative factor is
`max(min_factor, exp(-weight * score))`. A score of zero leaves the learned
sigma unchanged. When enabled, the output also records `raw_predicted_sigma_m`
and `candidate_uncertainty_consensus_factor` for ablation and provenance.
Select the weight and minimum factor on training folds before validation/test.

For an end-to-end experiment, compare the reservoir oracle-recall artifacts
before launching the expensive mixture-MAP grid. Keep this feature as an
ablation until train-CV selection confirms that it improves top-K recall and
final pose MSE.
