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

For an end-to-end experiment, compare the reservoir oracle-recall artifacts
before launching the expensive mixture-MAP grid. Keep this feature as an
ablation until train-CV selection confirms that it improves top-K recall and
final pose MSE.
