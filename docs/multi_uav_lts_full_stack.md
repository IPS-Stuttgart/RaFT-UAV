# Full-stack Multi-UAV LTS experiment

This experiment implements the remaining high-value layers around the existing
metric-aware proposal-graph work. It does not replace the raw seeded BoT-SORT
control and it does not merge a candidate merely because it completed.

## Implemented layers

1. **Stride-4 P2 thermal specialist.** `tiny_p2_detector` trains a compact
   centre/offset/size detector. Its box objective contains a size-adaptive
   normalized Gaussian-Wasserstein term, with stronger weight below a 24-pixel
   geometric-mean side. It exports a permissive proposal bank rather than final
   identities.
2. **Track-conditioned high-resolution proposals.**
   `track_conditioned_proposals` extrapolates established tracks, forms
   covariance-inflated ROIs, upscales them, applies the P2 specialist, and maps
   all proposals back to native coordinates.
3. **Motion-compensated temporal ROI proposals.**
   `temporal_roi_proposals` phase-aligns neighboring ROI crops, forms a robust
   temporal median background, and extracts residual connected components.
4. **Robust IMM smoothing.** `imm_trajectory` combines CV, CA, and damped-motion
   models, applies Student-t measurement reweighting, and performs a backward
   RTS pass for centre and log-size states.
5. **Seeded global multi-scan association.** `seeded_multiscan` solves one
   sequence-level proposal path per supplied frame-one identity. Dual prices
   resolve shared-proposal conflicts. Unseeded identities require persistent
   late-birth paths.
6. **Tiny-thermal affinity.** `thermal_edge_model` trains on same-identity pairs
   and nearest hard negatives. Features combine normalized crop correlation,
   gradients, histogram/DCT differences, context rings, geometry, motion, and
   temporal gap.
7. **Observable mixture of experts.** `observable_expert_gate` uses only early
   image, seed, and raw-tracker statistics. It selects a complete sequence
   output only when the predicted gain's one-sided lower bound clears a margin;
   otherwise it uses raw.

## Leakage boundary

`scripts/run_multi_uav_lts_full_stack_evidence.py` uses scenario-prefix folds.
For fold `k`, both trainable components are fitted only on the other folds. P2,
ROI, temporal, multi-scan, thermal-affinity, and IMM outputs are then generated
for fold `k` exactly once. The expert gate is fitted again on complementary
sequence scores and applied only to its held-out fold.

The final candidates are:

- `raw_imm`;
- `multiscan`;
- `multiscan_thermal`;
- `multiscan_thermal_imm`;
- `observable_gate`.

All are passed to the repository's existing guarded tournament together with
raw. Mean held-out Codabench HOTA, paired bootstrap confidence, MOTA/IDF1 floors,
scenario guards, exact coverage, and raw fallback remain authoritative.

## Local entry point

```bash
PYTHONPATH=src python scripts/run_multi_uav_lts_full_stack_evidence.py \
  --image-root /data/TrainImages \
  --truth-dir /data/TrainLabels \
  --seed-dir /data/TrainLabels_FirstFrameOnly \
  --raw-dir /outputs/proposal_baseline/predictions \
  --base-proposal-dir /outputs/proposal_baseline/proposals \
  --fold-assignments /outputs/full_stack/folds.csv \
  --run-dir /outputs/full_stack \
  --device cuda:0 \
  --require-improvement
```

The runner is resumable at model and proposal-directory boundaries. A complete
run writes `full-stack-summary.json`, per-fold model/proposal evidence, normalized
sequence scores, cross-fitted gate choices, tournament evidence, and logs.
