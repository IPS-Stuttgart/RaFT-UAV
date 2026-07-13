# MMUAD Track 5 trajectory-medoid ensemble

Row-wise weighted means, medians, and consensus clusters can combine incompatible
trajectory branches or change the contributing source at every timestamp. The
trajectory-medoid ensemble instead selects one submitted estimate trajectory per
sequence.

For candidate trajectory `i`, the selector computes the inference-safe objective

```text
sum_j weight_j * mean_t ||x_i(t) - x_j(t)|| / sum_j weight_j
```

using timestamps that are valid for both candidates. The lowest-scoring eligible
candidate is the weighted trajectory medoid. Because the output follows an actual
candidate stream, it preserves the selected pipeline's temporal and kinematic
coherence rather than synthesizing a new coordinate-wise path.

## Usage

The module provides a `python -m` CLI without adding another package entry point:

```bash
python -m raft_uav.mmuad.track5_trajectory_medoid_ensemble \
  --estimate-csv baseline=outputs/baseline_estimates.csv@1.0 \
  --estimate-csv calibrated=outputs/calibrated_estimates.csv@1.5 \
  --estimate-csv reservoir=outputs/reservoir_estimates.csv@1.0 \
  --template data/track5_template.csv \
  --class-map data/sequence_classes.csv \
  --output-dir outputs/trajectory_medoid \
  --require-leaderboard-ready
```

The default `--min-coverage-fraction 1.0` restricts selection to candidates valid
on every requested row of a sequence. If no candidate reaches the threshold, the
selector relaxes to the maximum-coverage set and records the relaxation. When the
chosen sequence candidate is invalid at an individual timestamp, the output uses
the highest-weight valid candidate for that row and marks
`trajectory_medoid_fallback=true`.

## Outputs

The output directory contains:

- `mmuad_track5_trajectory_medoid_estimates.csv`: selected Track 5 estimates and
  row-level provenance;
- `mmuad_track5_trajectory_medoid_diagnostics.csv`: per-sequence, per-candidate
  coverage, overlap, objective, and selection diagnostics;
- `mmaud_results.csv` and `ug2_submission.zip`: official upload artifacts;
- validation JSON/CSV and a manifest containing selected candidates and fallback
  counts.

## Evaluation

Treat the method as a non-oracle ablation. Select candidate weights and the minimum
coverage fraction on train/CV sequences, then compare against weighted mean,
weighted geometric median, row-wise consensus, and sequence-gated baselines. The
main hypothesis is that preserving one coherent candidate trajectory improves
sequences where row-wise aggregation creates branch hopping or implausible blended
motion.
