# Posterior-mass adaptive group top-K

The branch-preserving MMUAD pool has strongly frame-dependent ambiguity. A fixed
physical-hypothesis budget can waste slots on easy frames while truncating the
tail exactly where the ranker/uncertainty model is diffuse.

`candidate_mixture_group_mass_topk` converts the maintained group unary into a
framewise posterior and keeps the smallest number of groups whose cumulative
posterior reaches a target mass, bounded by train-selected minimum and maximum
budgets. The existing spatial-diversity selector determines which groups occupy
that budget, and grouped Huber mixture-MAP performs the final trajectory fit.

The budget uses no truth labels at inference time. Truth is optional and is
passed only to the standard downstream metric reporter.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_mass_topk.py \
  --candidates-csv outputs/mmuad_reservoir/candidates.csv \
  --initial-estimates-csv outputs/mmuad_baseline/estimates.csv \
  --output-dir outputs/mmuad_mass_group_topk \
  --min-group-top-k 3 \
  --max-group-top-k 20 \
  --target-posterior-mass 0.95 \
  --posterior-temperature 1.0 \
  --uniform-posterior-blend 0.02 \
  --max-siblings-per-group 2 \
  --diversity-weight 0.5 \
  --diversity-scale-m 5 \
  --score-column candidate_pair_forward_backward_score \
  --sigma-column predicted_sigma_m \
  --loss huber \
  --huber-delta 1 \
  --smoothness-weight 7200
```

## Train-CV ablation

Freeze the reservoir, candidate scores, learned sigma, group definition, Huber
loss, smoother, and classification features. Select only the adaptive-budget
parameters on train CV:

```text
min_group_top_k:          1, 3, 5
max_group_top_k:          10, 20, 30
target_posterior_mass:    0.80, 0.90, 0.95, 0.99
posterior_temperature:    0.5, 1.0, 2.0
uniform_posterior_blend:  0.00, 0.02, 0.05
```

Compare against fixed spatial group top-K using the same maximum budget. Report:

- train-CV and public-validation pose MSE;
- mean/p95 selected group budget;
- top-10/top-20/full-pool oracle recall;
- retained posterior mass and normalized entropy;
- runtime and candidate rows passed to mixture-MAP.

A useful result would improve pose or oracle recall while reducing the average
budget relative to fixed `group_top_k=20`. If the adaptive selector always
chooses the maximum budget, the unary posterior is not calibrated enough for
mass-based truncation and the method should remain an ablation.
