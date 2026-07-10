# MMUAD pair log-posterior mixture-MAP

The acceleration-aware pair-state forward-backward prior writes two candidate
columns:

```text
candidate_pair_forward_backward_score
candidate_pair_forward_backward_log_probability
```

The first is a posterior probability. The second is its log posterior.
Candidate-mixture MAP adds its configured score directly to a candidate log
weight. Consequently, using the posterior probability as that score compresses
strong preferences. A `0.9` versus `0.1` posterior contributes only `0.8` score
units, while the corresponding log posterior contributes `log(9) ~= 2.20`.

`candidate_pair_log_multistart` provides a controlled ablation that:

1. prefers the pair prior's existing log-probability column;
2. reconstructs missing log values from the posterior probability;
3. renormalizes each frame in log space;
4. uses the normalized log posterior as the robust mixture-MAP score;
5. retains learned candidate sigma, Huber residuals, trajectory smoothness, and
   branch-seeded multi-start inference.

Truth is optional and is used only for diagnostic metrics. It is not used to
construct the score, generate initializations, or select the winning restart.

## Run after the pair-prior pipeline

Use the pair-candidate CSV produced by
`candidate_risk_pair_multistart` or `candidate_pair_forward_backward`:

```bash
python scripts/mmuad_candidate_pair_log_multistart.py \
  --pair-candidates-csv \
    outputs/risk_pair/mmuad_risk_pair_multistart_pair_candidates.csv \
  --output-dir outputs/pair_log_multistart \
  --sigma-column predicted_sigma_m \
  --score-weight 1.0 \
  --temperature 1.0 \
  --sigma-log-weight 0.0 \
  --huber-delta 1.0 \
  --smoothness-weight 7200 \
  --iterations 5
```

For a public-validation diagnostic, add `--truth-csv`. For an upload artifact,
add the class map and official output paths:

```bash
python scripts/mmuad_candidate_pair_log_multistart.py \
  --pair-candidates-csv outputs/risk_pair/mmuad_risk_pair_multistart_pair_candidates.csv \
  --output-dir outputs/pair_log_multistart \
  --class-map outputs/classification/sequence_class_map.csv \
  --official-results-csv outputs/pair_log_multistart/mmaud_results.csv \
  --official-zip outputs/pair_log_multistart/ug2_submission.zip
```

## Recommended ablation

Keep the reservoir, pair-prior, sigma, Huber, smoothness, and multi-start
settings frozen. Compare only the score space:

```text
A: candidate_pair_forward_backward_score
B: candidate_pair_forward_backward_log_probability
```

Use train CV to select the pair-score weight or temperature. Then run public
validation once. A gain in B indicates that the temporal posterior was useful
but previously underweighted by probability-space insertion. A tie rules out
score-space compression as the next major assignment bottleneck.

## Outputs

```text
mmuad_pair_log_multistart_candidates.csv
mmuad_candidate_mixture_estimates.csv
mmuad_candidate_mixture_assignments.csv
mmuad_candidate_mixture_iterations.csv
mmuad_candidate_mixture_multistart_summary.csv
mmuad_candidate_mixture_multistart_initializations.csv
mmuad_pair_log_multistart_summary.json
```
