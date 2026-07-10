# MMUAD acceleration-aware pair-state prior

The ordinary MMUAD forward-backward prior is first-order: it scores candidate
transitions between adjacent frames. A sequence can therefore switch between
individually plausible candidates while producing an implausible velocity
change. The pair-state prior represents the hidden state as
`(candidate[t-1], candidate[t])` and adds a constant-velocity acceleration
factor over candidate triples.

The method stays soft and inference-safe:

- no truth is used when computing candidate posteriors;
- ranker/calibration score and learned candidate sigma form the unary term;
- distance, speed, source, branch, and track continuity form the first-order
  transition term;
- irregular-time acceleration forms the second-order term;
- exact pair-state forward-backward inference produces probabilities that sum to
  one inside every frame;
- the posterior can be passed directly to the maintained Huber mixture-MAP
  smoother.

## Example

```bash
python scripts/mmuad_candidate_pair_forward_backward.py \
  --candidate-csv outputs/mmuad_candidate_risk_reservoir/reservoir.csv \
  --output-csv outputs/mmuad_pair_fb/candidates.csv \
  --summary-json outputs/mmuad_pair_fb/summary.json \
  --score-column candidate_risk_adjusted_score \
  --sigma-column predicted_sigma_m \
  --transition-distance-std-m 2 \
  --transition-speed-std-mps 15 \
  --acceleration-std-mps2 20 \
  --max-acceleration-mps2 80 \
  --mixture-output-dir outputs/mmuad_pair_fb/mixture \
  --mixture-top-k 20 \
  --mixture-smoothness-weight 7200 \
  --mixture-huber-delta 1
```

For a local validation diagnostic, add `--mixture-truth-csv`. The truth file is
only passed to mixture-MAP metric reporting; it does not affect the pair-state
posterior.

## Primary outputs

- `candidates.csv`: original reservoir rows plus pair-state posterior, rank,
  entropy, effective candidate count, and minimum compatible acceleration;
- `summary.json`: configuration, posterior normalization checks, and provenance;
- optional standard mixture-MAP estimates, assignments, iteration table, and
  summary in `--mixture-output-dir`.

## Recommended comparison

Use exactly the same branch-preserving risk-adjusted reservoir and frozen Huber
mixture settings for:

1. ranker/risk score directly;
2. first-order forward-backward posterior;
3. acceleration-aware pair-state posterior.

The useful signal is lower train-CV and public-validation pose MSE without a
loss of full-pool/top-K oracle recall. A failure is also informative: it would
show that constant-velocity assignment is not the remaining source of the gap.

The pair-state computation is cubic in the candidates per frame. It is intended
for an already bounded reservoir, typically at most 20--40 candidates per
frame, rather than the unpruned raw point-cluster pool.
