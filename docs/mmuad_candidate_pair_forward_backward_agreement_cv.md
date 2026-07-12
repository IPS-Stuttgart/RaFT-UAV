# Train-CV selection for the agreement-adaptive pair prior

The agreement-adaptive pair-state prior has several blend controls:

```text
min_pair_weight
max_pair_weight
entropy_power
agreement_power
agreement_floor
```

These controls affect candidate assignment and must be selected on training
sequences only. Public-validation or hidden-test truth must not be used to tune
them.

The selector in
`raft_uav.mmuad.candidate_pair_forward_backward_agreement_cv` evaluates each
configuration per training sequence, runs the learned-sigma Huber
candidate-mixture smoother, and ranks configurations with

```text
risk_score = (1 - risk_aversion) * mean_metric
             + risk_aversion * tail_metric
```

The tail metric is a configurable sequence quantile. This avoids selecting a
configuration that has a good average but catastrophically follows a smooth
wrong mode on one difficult sequence.

Pair-state and local posteriors are computed once per sequence. Grid rows only
reblend these fixed posteriors, which is substantially cheaper than rerunning
pair-state inference for every blend configuration.

## Coarse train grid

```bash
python scripts/mmuad_candidate_pair_forward_backward_agreement_cv.py \
  --candidate-csv outputs/train_selected_reservoir.csv \
  --truth-csv data/mmuad/train_truth.csv \
  --output-dir outputs/mmuad_agreement_pair_cv \
  --score-column candidate_reservoir_grid_score \
  --sigma-column predicted_sigma_m \
  --min-pair-weight 0 \
  --min-pair-weight 0.1 \
  --max-pair-weight 0.75 \
  --max-pair-weight 1 \
  --entropy-power 1 \
  --entropy-power 2 \
  --agreement-power 0.5 \
  --agreement-power 1 \
  --agreement-floor 0 \
  --agreement-floor 0.1 \
  --selection-metric mse_3d_m \
  --risk-aversion 0.25 \
  --tail-quantile 0.9 \
  --mixture-top-k 20 \
  --mixture-smoothness-weight 7200 \
  --mixture-huber-delta 1
```

The defaults are the same 32-row coarse grid. Refine only around the selected
train configuration.

## Outputs

```text
mmuad_agreement_pair_cv_folds.csv
mmuad_agreement_pair_cv_aggregate.csv
mmuad_agreement_pair_cv_selected_config.json
mmuad_agreement_pair_cv_selected_candidates.csv
mmuad_agreement_pair_cv_selected_summary.json
```

`selected_candidates.csv` already contains the frozen adaptive posterior and
can be passed to the existing candidate-mixture MAP runner. The JSON records the
pair model, mixture configuration, selected blend, train sequence IDs, mean and
tail metrics, and whether truth was used. Truth is used for train-CV selection
only and never for constructing inference-time candidate priors.
