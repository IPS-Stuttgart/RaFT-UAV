# Risk-aware MMUAD candidate-reservoir selection

Mean LOSO oracle-recall performance can prefer a reservoir offset configuration
that works on most training sequences but catastrophically removes the useful
candidate mode on one difficult sequence. That is a poor match to the current
MMUAD failure pattern, where a small number of sequences dominate pose MSE.

The risk-aware selector uses training folds only and minimizes

```text
risk_score = (1 - alpha) * mean_metric + alpha * tail_metric
```

where the tail metric is a configurable held-out sequence quantile. With
`alpha=0`, the command exactly recovers mean aggregate CV selection. With
`alpha>0`, it can accept a small average penalty to reduce catastrophic
candidate-pruning failures.

## Train-only selection

```bash
python scripts/mmuad_candidate_reservoir_risk_cv.py \
  --candidate raw=/path/train_raw.csv \
  --candidate dynamic=/path/train_dynamic.csv \
  --candidate translated=/path/train_translated.csv \
  --truth-csv /path/train_truth.csv \
  --branch-score-offset-grid raw=-0.5,0,0.5,1 \
  --branch-score-offset-grid dynamic=-0.5,0,0.5 \
  --branch-score-offset-grid translated=-0.5,0,0.5 \
  --selection-metric oracle_top5_3d_m_mse \
  --risk-aversion 0.5 \
  --tail-quantile 1 \
  --global-top-n 20 \
  --per-source-top-n 3 \
  --per-branch-top-n 3 \
  --max-candidates-per-frame 40 \
  --write-best-reservoir \
  --output-dir outputs/mmuad_reservoir_risk_cv
```

Suggested train-CV ablation:

```text
risk_aversion: 0, 0.25, 0.5, 0.75, 1
 tail_quantile: 0.75, 0.9, 1
selection metric: oracle_top3_3d_m_mse, oracle_top5_3d_m_mse,
                  oracle_top10_3d_m_mse
```

Prefer the smallest risk aversion that materially reduces the held-out tail
without causing a large mean regression. Freeze the selected config before any
public-validation or hidden-test run.

## Outputs

```text
mmuad_candidate_reservoir_risk_cv_selected_config.json
mmuad_candidate_reservoir_risk_cv_folds.csv
mmuad_candidate_reservoir_risk_cv_aggregate.csv
mmuad_candidate_reservoir_risk_cv_selected.csv  # with --write-best-reservoir
```

The aggregate CSV records mean, standard deviation, minimum, maximum, selected
tail quantile, and the final risk score for every offset-grid row.

## Inference use

The selected JSON contains only frozen branch/source score offsets and reservoir
settings. Apply those settings to validation/test candidates without reading
truth, then run the existing learned-sigma Huber mixture-MAP pipeline.

This method changes candidate-preservation policy, not the final smoother. It is
therefore targeted at the current MMUAD bottleneck: protecting deeper physical
hypotheses on the hard sequences while keeping the inference-time candidate
budget bounded.
