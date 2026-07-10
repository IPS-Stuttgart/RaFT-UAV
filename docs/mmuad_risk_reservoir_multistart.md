# MMUAD risk-reservoir multi-start mixture-MAP

This pipeline combines two inference-safe improvements for the current MMUAD candidate-assignment bottleneck:

1. learned candidate uncertainty influences which candidates survive into a bounded branch-preserving reservoir;
2. the robust candidate-mixture MAP solver is restarted from global, median, and branch-specific initial trajectories.

The default keeps reservoir scoring and mixture weighting separate. Reservoir selection uses

```text
logit(p_good) - uncertainty_weight * log(sigma / sigma_floor)
```

while mixture-MAP uses the original calibrated/ranker score together with learned sigma. This avoids counting sigma twice.

## Example

```bash
python scripts/mmuad_candidate_risk_reservoir_multistart.py \
  --candidate-csv raw=/path/to/raw_candidates.csv \
  --candidate-csv translated=/path/to/source_translated_candidates.csv \
  --candidate-csv dynamic=/path/to/dynamic_candidates.csv \
  --output-dir outputs/mmuad_risk_reservoir_multistart \
  --risk-score-column candidate_class_calibrated_score \
  --sigma-column predicted_sigma_m \
  --uncertainty-weight 1.0 \
  --global-top-n 20 \
  --per-source-top-n 3 \
  --per-branch-top-n 3 \
  --max-candidates-per-frame 40 \
  --mixture-score-column candidate_class_calibrated_score \
  --loss huber \
  --huber-delta 1.0 \
  --smoothness-weight 7200 \
  --iterations 5
```

For labeled train/public-validation diagnostics, add:

```bash
  --truth-csv /path/to/truth.csv \
  --oracle-top-k 1 \
  --oracle-top-k 3 \
  --oracle-top-k 5 \
  --oracle-top-k 10 \
  --oracle-top-k 20
```

Truth is never used to select the multi-start winner. It only adds pose and reservoir-oracle diagnostics.

## Outputs

The output directory contains:

- risk-scored full candidate pool;
- bounded risk-adjusted reservoir;
- selected standard mixture estimates, assignments, and iteration rows;
- multi-start ranking and initialization rows;
- optional reservoir top-K oracle recall tables;
- combined provenance JSON;
- optional official Track 5 CSV and ZIP.

## Recommended ablation

Compare these on train-selected settings and one frozen public-validation run:

1. original ranker reservoir + single-start mixture-MAP;
2. risk-adjusted reservoir + single-start mixture-MAP;
3. original reservoir + multi-start mixture-MAP;
4. risk-adjusted reservoir + multi-start mixture-MAP.

A gain in row 2 isolates top-K retention. A gain in row 3 isolates local initialization. A gain in row 4 indicates that both failure modes matter.
