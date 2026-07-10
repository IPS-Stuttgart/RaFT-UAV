# MMUAD branch-aware class-conditioned uncertainty

The branch-preserving reservoir intentionally keeps raw, static, dynamic,
source-translated, calibrated, and cross-sensor candidates alive.  A single
learned-sigma model can nevertheless assign similar uncertainty to candidates
whose branch semantics and calibration displacement imply very different error
statistics.

`candidate_branch_uncertainty` augments candidates with inference-safe numeric
features before fitting the maintained uncertainty regressor:

- raw/static/dynamic/translated/calibrated/merged branch flags;
- translation displacement from preserved original coordinates;
- frame branch count and branch/source-branch fractions;
- within-branch and within-source-branch score ranks and score gaps;
- soft interactions between all of these features and UAV class probabilities.

Training truth is used only to fit expected candidate distance.  Application on
validation/test requires only candidate rows and sequence-level class
probabilities.

## Train

Use out-of-fold training class probabilities:

```bash
python scripts/mmuad_candidate_branch_uncertainty.py train \
  --candidates-csv outputs/train_branch_reservoir.csv \
  --truth-csv challenge_meta/train_ref.csv \
  --class-probabilities-csv outputs/train_class_probabilities_oof.csv \
  --model-json outputs/branch_sigma/model.json \
  --features-csv outputs/branch_sigma/train_features.csv \
  --summary-json outputs/branch_sigma/train_summary.json \
  --model-type hist-gradient-boosting \
  --sigma-min-m 1 \
  --sigma-max-m 30
```

## Apply

```bash
python scripts/mmuad_candidate_branch_uncertainty.py apply \
  --candidates-csv outputs/validation_branch_reservoir.csv \
  --class-probabilities-csv outputs/validation_class_probabilities.csv \
  --model-json outputs/branch_sigma/model.json \
  --output-csv outputs/validation_branch_sigma_candidates.csv \
  --output-column predicted_sigma_m_branch_class
```

Use `predicted_sigma_m_branch_class` as the sigma column for the existing
risk-reservoir, pair-forward-backward, or robust multi-start mixture pipeline.
The clean ablation is the current learned sigma versus this branch-aware sigma
with every downstream setting frozen and selected on train CV.
