# MMUAD candidate-sigma calibration train-CV

The learned candidate-uncertainty model and hierarchical source/branch
calibration are both truth-free when applied to validation or test candidates.
The remaining calibration hyperparameters should also be chosen without reading
target truth.

`raft-uav-mmuad-candidate-sigma-calibration-train-cv` performs
leave-one-sequence-out selection on training candidates for:

- target quantile of `truth_distance_3d_m / predicted_sigma_m`;
- minimum rows required for source, branch, and source+branch groups;
- hierarchical shrinkage strength.

Each held-out sequence is scored with a 3-D Gaussian negative-log-likelihood
surrogate, one-sigma coverage, coverage error, normalized squared error, and
mean calibrated sigma. The best configuration is then refit on every supplied
training sequence and saved as the standard candidate-sigma calibration JSON.

```bash
raft-uav-mmuad-candidate-sigma-calibration-train-cv \
  --candidates-csv outputs/mmuad/train_candidates_with_predicted_sigma.csv \
  --truth-csv data/mmuad/train_truth.csv \
  --output-dir outputs/mmuad/sigma_calibration_train_cv \
  --target-quantile 0.5 \
  --target-quantile 0.68 \
  --target-quantile 0.8 \
  --min-group-rows 10 \
  --min-group-rows 20 \
  --min-group-rows 50 \
  --shrinkage-rows 0 \
  --shrinkage-rows 25 \
  --shrinkage-rows 50 \
  --shrinkage-rows 100
```

The output directory contains:

- `mmuad_candidate_sigma_calibration_train_selected.json` — frozen calibration
  for validation/test application;
- `mmuad_candidate_sigma_calibration_train_cv_selection.json` — selected
  hyperparameters and pooled metrics;
- `mmuad_candidate_sigma_calibration_train_cv_folds.csv` — held-out sequence
  diagnostics for every grid point;
- `mmuad_candidate_sigma_calibration_train_cv_summary.csv` — weighted pooled
  ranking of the grid.

Apply the selected calibration with the existing command:

```bash
raft-uav-mmuad-apply-candidate-sigma-calibration \
  --candidates-csv outputs/mmuad/validation_candidates_with_predicted_sigma.csv \
  --calibration-json \
    outputs/mmuad/sigma_calibration_train_cv/mmuad_candidate_sigma_calibration_train_selected.json \
  --output-csv outputs/mmuad/validation_candidates_calibrated.csv \
  --replace-covariance
```

The selection artifacts are train-only diagnostics. Public-validation or hidden
truth must not be used to rerank the grid.
