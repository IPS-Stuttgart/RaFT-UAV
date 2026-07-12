# MMUAD class-probability calibration

Several MMUAD pose components consume sequence-level UAV-type probabilities rather than
only the final hard class label. Examples include branch-aware candidate uncertainty,
class-conditioned anchor reliability, and confidence-adaptive class conditioning.
Classifier accuracy alone does not guarantee that these probabilities are calibrated.
An overconfident wrong class can therefore alter candidate survival or uncertainty more
strongly than intended.

This module fits scalar temperature scaling on **train-only out-of-fold predictions**:

```text
p_cal(c) = softmax(log(p_raw(c)) / T)
```

The scalar temperature preserves the predicted class ordering while adjusting
confidence. No validation or test class labels are used when the frozen calibrator is
applied.

## Fit on train OOF predictions

The prediction CSV may contain `class_prob_0..3`, `image_class_prob_0..3`, or
`predicted_probability_0..3`. The labels CSV may be a two-column class map or an
official Track 5 result/reference file.

```bash
python scripts/mmuad_calibrate_class_probabilities.py fit \
  --predictions-csv outputs/train_oof_class_probabilities.csv \
  --labels-csv data/mmuad/train_reference.csv \
  --model-json outputs/class_probability_calibrator.json \
  --output-csv outputs/train_oof_class_probabilities_calibrated.csv \
  --summary-json outputs/class_probability_calibration_summary.json
```

The summary reports accuracy, negative log likelihood, multiclass Brier score, expected
calibration error, and mean confidence before and after calibration.

## Apply to public validation or test

```bash
python scripts/mmuad_calibrate_class_probabilities.py apply \
  --predictions-csv outputs/val_class_probabilities.csv \
  --model-json outputs/class_probability_calibrator.json \
  --output-csv outputs/val_class_probabilities_calibrated.csv
```

By default, calibrated values are written as `calibrated_class_prob_0..3`. To feed them
into code that already expects the original probability-column names, add
`--replace-probabilities`. The original values are retained as
`raw_class_prob_0..3`.

## Intended experiment

Select the probability-calibration protocol on train only, then compare the same frozen
pose pipeline with:

```text
raw class probabilities
calibrated class probabilities
hard predicted class
no class conditioning
```

Report pose MSE and P95 together with class NLL, Brier score, ECE, and the frequency with
which calibration changes effective anchor weights or candidate sigma. Temperature
scaling cannot repair an incorrect class ranking, but it can reduce the damage caused by
an overconfident wrong prediction while retaining the strong classification signal on
confident correct sequences.
