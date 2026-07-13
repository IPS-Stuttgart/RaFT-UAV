# Entropy-adaptive MMUAD sequence-classifier fusion

The maintained train-safe fusion selects one global image weight by training-label
cross-validation. That is a strong baseline, but it assumes the relative quality
of image and non-image classifiers is constant across all sequences.

This experiment keeps a train-selected global image-weight prior and adapts it per
sequence using each modality's normalized predictive entropy. Lower entropy means
higher reliability. The effective image weight is

```text
(1 - adaptation_strength) * prior_image_weight
+ adaptation_strength * relative_image_reliability
```

The reliability ratio can be sharpened with `entropy_power`. Setting
`adaptation_strength=0` exactly recovers ordinary global linear fusion, so the OOF
selection grid can reject adaptation when it is not supported by training data.
No public-validation or hidden-test labels are used for selection.

## Inputs

The command consumes one out-of-fold probability table per modality for train-only
selection and one prediction probability table per modality for frozen inference.
Each table contains one row per sequence and these columns:

```text
sequence_id
predicted_probability_0
predicted_probability_1
predicted_probability_2
predicted_probability_3
```

## Example

```bash
python scripts/mmuad_sequence_classifier_entropy_fusion.py \
  --image-oof-probabilities outputs/image_oof.csv \
  --nonimage-oof-probabilities outputs/nonimage_oof.csv \
  --image-predict-probabilities outputs/image_public.csv \
  --nonimage-predict-probabilities outputs/nonimage_public.csv \
  --train-labels data/train_sequence_labels.csv \
  --output-dir outputs/mmuad_entropy_adaptive_fusion \
  --prior-image-weight-grid 0:1:0.1 \
  --adaptation-strength-grid 0:1:0.1 \
  --entropy-power-grid 0.5,1,2 \
  --selection-metric accuracy
```

The selected output includes per-sequence image/non-image entropy, reliability,
adaptive target weight, effective weight, fused class probabilities, and the final
predicted class. The CV summary reports accuracy, balanced accuracy, and log loss
for every configuration.

## Evaluation

Compare against the existing global-weight fusion using the same OOF folds and
base classifier models. Primary checks are train-OOF accuracy and held-out
classification accuracy. Log loss is a useful tie-break because it rewards
calibrated confidence and can distinguish configurations with identical hard
predictions.
