# MMUAD uncertainty-aware source/branch quota

The source/branch reservoir is primarily score driven. A candidate can therefore
be removed before mixture-MAP even when the train-learned uncertainty model gives
it a small predicted sigma. `candidate_uncertainty_quota` augments the reservoir
with the lowest-sigma candidates from each `(source, candidate_branch)` cell before
the final frame cap.

This is inference safe: only candidate metadata and predicted uncertainty are
used. Ground-truth columns are ignored.

## Suggested ablation

Keep the trained sigma model, Huber loss, temporal priors, smoothing, and all
other candidate-pool settings frozen. Compare:

1. score/source/branch reservoir only;
2. uncertainty quota `top_n=1`;
3. uncertainty quota `top_n=2` when the frame budget allows it.

Use train CV for selection. Report pose MSE together with top-10/top-20 oracle
recall and the fraction of final candidates selected by the uncertainty quota.
The intended benefit is recall: preserve a reliable moderate-score hypothesis
without replacing the existing high-score and spatial-diversity quotas.
