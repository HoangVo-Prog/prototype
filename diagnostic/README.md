# Cue-Swap Diagnostic

This package implements a matched empirical probe for text-based person search
gallery sensitivity. It tests whether representative frozen gallery-agnostic
TBPS retrievers exhibit reliable retrieval changes when the distribution of
visual cues among unlabeled distractors is perturbed using an off-the-shelf
CLIP cue scorer.

The cue scorer is used only for external cue-affinity scores and Cue Shift
validity. It is not treated as ground-truth cue annotation. The evaluated
retriever remains frozen and is used only for retrieval scores and the
hardness-matched control.

Example:

```bash
python diagnostic/run_cue_swap_diagnostic.py \
  --dataset RSTPReid \
  --split test \
  --retriever_name itself \
  --retriever_config config/itself_best.yaml \
  --retriever_checkpoint /path/to/best.pth \
  --cue_scorer off_the_shelf_clip \
  --clip_model_name ViT-B/16 \
  --output_dir outputs/diagnostic_rstp_itself \
  --gallery_size 500 \
  --dense_ratio 0.5 \
  --num_trials 3 \
  --score_mode fusion \
  --lambda_global 0.68 \
  --device cuda
```

This implementation supports the retrievers available for the current scope:
`trained_clip` and `itself`. IRRA is intentionally not wired here.

The trained-CLIP-style retriever path uses the same repository model wrapper in
global-only mode:

```bash
python diagnostic/run_cue_swap_diagnostic.py \
  --dataset RSTPReid \
  --retriever_name trained_clip \
  --retriever_config config/clip_best.yaml \
  --retriever_checkpoint /path/to/best.pth \
  --output_dir outputs/diagnostic_rstp_clip \
  --score_mode global
```

Required outputs include per-gallery results, paired cue-swap results,
hardness-matched control results, cue-swap minus control deltas, validity
counts, and cluster-bootstrap confidence intervals over query clusters.

## Cue Galleries And Hardness Audit

For each valid `(case_id, query_id, trial_id)`, the two Cue galleries share one
neutral distractor set. Both galleries contain the complete positive set for
the query identity; only the cue-dense distractor subset differs between
`a_dense` and `b_dense`.

The HM control approximately matches retriever-score difficulty. Residual
top-rank mismatch is audited using hardest-negative scores,
positive-negative margins, and a fixed tight-match robustness subset.
HM replaces the complete distractor set independently for each direction; it
does not replace only the cue-dense subset and it does not need to share
distractors across `hm_a` and `hm_b`.

Per-gallery results report:

```text
best_positive_score
max_negative_score
positive_negative_margin
negative_score_scale
```

Paired delta results additionally report signed and normalized Cue-versus-HM
hardness gaps plus:

```text
mean_normalized_max_negative_gap
tight_hardness_match
tight_hardness_z_tolerance
```

`mean_normalized_max_negative_gap` is the average of the A and B direction-wise
hardest-negative gaps on the query-specific normalized retriever-score scale.
Positive values mean the Cue galleries have harder maximum negatives on
average; negative values mean the HM galleries are harder.

Use:

```bash
--tight_hardness_z_tolerance 0.10
```

The default is `0.10`. The tight-hardness subset is a robustness analysis and
does not replace the primary Cue-minus-HM Top-1 flip result over all valid
trials. Runs also write:

```text
hardness_audit_with_ci.csv
tight_hardness_summary_with_ci.csv
residual_hardness_adjusted_summary_with_ci.csv
```

The hardness audit includes raw-score residual metrics and normalized rows:

```text
mean_signed_normalized_max_negative_gap
mean_max_abs_normalized_max_negative_gap
```

`residual_hardness_adjusted_summary_with_ci.csv` reports a simple paired linear
sensitivity adjustment:

```text
delta_r1_flip = alpha + beta * mean_normalized_max_negative_gap + error
```

`adjusted_delta_r1_flip_at_zero_gap` is the intercept `alpha`. The adjusted
intercept estimates the Cue-minus-HM Top-1 flip difference at zero signed
normalized residual hardness under a linear sensitivity model.
`hardness_slope_delta_per_z` is the fitted slope in delta-flip units per
normalized score unit. This is a robustness/sensitivity analysis; it does not
prove causal removal of hardness confounding, and it does not imply perfect
hardness matching.

Paper-facing interpretation:

> We report both the unadjusted Cue-minus-HM flip difference and its linearly
> adjusted estimate at zero normalized residual hardest-negative gap. A similar
> adjusted estimate indicates that the measured residual hardness imbalance
> does not account for the observed difference.

A large raw-to-adjusted change would instead indicate sensitivity and should be
reported directly. The adjustment does not remove every possible notion of
retrieval difficulty.

## Bootstrap Units

The diagnostic supports two cluster-bootstrap units:

Case-query-instance bootstrap:
Resamples each eligible `(case_id, query_id)` instance jointly with all of its
repeated trials. This preserves the original implementation's uncertainty unit.

Unique-query bootstrap:
Resamples each underlying `query_id` jointly with every associated cue case and
all repeated trials. This is the recommended primary analysis because it
accounts for dependence among cue cases sharing the same query text, target
identity, positive set, and retriever behavior.

The reported metrics remain micro-averaged over valid case-query trials. The
unique-query bootstrap changes the uncertainty estimate, not the full-sample
point-estimate weighting. It is not a query-macro average.

Use:

```bash
--bootstrap_unit unique_query
--bootstrap_unit case_query
--bootstrap_unit both
```

When `both` is selected, `summary_with_ci.csv` uses the unique-query bootstrap
as the primary result and the run also writes:

```text
summary_with_ci_unique_query.csv
summary_with_ci_case_query.csv
```
