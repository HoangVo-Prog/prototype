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
tight_hardness_match
tight_hardness_z_tolerance
```

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
```

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
