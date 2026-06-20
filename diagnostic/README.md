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
