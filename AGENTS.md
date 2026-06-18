# AGENTS.md

## Project Goal

This repository is used for text-based person search (TBPS) research. The current task is to implement a diagnostic protocol for a proposed paradigm called **Transductive Text-Based Person Search (T-TBPS)**.

The protocol is called **Cue-Swap Counterfactual Gallery Evaluation**. It constructs two counterfactual target galleries for the same text query and the same positive identity, but changes the distribution of two human-defined textual cues. The purpose is to measure whether TBPS performance depends on the unlabeled target gallery composition.

## Required Behavior

When implementing this task:

1. Do not change the training pipeline unless absolutely necessary.
2. Prefer adding one standalone evaluation script.
3. The script must load an existing best checkpoint.
4. The script must use the repository's existing dataset/model/evaluation utilities whenever possible.
5. The script must work on the test split, especially RSTPReid, but should be written generally enough for CUHK-PEDES or ICFG-PEDES if the repository supports them.
6. Identity labels may only be used for:

   * selecting eligible positive images for a query;
   * excluding same-identity images from distractor pools during synthetic gallery construction;
   * computing retrieval metrics.
7. Identity labels must never be used by the model during inference.
8. Cue-image matching must be label-free. Use the loaded model's text encoder and image encoder when possible.
9. The code must be deterministic under a fixed seed.
10. Save all outputs to disk in CSV/JSONL format.

## Preferred Implementation

Create a new file:

```text
tools/cue_swap_counterfactual_eval.py
```

The script should support

```bash
python tools/cue_swap_counterfactual_eval.py \
  --dataset RSTPReid \
  --split test \
  --checkpoint /path/to/best_checkpoint.pth \
  --config /path/to/config.yaml \
  --cases_file /path/to/cue_cases.json \
  --output_dir outputs/cue_swap_rstp \
  --gallery_size 500 \
  --dense_ratio 0.5 \
  --num_random_trials 3 \
  --seed 42 \
  --device cuda
```

The implementation should inspect the repository and adapt to the existing APIs for:

* dataset loading;
* model construction;
* checkpoint loading;
* text/image feature extraction;
* retrieval scoring;
* metric computation.

If there are multiple possible APIs, choose the most stable existing evaluation path and wrap it cleanly.

## Cue Case Input Format

The script must accept a JSON file with a list of cases. Example:

```json
[
  {
    "case_id": "black_jacket_bag",
    "cue_a": "black jacket",
    "cue_b": "bag",
    "query_include_all": ["black", "jacket", "bag"],
    "max_queries": 50
  },
  {
    "case_id": "red_shirt_backpack",
    "cue_a": "red shirt",
    "cue_b": "backpack",
    "query_include_all": ["red", "shirt", "backpack"],
    "max_queries": 50
  }
]
```

The query selection must be automatic. For each case, find test queries whose normalized text contains all strings in `query_include_all`. If `query_include_all` is missing, fall back to requiring the query text to contain the normalized tokens from both `cue_a` and `cue_b`.

Optional fields that should be supported if easy:

```json
{
  "query_ids": [1, 2, 3],
  "query_regex": "black.*jacket.*bag|bag.*black.*jacket",
  "min_queries": 10
}
```

If `query_ids` is provided, use those query IDs directly after validating that they exist in the test split and have positives in the gallery.

## Counterfactual Gallery Construction

For each eligible query `q` and cue pair `(cue_a, cue_b)`:

1. Get the query identity label `pid_q`.
2. Find all positive gallery images with the same identity label.
3. Always include the positive images in both galleries.
4. Exclude same-identity images from distractor candidates.
5. Encode `cue_a` and `cue_b` with the model text encoder.
6. Encode all test gallery images with the model image encoder.
7. Compute label-free cue-image affinity:

```text
psi_a(x) = cosine(text_emb(cue_a), image_emb(x))
psi_b(x) = cosine(text_emb(cue_b), image_emb(x))
```

8. Construct two galleries of equal size:

```text
G_a_dense: positives + distractors where cue_a is common and cue_b is less common
G_b_dense: positives + distractors where cue_b is common and cue_a is less common
```

Use contrastive cue scores:

```text
score_a_dense(x) = psi_a(x) - lambda_contrast * psi_b(x)
score_b_dense(x) = psi_b(x) - lambda_contrast * psi_a(x)
```

Default:

```text
lambda_contrast = 0.5
```

Let:

```text
num_pos = number of positive images
num_distractors = gallery_size - num_pos
num_dense = round(dense_ratio * num_distractors)
num_neutral = num_distractors - num_dense
```

For `G_a_dense`:

* choose `num_dense` distractors with highest `score_a_dense`;
* choose `num_neutral` filler distractors from the remaining images, preferably with low max cue affinity or random negatives;
* avoid duplicates.

For `G_b_dense`:

* choose `num_dense` distractors with highest `score_b_dense`;
* choose `num_neutral` filler distractors similarly;
* avoid duplicates.

Both galleries must:

* have the same number of images;
* include the same positives;
* not include duplicate image IDs;
* not include same-identity distractors;
* be reproducible under the seed.

If there are not enough valid distractors, skip that query and log the reason.

## Metrics

For each model/checkpoint and each constructed gallery:

Compute standard retrieval metrics:

```text
R@1
R@5
R@10
mAP
min_positive_rank
AP
```

For each query pair, also compute diagnostic metrics:

```text
rank_volatility = abs(min_positive_rank_a_dense - min_positive_rank_b_dense)
ap_delta = AP_a_dense - AP_b_dense
r1_flip = whether R@1 differs between the two counterfactual galleries
```

Compute cue-density diagnostics for both galleries:

Hard density:

```text
D_hard(cue, G) = mean[psi(cue, x) > threshold_cue]
```

Soft density:

```text
D_soft(cue, G) = mean[sigmoid((psi(cue, x) - threshold_cue) / tau_density)]
```

The threshold can be a quantile over the full test gallery, default 75th percentile. Default `tau_density = 0.02`.

Compute cue-swap strength:

```text
swap_strength =
  (D_soft(cue_a, G_a_dense) - D_soft(cue_a, G_b_dense))
+ (D_soft(cue_b, G_b_dense) - D_soft(cue_b, G_a_dense))
```

Compute optional host crowding score:

```text
crowding(q, G) = logsumexp(sim(q, x) / tau_crowding) over non-positive gallery images
```

Default `tau_crowding = 0.07`.

## Output Files

The script must save:

```text
output_dir/
  config_used.json
  selected_queries.csv
  per_query_results.csv
  summary_by_case.csv
  summary_overall.csv
  galleries.jsonl
  skipped_queries.jsonl
```

`per_query_results.csv` should include at least:

```text
case_id
query_id
query_text
pid
cue_a
cue_b
gallery_type
gallery_size
num_positives
R1
R5
R10
AP
min_positive_rank
D_soft_a
D_soft_b
D_hard_a
D_hard_b
crowding
swap_strength
seed
trial_id
```

`summary_by_case.csv` should aggregate metrics by:

```text
case_id
gallery_type
num_queries
R1
R5
R10
mAP
mean_min_positive_rank
mean_crowding
mean_D_soft_a
mean_D_soft_b
mean_swap_strength
```

`summary_overall.csv` should include:

```text
num_cases
num_queries_total
R1_a_dense
R1_b_dense
R5_a_dense
R5_b_dense
R10_a_dense
R10_b_dense
mAP_a_dense
mAP_b_dense
mean_rank_volatility
r1_flip_rate
mean_swap_strength
```

## Code Quality Requirements

* Use argparse.
* Use pathlib.
* Use pandas for CSV output.
* Use torch.no_grad for feature extraction.
* Normalize all embeddings before cosine similarity.
* Cache image embeddings and query embeddings in memory.
* Add helpful logging.
* Add clear error messages for missing fields.
* Do not silently fail.
* Keep the script self-contained except for importing existing repo modules.
* Add comments where repository-specific adaptation is needed.
* At the end of the script, print a concise summary table.

## Validation

Before finishing, ensure:

1. The script imports successfully.
2. `--help` works.
3. The cue case file is validated.
4. The output directory is created.
5. A dry-run mode works if implemented:

```bash
python tools/cue_swap_counterfactual_eval.py \
  --dataset RSTPReid \
  --split test \
  --checkpoint /path/to/best_checkpoint.pth \
  --cases_file examples/cue_cases_example.json \
  --output_dir outputs/debug_cue_swap \
  --gallery_size 100 \
  --max_queries_per_case 3 \
  --dry_run
```

## Scientific Intent

This diagnostic is not meant to replace the full-gallery benchmark. It is meant to show that the same query and same positive identity can produce different retrieval outcomes under different unlabeled target-gallery compositions. This supports the T-TBPS formulation:

```text
standard TBPS:      score(x | q)
transductive TBPS:  score(x | q, G)
```

The final implementation should make this claim measurable.
