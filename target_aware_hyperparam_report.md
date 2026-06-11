# Target-Aware Hyperparameter Impact Analysis

Primary metric: `best_R1`. Secondary metrics were not used to override `best_R1` conclusions.

## 1. Data Inventory

| CSV | Rows | Columns |
| --- | --- | --- |
| wandb_export_2026-06-02T21_26_54.842+07_00.csv | 70 | 500 |
| wandb_export_2026-06-02T21_21_26.432+07_00.csv | 55 | 353 |

- Combined/de-duplicated runs: **124**. De-dup key: W&B `Name` when present, else `ID`.
- Valid runs: **107**.
- Excluded runs: **17**.
- Crashed runs: **8**.
- Running runs: **1**.
- Finished but unfinished epoch: **14**.
- Missing `best_R1`: **5**.
- Suspiciously short runtime: **8** using <25% of median runtime for the same `num_epoch`.

### Excluded Runs
| Name | ID | State | best_R1 | epoch/num_epoch | Reason |
| --- | --- | --- | --- | --- | --- |
| 20260531_234602 | nan | crashed | 62.000 | 17/30 | crashed; unfinished epoch 17/30; unsupported current context_pooling |
| 20260601_004419 | nan | crashed |  | 1/60 | crashed; missing best_R1; unfinished epoch 1/60 |
| 20260601_154924 | nan | crashed | 62.000 | 2/30 | crashed; unfinished epoch 2/30 |
| 20260601_155441 | nan | crashed |  | 1/30 | crashed; missing best_R1; unfinished epoch 1/30 |
| 20260601_201912 | nan | crashed | 62.300 | 24/60 | crashed; unfinished epoch 24/60 |
| 20260601_202231 | nan | crashed | 61.950 | 27/30 | crashed; unfinished epoch 27/30 |
| 20260602_122810 | rlvbcgnv | crashed | 61.550 | 5/30 | crashed; unfinished epoch 5/30 |
| 20260602_122957 | zaqineri | crashed |  | 1/30 | crashed; missing best_R1; unfinished epoch 1/30 |
| 20260531_203417 | nan | finished | 62.550 | 30/30 | unsupported current context_pooling |
| 20260531_213524 | nan | finished | 62.000 | 30/30 | unsupported current context_pooling |
| 20260531_220028 | nan | finished | 62.450 | 30/30 | unsupported current context_pooling |
| 20260601_155320 | nan | finished |  | 0/30 | missing best_R1; unfinished epoch 0/30 |
| 20260601_215047 | 7biqos12 | finished | 61.350 | 1/30 | unfinished epoch 1/30 |
| 20260602_102148 | zhqobwv6 | finished | 61.900 | 27/30 | unfinished epoch 27/30 |
| 20260602_144509 | 775ielml | finished |  | 0/30 | missing best_R1; unfinished epoch 0/30 |
| 20260602_202445 | po2nm7xq | finished | 61.850 | 27/30 | unfinished epoch 27/30 |
| 20260601_214942 | r5cr1u6r | running | 61.500 | 3/30 | running; unfinished epoch 3/30 |

### Suspiciously Short Runtime
| Name | State | best_R1 | runtime_sec | status |
| --- | --- | --- | --- | --- |
| 20260601_155320 | finished |  | 50 | missing best_R1; unfinished epoch 0/30 |
| 20260602_144509 | finished |  | 68 | missing best_R1; unfinished epoch 0/30 |
| 20260602_122957 | crashed |  | 90 | crashed; missing best_R1; unfinished epoch 1/30 |
| 20260601_155441 | crashed |  | 90 | crashed; missing best_R1; unfinished epoch 1/30 |
| 20260601_215047 | finished | 61.350 | 109.00 | unfinished epoch 1/30 |
| 20260601_004419 | crashed |  | 121.00 | crashed; missing best_R1; unfinished epoch 1/60 |
| 20260602_122810 | crashed | 61.550 | 210.00 | crashed; unfinished epoch 5/30 |
| 20260601_154924 | crashed | 62.000 | 211.00 | crashed; unfinished epoch 2/30 |

### Detected CLI Parameters From `utils/options.py`
| Group | Canonical parameter | Default | Choices | Aliases | Validation |
| --- | --- | --- | --- | --- | --- |
| target-aware text enrichment | target_enrichment | False |  |  |  |
| target-aware text enrichment | enrichment_start | 1 |  |  | >= 1 |
| target-aware text enrichment | enrichment_space | global | global, grab |  |  |
| target-aware text enrichment | top_m | 32 |  |  |  |
| target-aware text enrichment | extractor_mode | global,horizontal | comma-separated: global, horizontal, vertical, grid | global_horizontal, global_vertical, global_grid | at least one supported mode; aliases expand and duplicates are removed |
| target-aware text enrichment | num_parts | 6 |  |  | >= 1 |
| target-aware text enrichment | use_freeze_indices | False |  | --freeze_indices | requires target_enrichment |
| target-aware text enrichment | pnp_text_only | False |  |  | requires freeze_host, no_use_host_loss, use_freeze_indices, enrichment_space=global |
| target-aware text enrichment | enrich_gamma | None |  |  | required only when residual_gate=static; forbidden when residual_gate=residual |
| target-aware text enrichment | residual_gate | residual | static, residual | --gate_mode |  |
| target-aware text enrichment | residual_gate_hidden_dim | 128 |  |  | >= 1 |
| target-aware text enrichment | recompute_level | epoch | epoch, step |  |  |
| target-aware text enrichment | recompute_interval | 1 |  | --pool_interval | -1 means compute once; otherwise refresh every N epochs/steps |
| mlp-mixer module settings | context_module | mixer | mixer |  |  |
| mlp-mixer module settings | mixer_dim | 256 |  |  | >= 1 |
| mlp-mixer module settings | mixer_depth | 2 |  |  | >= 1 |
| mlp-mixer module settings | mixer_hidden_part | 32 |  |  | >= 1 |
| mlp-mixer module settings | mixer_hidden_rank | 64 |  |  | >= 1 |
| mlp-mixer module settings | mixer_hidden_channel | 512 |  |  | >= 1 |
| mlp-mixer module settings | mixer_hidden_readout | 128 |  |  | >= 1 |
| mlp-mixer module settings | context_pooling | mlp | mlp | --mixer_context_pooling | fixed to mlp |
| target-aware loss settings | lambda_ret | 1 |  |  | > 0 |

- Varied in combined CSV: top_m, extractor_mode, num_parts, pnp_text_only, enrich_gamma, residual_gate, residual_gate_hidden_dim, mixer_dim, mixer_depth, mixer_hidden_part, mixer_hidden_rank, mixer_hidden_channel, mixer_hidden_readout, context_pooling, lambda_ret.
- Fixed in combined CSV: target_enrichment, enrichment_start, enrichment_space, use_freeze_indices, recompute_level, recompute_interval, context_module.
- Varied among valid runs: top_m, extractor_mode, num_parts, pnp_text_only, enrich_gamma, residual_gate, residual_gate_hidden_dim, mixer_dim, mixer_depth, mixer_hidden_part, mixer_hidden_rank, mixer_hidden_channel, mixer_hidden_readout, lambda_ret.
- Fixed among valid runs: target_enrichment, enrichment_start, enrichment_space, use_freeze_indices, recompute_level, recompute_interval, context_module, context_pooling.

## 2. Default Baseline And Noise

### Official Parser Default
| count | mean best_R1 | std | min | max | best default run |
| --- | --- | --- | --- | --- | --- |
| 0 |  |  |  |  |  |

### Operational Sweep Anchor
| count | mean best_R1 | std | min | max | best anchor run | Anchor differs from parser default by |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | 62.056 | 0.269 | 61.750 | 62.450 | 20260530_163955 | target_enrichment=True, num_parts=24, use_freeze_indices=True, pnp_text_only=True, enrich_gamma=0.1, residual_gate=static, recompute_interval=-1, mixer_hidden_rank=128, mixer_hidden_readout=256, lambda_ret=0.5 |

Official parser-default valid runs are absent. Downstream deltas use the **operational sweep anchor** because it is the measured baseline available in the CSV.
Baseline noise threshold used for conclusions: **0.269 best_R1** (official parser default absent; noise estimated from repeated operational-anchor runs with a conservative 0.25 floor).
Deltas smaller than this threshold are treated as weak or inconclusive.

## 3. Overall Ranking By `best_R1`

### Top 10 Valid Configurations
| Run | best_R1 | delta vs anchor mean | delta vs best anchor | #anchor diffs | Anchor-relative diffs |
| --- | --- | --- | --- | --- | --- |
| 20260531_183800 | 62.950 | 0.894 | 0.500 | 3 | top_m=64, extractor_mode=global,vertical, mixer_depth=1 |
| 20260601_122636 | 62.950 | 0.894 | 0.500 | 3 | top_m=64, extractor_mode=global,vertical, mixer_depth=1 |
| 20260601_173609 | 62.900 | 0.844 | 0.450 | 3 | extractor_mode=global,vertical, enrich_gamma=0.2, mixer_depth=1 |
| 20260601_091653 | 62.800 | 0.744 | 0.350 | 3 | top_m=128, extractor_mode=global,vertical, mixer_depth=1 |
| 20260601_175258 | 62.800 | 0.744 | 0.350 | 5 | num_parts=6, pnp_text_only=False, enrich_gamma=0.5, mixer_hidden_rank=64, mixer_hidden_readout=128 |
| 20260601_181525 | 62.750 | 0.694 | 0.300 | 5 | num_parts=6, pnp_text_only=False, enrich_gamma=1, mixer_hidden_rank=64, mixer_hidden_readout=128 |
| 20260531_122513 | 62.700 | 0.644 | 0.250 | 2 | top_m=64, extractor_mode=global,vertical |
| 20260601_124430 | 62.650 | 0.594 | 0.200 | 5 | num_parts=6, pnp_text_only=False, enrich_gamma=0.7, mixer_hidden_rank=64, mixer_hidden_readout=128 |
| 20260602_123348 | 62.650 | 0.594 | 0.200 | 9 | num_parts=6, enrich_gamma=None, residual_gate=residual, mixer_dim=128, mixer_depth=1, mixer_hidden_part=16, mixer_hidden_rank=32, mixer_hidden_channel=256, mixer_hidden_readout=64 |
| 20260601_151922 | 62.650 | 0.594 | 0.200 | 5 | num_parts=6, pnp_text_only=False, enrich_gamma=0.3, mixer_hidden_rank=64, mixer_hidden_readout=128 |

### Bottom 10 Valid Configurations
| Run | best_R1 | delta vs anchor mean | delta vs best anchor | #anchor diffs | Anchor-relative diffs |
| --- | --- | --- | --- | --- | --- |
| 20260602_131542 | 61.550 | -0.506 | -0.900 | 4 | num_parts=6, enrich_gamma=None, residual_gate=residual, mixer_hidden_rank=64 |
| 20260602_162211 | 61.700 | -0.356 | -0.750 | 5 | num_parts=6, enrich_gamma=None, residual_gate=residual, mixer_hidden_rank=32, mixer_hidden_readout=128 |
| 20260530_143045 | 61.750 | -0.306 | -0.700 | 0 | anchor |
| 20260530_144312 | 61.750 | -0.306 | -0.700 | 0 | anchor |
| 20260601_210730 | 61.800 | -0.256 | -0.650 | 7 | num_parts=6, enrich_gamma=None, residual_gate=residual, mixer_depth=1, mixer_hidden_rank=64, mixer_hidden_channel=256, mixer_hidden_readout=128 |
| 20260602_123502 | 61.800 | -0.256 | -0.650 | 5 | num_parts=6, enrich_gamma=None, residual_gate=residual, mixer_hidden_rank=64, mixer_hidden_readout=64 |
| 20260530_163954 | 61.850 | -0.206 | -0.600 | 3 | top_m=16, enrich_gamma=None, residual_gate=residual |
| 20260530_164001 | 61.850 | -0.206 | -0.600 | 0 | anchor |
| 20260601_165108 | 61.850 | -0.206 | -0.600 | 4 | top_m=64, extractor_mode=global,grid, num_parts=3, mixer_depth=1 |
| 20260602_123427 | 61.850 | -0.206 | -0.600 | 6 | num_parts=6, enrich_gamma=None, residual_gate=residual, mixer_dim=512, mixer_hidden_rank=64, mixer_hidden_readout=128 |

| Category | Run | best_R1 | Relevant diffs |
| --- | --- | --- | --- |
| overall | 20260531_183800 | 62.950 | top_m=64, extractor_mode=global,vertical, mixer_depth=1 |
| non-anchor | 20260531_183800 | 62.950 | top_m=64, extractor_mode=global,vertical, mixer_depth=1 |
| target-aware text enrichment | 20260531_183800 | 62.950 | top_m=64, extractor_mode=global,vertical |
| mlp-mixer module settings | 20260531_183800 | 62.950 | mixer_depth=1 |
| target-aware loss settings | 20260531_223814 | 62.300 | lambda_ret=5 |

## 4. One-Factor-At-A-Time Analysis

Strict parser-default OFAT is unavailable because there are no valid official parser-default runs. This OFAT table is relative to the operational sweep anchor.

### target-aware text enrichment
| Parameter | Tested value | n | best_R1 | delta vs anchor mean | Conclusion |
| --- | --- | --- | --- | --- | --- |
| target_enrichment | not tested OFAT |  |  |  | insufficient evidence |
| enrichment_start | not tested OFAT |  |  |  | insufficient evidence |
| enrichment_space | not tested OFAT |  |  |  | insufficient evidence |
| top_m | 64 | 1 | 62.300 | 0.244 | neutral/within noise (weak); delta vs best anchor -0.150 |
| extractor_mode | global,vertical | 1 | 62.200 | 0.144 | neutral/within noise (weak); delta vs best anchor -0.250 |
| num_parts | 48 | 1 | 61.950 | -0.106 | neutral/within noise (weak); delta vs best anchor -0.500 |
| use_freeze_indices | not tested OFAT |  |  |  | insufficient evidence |
| pnp_text_only | not tested OFAT |  |  |  | insufficient evidence |
| enrich_gamma | 0.2 | 1 | 62.200 | 0.144 | neutral/within noise (weak); delta vs best anchor -0.250 |
| enrich_gamma | 0.3 | 1 | 61.950 | -0.106 | neutral/within noise (weak); delta vs best anchor -0.500 |
| enrich_gamma | 0.4 | 1 | 62.050 | -0.006 | neutral/within noise (weak); delta vs best anchor -0.400 |
| enrich_gamma | 0.5 | 1 | 62.050 | -0.006 | neutral/within noise (weak); delta vs best anchor -0.400 |
| residual_gate | not tested OFAT |  |  |  | insufficient evidence |
| residual_gate_hidden_dim | not tested OFAT |  |  |  | insufficient evidence |
| recompute_level | not tested OFAT |  |  |  | insufficient evidence |
| recompute_interval | not tested OFAT |  |  |  | insufficient evidence |

### mlp-mixer module settings
| Parameter | Tested value | n | best_R1 | delta vs anchor mean | Conclusion |
| --- | --- | --- | --- | --- | --- |
| context_module | not tested OFAT |  |  |  | insufficient evidence |
| mixer_dim | not tested OFAT |  |  |  | insufficient evidence |
| mixer_depth | 1 | 2 | 62.600 | 0.544 | beneficial (moderate); delta vs best anchor 0.150 |
| mixer_depth | 4 | 1 | 62.250 | 0.194 | neutral/within noise (weak); delta vs best anchor -0.200 |
| mixer_hidden_part | not tested OFAT |  |  |  | insufficient evidence |
| mixer_hidden_rank | not tested OFAT |  |  |  | insufficient evidence |
| mixer_hidden_channel | not tested OFAT |  |  |  | insufficient evidence |
| mixer_hidden_readout | not tested OFAT |  |  |  | insufficient evidence |
| context_pooling | not tested OFAT |  |  |  | insufficient evidence |

### target-aware loss settings
| Parameter | Tested value | n | best_R1 | delta vs anchor mean | Conclusion |
| --- | --- | --- | --- | --- | --- |
| lambda_ret | not tested OFAT |  |  |  | insufficient evidence |

## 5. Marginal Trend Analysis

Marginal trends use all valid runs, including combo runs, so they are confounded when parameters move together.
### target-aware text enrichment
- `target_enrichment`: fixed at `True` among valid runs; not enough evidence.
- `enrichment_start`: fixed at `1` among valid runs; not enough evidence.
- `enrichment_space`: fixed at `global` among valid runs; not enough evidence.
**`top_m`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 64 | 7 | 62.507 | 62.950 | 61.850 | 0.393 |
| 32 | 95 | 62.204 | 62.900 | 61.550 | 0.263 |
| 128 | 3 | 62.517 | 62.800 | 62.350 | 0.247 |
| 16 | 2 | 61.900 | 61.950 | 61.850 | 0.071 |
**`extractor_mode`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| global,vertical | 17 | 62.462 | 62.950 | 62.100 | 0.296 |
| global,horizontal | 85 | 62.185 | 62.800 | 61.550 | 0.265 |
| global,grid | 4 | 62.187 | 62.350 | 61.850 | 0.236 |
| global | 1 | 62.000 | 62.000 | 62.000 |  |
**`num_parts`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 24 | 43 | 62.287 | 62.950 | 61.750 | 0.304 |
| 6 | 56 | 62.190 | 62.800 | 61.550 | 0.277 |
| 12 | 2 | 62.225 | 62.350 | 62.100 | 0.177 |
| 3 | 4 | 62.187 | 62.350 | 61.850 | 0.236 |
| 48 | 2 | 62.050 | 62.150 | 61.950 | 0.141 |
- `use_freeze_indices`: fixed at `True` among valid runs; not enough evidence.
**`pnp_text_only`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| True | 93 | 62.205 | 62.950 | 61.550 | 0.283 |
| False | 14 | 62.371 | 62.800 | 62.100 | 0.268 |
**`enrich_gamma`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 38 | 62.288 | 62.950 | 61.750 | 0.297 |
| 0.2 | 4 | 62.450 | 62.900 | 62.200 | 0.308 |
| 0.5 | 2 | 62.425 | 62.800 | 62.050 | 0.530 |
| 1 | 2 | 62.500 | 62.750 | 62.250 | 0.354 |
| 0.7 | 1 | 62.650 | 62.650 | 62.650 |  |
| 0.3 | 2 | 62.300 | 62.650 | 61.950 | 0.495 |
| None | 52 | 62.130 | 62.650 | 61.550 | 0.239 |
| 0.9 | 1 | 62.600 | 62.600 | 62.600 |  |
| 0.6 | 1 | 62.450 | 62.450 | 62.450 |  |
| 0.4 | 3 | 62.150 | 62.350 | 62.050 | 0.173 |
| 0.8 | 1 | 62.200 | 62.200 | 62.200 |  |
**`residual_gate`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| static | 55 | 62.319 | 62.950 | 61.750 | 0.297 |
| residual | 52 | 62.130 | 62.650 | 61.550 | 0.239 |
**`residual_gate_hidden_dim`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 128 | 105 | 62.229 | 62.950 | 61.550 | 0.288 |
| 512 | 1 | 62.150 | 62.150 | 62.150 |  |
| 256 | 1 | 62.100 | 62.100 | 62.100 |  |
- `recompute_level`: fixed at `epoch` among valid runs; not enough evidence.
- `recompute_interval`: fixed at `-1` among valid runs; not enough evidence.

### mlp-mixer module settings
- `context_module`: fixed at `mixer` among valid runs; not enough evidence.
**`mixer_dim`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 256 | 96 | 62.235 | 62.950 | 61.550 | 0.286 |
| 128 | 5 | 62.250 | 62.650 | 62.000 | 0.289 |
| 512 | 3 | 62.067 | 62.450 | 61.850 | 0.333 |
| 384 | 2 | 62.200 | 62.400 | 62.000 | 0.283 |
| 192 | 1 | 61.900 | 61.900 | 61.900 |  |
**`mixer_depth`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 1 | 35 | 62.384 | 62.950 | 61.800 | 0.270 |
| 2 | 60 | 62.167 | 62.800 | 61.550 | 0.277 |
| 3 | 8 | 62.025 | 62.350 | 61.850 | 0.163 |
| 4 | 4 | 62.162 | 62.250 | 62.100 | 0.075 |
**`mixer_hidden_part`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 32 | 97 | 62.225 | 62.950 | 61.550 | 0.291 |
| 16 | 5 | 62.310 | 62.650 | 62.000 | 0.253 |
| 128 | 3 | 62.167 | 62.450 | 62.000 | 0.247 |
| 64 | 2 | 62.200 | 62.400 | 62.000 | 0.283 |
**`mixer_hidden_rank`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 128 | 55 | 62.263 | 62.950 | 61.750 | 0.288 |
| 64 | 44 | 62.185 | 62.800 | 61.550 | 0.279 |
| 32 | 5 | 62.220 | 62.650 | 61.700 | 0.375 |
| 256 | 3 | 62.200 | 62.450 | 62.000 | 0.229 |
**`mixer_hidden_channel`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 512 | 98 | 62.230 | 62.950 | 61.550 | 0.283 |
| 256 | 5 | 62.310 | 62.650 | 61.800 | 0.370 |
| 1024 | 2 | 62.150 | 62.400 | 61.900 | 0.354 |
| 768 | 2 | 61.950 | 62.000 | 61.900 | 0.071 |
**`mixer_hidden_readout`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 256 | 55 | 62.255 | 62.950 | 61.550 | 0.300 |
| 128 | 44 | 62.184 | 62.800 | 61.700 | 0.270 |
| 64 | 5 | 62.300 | 62.650 | 61.800 | 0.332 |
| 512 | 3 | 62.217 | 62.350 | 62.150 | 0.115 |
- `context_pooling`: fixed at `mlp` among valid runs; not enough evidence.

### target-aware loss settings
**`lambda_ret`**
| value | n | mean | max | min | std |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 102 | 62.229 | 62.950 | 61.550 | 0.291 |
| 5 | 1 | 62.300 | 62.300 | 62.300 |  |
| 1 | 2 | 62.200 | 62.250 | 62.150 | 0.071 |
| 20 | 1 | 62.100 | 62.100 | 62.100 |  |
| 40 | 1 | 62.100 | 62.100 | 62.100 |  |

## 6. Interaction Analysis

| Interaction | Support | n runs | n combos | Best observed | Clearly bad combos |
| --- | --- | --- | --- | --- | --- |
| enrichment_space x extractor_mode | insufficient | 107 | 4 | enrichment_space=global, extractor_mode=global,vertical -> max 62.950, mean 62.462, n=17, run=20260531_183800 | none clearly below baseline-noise |
| residual_gate x enrich_gamma | strong | 107 | 11 | residual_gate=static, enrich_gamma=0.1 -> max 62.950, mean 62.288, n=38, run=20260531_183800 | none clearly below baseline-noise |
| residual_gate x residual_gate_hidden_dim | strong | 107 | 4 | residual_gate=static, residual_gate_hidden_dim=128 -> max 62.950, mean 62.319, n=55, run=20260531_183800 | none clearly below baseline-noise |
| recompute_level x recompute_interval | insufficient | 107 | 1 | recompute_level=epoch, recompute_interval=-1 -> max 62.950, mean 62.227, n=107, run=20260531_183800 | none clearly below baseline-noise |
| mixer_dim x mixer_depth | strong | 107 | 12 | mixer_dim=256, mixer_depth=1 -> max 62.950, mean 62.386, n=32, run=20260531_183800 | none clearly below baseline-noise |
| mixer_depth x mixer_hidden_channel | strong | 107 | 10 | mixer_depth=1, mixer_hidden_channel=512 -> max 62.950, mean 62.394, n=32, run=20260531_183800 | none clearly below baseline-noise |
| mixer_depth x mixer_hidden_readout | strong | 107 | 13 | mixer_depth=1, mixer_hidden_readout=256 -> max 62.950, mean 62.429, n=26, run=20260531_183800 | none clearly below baseline-noise |
| mixer_hidden_part x mixer_hidden_rank | strong | 107 | 12 | mixer_hidden_part=32, mixer_hidden_rank=128 -> max 62.950, mean 62.268, n=54, run=20260531_183800 | mixer_hidden_part=32, mixer_hidden_rank=32 mean 61.700 n=1 |
| context_pooling x mixer_depth | insufficient | 107 | 4 | context_pooling=mlp, mixer_depth=1 -> max 62.950, mean 62.384, n=35, run=20260531_183800 | none clearly below baseline-noise |
| extractor_mode/num_parts x mixer_hidden_part | strong | 107 | 13 | extractor_mode=global,vertical, num_parts=24, mixer_hidden_part=32 -> max 62.950, mean 62.568, n=11, run=20260531_183800 | none clearly below baseline-noise |
| top_m x mixer_hidden_rank | strong | 107 | 7 | top_m=64, mixer_hidden_rank=128 -> max 62.950, mean 62.507, n=7, run=20260531_183800 | none clearly below baseline-noise |
| target retrieval weight x top_m | strong | 107 | 8 | lambda_ret=0.5, top_m=64 -> max 62.950, mean 62.507, n=7, run=20260531_183800 | none clearly below baseline-noise |
| small-capacity vs large-capacity mixer profile | moderate | 107 | 3 | large -> max 62.950, mean 62.224, n=70 | none clearly below baseline-noise |
| target enrichment cache profile x mixer capacity | moderate | 107 | 6 | top_m=64/frozen=True, large -> max 62.950, mean 62.507, n=7 | none clearly below baseline-noise |

## 7. Capacity And Stability Analysis

| Capacity parameter | Observed values | Fixed value | Direction |
| --- | --- | --- | --- |
| mixer_dim | 256 n=96 mean=62.235 max=62.950; 128 n=5 mean=62.250 max=62.650; 512 n=3 mean=62.067 max=62.450; 384 n=2 mean=62.200 max=62.400; 192 n=1 mean=61.900 max=61.900 |  | within noise |
| mixer_depth | 1 n=35 mean=62.384 max=62.950; 2 n=60 mean=62.167 max=62.800; 3 n=8 mean=62.025 max=62.350; 4 n=4 mean=62.162 max=62.250 |  | within noise |
| mixer_hidden_channel | 512 n=98 mean=62.230 max=62.950; 256 n=5 mean=62.310 max=62.650; 1024 n=2 mean=62.150 max=62.400; 768 n=2 mean=61.950 max=62.000 |  | within noise |
| mixer_hidden_readout | 256 n=55 mean=62.255 max=62.950; 128 n=44 mean=62.184 max=62.800; 64 n=5 mean=62.300 max=62.650; 512 n=3 mean=62.217 max=62.350 |  | within noise |
| mixer_hidden_part | 32 n=97 mean=62.225 max=62.950; 16 n=5 mean=62.310 max=62.650; 128 n=3 mean=62.167 max=62.450; 64 n=2 mean=62.200 max=62.400 |  | within noise |
| mixer_hidden_rank | 128 n=55 mean=62.263 max=62.950; 64 n=44 mean=62.185 max=62.800; 32 n=5 mean=62.220 max=62.650; 256 n=3 mean=62.200 max=62.450 |  | within noise |
| top_m | 64 n=7 mean=62.507 max=62.950; 32 n=95 mean=62.204 max=62.900; 128 n=3 mean=62.517 max=62.800; 16 n=2 mean=61.900 max=61.950 |  | larger mean improves |
| num_parts | 24 n=43 mean=62.287 max=62.950; 6 n=56 mean=62.190 max=62.800; 12 n=2 mean=62.225 max=62.350; 3 n=4 mean=62.187 max=62.350; 48 n=2 mean=62.050 max=62.150 |  | within noise |
| residual_gate_hidden_dim | 128 n=105 mean=62.229 max=62.950; 512 n=1 mean=62.150 max=62.150; 256 n=1 mean=62.100 max=62.100 |  | within noise |

Stability/crash regions:
| Parameter | Value | Crashed runs | All runs |
| --- | --- | --- | --- |
| mixer_dim | 256 | 7 | 110 |
| mixer_dim | 512 | 1 | 5 |
| mixer_depth | 1 | 5 | 41 |
| mixer_depth | 2 | 2 | 68 |
| mixer_depth | 3 | 1 | 11 |
| mixer_hidden_channel | 512 | 8 | 111 |
| mixer_hidden_readout | 64 | 1 | 6 |
| mixer_hidden_readout | 128 | 2 | 50 |
| mixer_hidden_readout | 256 | 5 | 64 |
| mixer_hidden_part | 32 | 8 | 113 |
| mixer_hidden_rank | 64 | 3 | 51 |
| mixer_hidden_rank | 128 | 5 | 64 |
| top_m | 32 | 5 | 108 |
| top_m | 128 | 3 | 7 |
| num_parts | 3 | 1 | 5 |
| num_parts | 6 | 4 | 66 |
| num_parts | 24 | 3 | 49 |
| residual_gate_hidden_dim | 128 | 8 | 122 |

## 8. Freeze-Region Recommendation

| Group | Parameter | Region | Freeze/test value | Confidence | Evidence summary |
| --- | --- | --- | --- | --- | --- |
| target-aware text enrichment | target_enrichment | YELLOW / Keep Testing | True | insufficient | Fixed at a non-parser sweep-base value; keep it for continuity, but add a parser-default control before calling it better. |
| target-aware text enrichment | enrichment_start | GREEN / Freeze | 1 | insufficient | Fixed at parser default in valid data; not enough evidence to tune it. |
| target-aware text enrichment | enrichment_space | GREEN / Freeze | global | insufficient | Fixed at parser default in valid data; not enough evidence to tune it. |
| target-aware text enrichment | top_m | YELLOW / Keep Testing | 64 | moderate | Best non-anchor max 62.950, delta 0.894, n=7; confirm before hard freeze. |
| target-aware text enrichment | extractor_mode | YELLOW / Keep Testing | global,vertical | moderate | Best non-anchor max 62.950, delta 0.894, n=17; confirm before hard freeze. |
| target-aware text enrichment | num_parts | YELLOW / Keep Testing | 6 | moderate | Best non-anchor max 62.800, delta 0.744, n=56; confirm before hard freeze. |
| target-aware text enrichment | use_freeze_indices | YELLOW / Keep Testing | True | insufficient | Fixed at a non-parser sweep-base value; keep it for continuity, but add a parser-default control before calling it better. |
| target-aware text enrichment | pnp_text_only | YELLOW / Keep Testing | False | moderate | Best non-anchor max 62.800, delta 0.744, n=14; confirm before hard freeze. |
| target-aware text enrichment | enrich_gamma | YELLOW / Keep Testing | 0.2 | moderate | Best non-anchor max 62.900, delta 0.844, n=4; confirm before hard freeze. |
| target-aware text enrichment | residual_gate | YELLOW / Keep Testing | residual | moderate | Best non-anchor max 62.650, delta 0.594, n=52; confirm before hard freeze. |
| target-aware text enrichment | residual_gate_hidden_dim | GREEN / Freeze | 128 | moderate | No non-anchor value clearly beats the measured anchor beyond noise; best non-anchor 512 max 62.150, delta 0.094. |
| target-aware text enrichment | recompute_level | GREEN / Freeze | epoch | insufficient | Fixed at parser default in valid data; not enough evidence to tune it. |
| target-aware text enrichment | recompute_interval | YELLOW / Keep Testing | -1 | insufficient | Fixed at a non-parser sweep-base value; keep it for continuity, but add a parser-default control before calling it better. |
| mlp-mixer module settings | context_module | GREEN / Freeze | mixer | insufficient | Fixed at parser default in valid data; not enough evidence to tune it. |
| mlp-mixer module settings | mixer_dim | YELLOW / Keep Testing | 128 | moderate | Best non-anchor max 62.650, delta 0.594, n=5; confirm before hard freeze. |
| mlp-mixer module settings | mixer_depth | YELLOW / Keep Testing | 1 | moderate | Best non-anchor max 62.950, delta 0.894, n=35; confirm before hard freeze. |
| mlp-mixer module settings | mixer_hidden_part | YELLOW / Keep Testing | 16 | moderate | Best non-anchor max 62.650, delta 0.594, n=5; confirm before hard freeze. |
| mlp-mixer module settings | mixer_hidden_rank | YELLOW / Keep Testing | 64 | moderate | Best non-anchor max 62.800, delta 0.744, n=44; confirm before hard freeze. |
| mlp-mixer module settings | mixer_hidden_channel | YELLOW / Keep Testing | 256 | moderate | Best non-anchor max 62.650, delta 0.594, n=5; confirm before hard freeze. |
| mlp-mixer module settings | mixer_hidden_readout | YELLOW / Keep Testing | 128 | moderate | Best non-anchor max 62.800, delta 0.744, n=44; confirm before hard freeze. |
| mlp-mixer module settings | context_pooling | GREEN / Freeze | mlp | insufficient | Fixed at parser default in valid data; not enough evidence to tune it. |
| target-aware loss settings | lambda_ret | GREEN / Freeze | 0.5 | moderate | No non-anchor value clearly beats the measured anchor beyond noise; best non-anchor 5 max 62.300, delta 0.244. |

## 9. Main Conclusions

- Strongest positive trends: `top_m=64` max 62.950 (delta 0.894), `extractor_mode=global,vertical` max 62.950 (delta 0.894), `mixer_depth=1` max 62.950 (delta 0.894).
- Strongest negative/anchor-favoring trends: mostly within noise or confounded.
- Parameters needing more evidence: `num_parts`, `pnp_text_only`, `enrich_gamma`, `residual_gate`, `residual_gate_hidden_dim`, `mixer_dim`, `mixer_hidden_part`, `mixer_hidden_rank`, `mixer_hidden_channel`, `mixer_hidden_readout`, `lambda_ret`.
- Best current configuration: `20260531_183800` with best_R1 **62.950**; anchor-relative diffs: top_m=64, extractor_mode=global,vertical, mixer_depth=1.
- Best cost-effective configuration: `20260601_175258` best_R1 **62.800**, capacity `medium/default-ish`; anchor-relative diffs: num_parts=6, pnp_text_only=False, enrich_gamma=0.5, mixer_hidden_rank=64, mixer_hidden_readout=128.
- Risky/unstable regions: 8 crashed runs; exclude unfinished and crashed runs from conclusions.

## 10. Recommended Next Experiments

These minimize run count by repeating only the promising regions and adding small local probes. The repo has no local `add_run` definition, so commands are written as `add_run "<train.py CLI flags>"` queue entries.
| Purpose | Command |
| --- | --- |
| confirm_best_nondefault_a | add_run "--target_enrichment --num_parts 24 --use_freeze_indices --pnp_text_only --enrich_gamma 0.1 --residual_gate static --recompute_interval -1 --mixer_hidden_rank 128 --mixer_hidden_readout 256 --lambda_ret 0.5 --top_m 64 --extractor_mode global,vertical --mixer_depth 1 --freeze_host --no_use_host_loss --wandb_run_name confirm_best_nondefault_a" |
| confirm_best_nondefault_b | add_run "--target_enrichment --num_parts 24 --use_freeze_indices --pnp_text_only --enrich_gamma 0.1 --residual_gate static --recompute_interval -1 --mixer_hidden_rank 128 --mixer_hidden_readout 256 --lambda_ret 0.5 --top_m 64 --extractor_mode global,vertical --mixer_depth 1 --freeze_host --no_use_host_loss --wandb_run_name confirm_best_nondefault_b" |
| probe_topm16 | add_run "--target_enrichment --num_parts 24 --use_freeze_indices --pnp_text_only --enrich_gamma 0.1 --residual_gate static --recompute_interval -1 --mixer_hidden_rank 128 --mixer_hidden_readout 256 --lambda_ret 0.5 --top_m 16 --freeze_host --no_use_host_loss --wandb_run_name probe_topm16" |
| probe_topm64_repeat | add_run "--target_enrichment --num_parts 24 --use_freeze_indices --pnp_text_only --enrich_gamma 0.1 --residual_gate static --recompute_interval -1 --mixer_hidden_rank 128 --mixer_hidden_readout 256 --lambda_ret 0.5 --top_m 64 --freeze_host --no_use_host_loss --wandb_run_name probe_topm64_repeat" |
| probe_static_gamma03 | add_run "--target_enrichment --num_parts 24 --use_freeze_indices --pnp_text_only --enrich_gamma 0.3 --residual_gate static --recompute_interval -1 --mixer_hidden_rank 128 --mixer_hidden_readout 256 --lambda_ret 0.5 --freeze_host --no_use_host_loss --wandb_run_name probe_static_gamma03" |
| probe_static_gamma05 | add_run "--target_enrichment --num_parts 24 --use_freeze_indices --pnp_text_only --enrich_gamma 0.5 --residual_gate static --recompute_interval -1 --mixer_hidden_rank 128 --mixer_hidden_readout 256 --lambda_ret 0.5 --freeze_host --no_use_host_loss --wandb_run_name probe_static_gamma05" |
| probe_mlp_depth1 | add_run "--target_enrichment --num_parts 24 --use_freeze_indices --pnp_text_only --enrich_gamma 0.1 --residual_gate static --recompute_interval -1 --mixer_hidden_rank 128 --mixer_hidden_readout 256 --lambda_ret 0.5 --context_pooling mlp --mixer_depth 1 --freeze_host --no_use_host_loss --wandb_run_name probe_mlp_depth1" |
| probe_mlp_depth2 | add_run "--target_enrichment --num_parts 24 --use_freeze_indices --pnp_text_only --enrich_gamma 0.1 --residual_gate static --recompute_interval -1 --mixer_hidden_rank 128 --mixer_hidden_readout 256 --lambda_ret 0.5 --context_pooling mlp --mixer_depth 2 --freeze_host --no_use_host_loss --wandb_run_name probe_mlp_depth2" |
