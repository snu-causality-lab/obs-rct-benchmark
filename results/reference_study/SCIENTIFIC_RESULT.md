# What the static reference comparison shows

All **3,072 requested fits** completed: four matched M2 variants, four OBS/RCT size pairs, six fixed public recipes and 32 independent sample seeds. CorNet uses 500 source plus 200 correction epochs; the RCT controls use nonlinear ExtraTrees T- and DR-learners. The [protocol](../../docs/REFERENCE_STUDY.md) fixes information, budgets and all 64 primary contrasts before result inspection.

## Borrowing depends on the procedure and trial size

The table counts two-sided paired tests against zero, adjusted jointly by Holm over all 64 contrasts at level .05. A favorable decision has a negative paired mean and adjusted p ≤ .05; an adverse decision has a positive paired mean and adjusted p ≤ .05. Unresolved does not mean equivalent. These counts summarize directions and uncertainty, not effect sizes.

| Fusion recipe minus RCT reference | Favorable after Holm64 | Adverse after Holm64 | Unresolved |
| --- | ---: | ---: | ---: |
| CorNet minus T-learner | 15 | 0 | 1 |
| CorNet minus DR-learner | 16 | 0 | 0 |
| DataFusionPPI minus T-learner | 3 | 8 | 5 |
| DataFusionPPI minus DR-learner | 8 | 2 | 6 |

CorNet improves both nonlinear RCT controls across most of these matched conditions. DataFusionPPI's comparison changes with trial size: its small-trial error can exceed the controls, while its larger-trial results are more favorable. This study crosses hidden treatment selection on/off, OBS CATE shift on/off, two OBS sizes and two RCT training sizes within one fixed M2 construction. It does not evaluate the other catalog designs. The final recipes and 64 primary contrasts were frozen before inspecting these 32 evaluation seeds; M2 had been inspected during development. Fixed optimization recipes also need not be equally effective across methods.

The [complete condition report](README.md) retains OBS-only and naive-pooling results, every method's mean and Monte Carlo standard error, and all favorable, adverse and unresolved primary differences. The paired-difference MCSEs range from **0.0076893 to 0.0930184**, so 32 seeds do not imply uniform precision. Displayed intervals are pointwise; the Holm decisions concern the full prespecified family.

- [Budget curves](reference_budget_curves.svg)
- [All 64 contrasts](reference_primary_contrasts.svg)
- [Condition scores](condition_scores.csv), [primary contrasts](primary_contrasts.csv), [machine-readable analysis](summary.json)
- [Selected execution manifests and terminal metrics](runs/)

## Verification and reproduction

Aggregation checks recorded-artifact consistency: the disjoint seed union, configuration, source/environment fingerprints and every requested terminal metric. It does not independently validate every fitted prediction. A separate refit of the first prespecified seed checks all 16 conditions and six methods using a direct dot-product MSE calculation: 96/96 match, with maximum absolute difference 2.220446049250313e-16 under a declared 1e-12 absolute/relative tolerance. This is deterministic reconstruction of one seed, not a new scientific replication or full-study prediction replay. The [portable replay script](../../scripts/replay_static_shard.py) and [recorded checks](refit_check.json) bind the replay to its configuration, exact source/environment fingerprint and input hashes.

Recreate the published shard layout with `python scripts/make_reference_configs.py --config configs/reference_study.toml --output outputs/reference-configs --shards 32`, execute all 32 configurations, then run `python scripts/analyze_reference_study.py --config configs/reference_study.toml --runs outputs/reference/shard-* --output outputs/reference-analysis`. Exact commands and optional dependencies are in the protocol. Re-running with `--paper-assets` also generates vector PDFs; the public export retains text-based SVGs and records the omitted PDF hashes in [delivery metadata](DELIVERY_MANIFEST.json).
