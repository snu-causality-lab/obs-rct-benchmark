# What the full catalog study shows

All **31 native presets, four OBS/RCT budget pairs, 64 seed blocks and ten fixed recipes** are reported. The nominal grid has 79,360 cells: **256 prospectively unsupported cells**, **79,083 successful fits**, **21 terminal failures**, and **zero missing applicable cells**. CorNet uses 500 source and 200 correction epochs; none of the recipes is tuned on evaluation truth. The [frozen protocol](../../docs/CATALOG_STUDY.md) was published before fitting.

## Numerical coverage is part of the result

All failures occur in official DataFusionPPI on the Elastic preset with OBS512. Its density-ratio calibration does not reach the upstream balance equations. RCT128 has 55/64 successful seeds and nine failures; RCT512 has 52/64 successes and twelve failures. Their successful-fit mean MSEs are 5.4524 and 2.3372. These are conditional on numerical success, **not risks over the complete requested grid**. At OBS2048, both RCT conditions have 64/64 successful fits, with means 1.8364 and .67456. This observation does not by itself establish why calibration fails.

Every failure remains terminal. A separate post-hoc diagnosis refits all 21 failed cases twice: all 42 invocations reproduce the original failure stage/type. No retry replaces a failed score. The [failure records](evidence/terminal_failures.jsonl), [diagnosis](evidence/failure-diagnosis-v2.json) and exact executed diagnostic source are retained.

The four incomplete eligible fusion/control contrasts trigger the prespecified rule: **all 472 Holm decisions are withheld**. This is neither 472 nonsignificant results nor a smaller family tested after observing failures. Complete individual pointwise intervals remain unadjusted summaries. M11's canonical Student-t(3) noise is retained; its 16 supported comparisons are descriptive, with no mean-MCSE, mean-t interval or p-value. Eight additional contrasts were prospectively unsupported. All 488 supported contrast rows remain in the [CSV](primary_contrasts.csv).

## Read the conditions rather than a global raw-error ranking

The preview below uses the same OBS512/RCT128 condition in every preset and lists all 31 presets without selecting on score. It is one of four budgets, and shows four of ten recipes. Different outcome scales and target laws prevent comparing raw MSEs across rows as a universal ranking. Complete reports retain all budgets, all ten recipes, coverage, intervals, paired differences and the heavy-tail empirical quantiles.

| Preset / complete report | Native d | RCT T trees | RCT DR trees | CorNet | DataFusionPPI |
| --- | ---: | ---: | ---: | ---: | ---: |
| [structural:M0](designs/structural__M0/README.md) | 5 | 0.30132 | 0.49515 | 0.20322 | 0.63944 |
| [structural:M1](designs/structural__M1/README.md) | 5 | 0.28151 | 0.43263 | 0.19827 | 0.54282 |
| [structural:M2](designs/structural__M2/README.md) | 5 | 0.28728 | 0.4655 | 0.22566 | 0.66289 |
| [structural:M3](designs/structural__M3/README.md) | 5 | 0.29468 | 0.47919 | 0.26084 | 0.83511 |
| [structural:M4](designs/structural__M4/README.md) | 5 | 0.31266 | 0.51843 | 0.25779 | 0.83373 |
| [structural:M5](designs/structural__M5/README.md) | 5 | 0.30824 | 0.51486 | 0.19564 | 0.81957 |
| [structural:M6](designs/structural__M6/README.md) | 5 | 0.3545 | 0.57933 | 0.31994 | 0.90613 |
| [structural:M7](designs/structural__M7/README.md) | 5 | 0.28512 | 0.44775 | 0.20471 | 0.62178 |
| [structural:M8](designs/structural__M8/README.md) | 5 | 0.47431 | 0.66542 | 0.63294 | 1.1344 |
| [structural:M9](designs/structural__M9/README.md) | 5 | 0.60717 | 0.6962 | 0.48015 | 0.81516 |
| [structural:M10](designs/structural__M10/README.md) | 5 | 0.23335 | 0.36838 | 0.20771 | 0.54826 |
| [structural:M11](designs/structural__M11/README.md) | 5 | 0.57458 † | 0.87093 † | 0.54682 † | 1.3217 † |
| [addendum:C01](designs/addendum__C01/README.md) | 5 | 0.33551 | 0.54662 | 0.22566 | 0.91466 |
| [addendum:C02](designs/addendum__C02/README.md) | 5 | 0.26573 | 0.43943 | 0.18453 | 0.58602 |
| [addendum:C03](designs/addendum__C03/README.md) | 5 | 0.65473 | 0.99585 | 0.64175 | 1.5502 |
| [addendum:C04](designs/addendum__C04/README.md) | 5 | 0.30115 | 0.48731 | 0.22105 | 0.73408 |
| [addendum:C05](designs/addendum__C05/README.md) | 5 | 0.25781 | 0.46597 | 0.15499 | 0.6498 |
| [addendum:C06](designs/addendum__C06/README.md) | 5 | 0.43364 | 0.60468 | 0.34232 | 0.75762 |
| [addendum:C07](designs/addendum__C07/README.md) | 5 | 0.2477 | 0.41549 | 0.18366 | 0.64041 |
| [addendum:C08](designs/addendum__C08/README.md) | 5 | 1.6317 | 1.7985 | 1.4866 | 2.1774 |
| [addendum:C09](designs/addendum__C09/README.md) | 6 | 0.30844 | 0.50285 | 0.23213 | 0.67523 |
| [addendum:C10](designs/addendum__C10/README.md) | 6 | 0.29881 | 0.4848 | 0.21385 | 0.67418 |
| [paper:cornet_official_n_conf](designs/paper__cornet_official_n_conf/README.md) | 25 | 2.0754 | 1.9459 | 1.5806 | 0.91122 |
| [paper:experimental_grounding_kps](designs/paper__experimental_grounding_kps/README.md) | 1 | 0.60592 | 0.64334 | 0.409 | 0.3062 |
| [paper:integrative_r_learner_kps_normal_target_extension](designs/paper__integrative_r_learner_kps_normal_target_extension/README.md) | 1 | 2.6458 | 2.4606 | 1.5578 | 0.80608 |
| [paper:integrative_r_learner_kps_code](designs/paper__integrative_r_learner_kps_code/README.md) | 1 | 0.23495 | 0.27298 | 0.1324 | 0.1219 |
| [paper:integrative_r_learner_multivariate_paper](designs/paper__integrative_r_learner_multivariate_paper/README.md) | 9 | 6.1881 | 6.0459 | 1.7062 | N/A |
| [paper:elastic_integrative_gen_data_overlap](designs/paper__elastic_integrative_gen_data_overlap/README.md) | 2 | 6.5034 | 4.919 | 1.0426 | 5.4524 (55/64) |
| [paper:power_likelihood_paper_text](designs/paper__power_likelihood_paper_text/README.md) | 2 | 0.089303 | 0.12695 | 0.042747 | 0.04461 |
| [paper:power_likelihood_supplement_code](designs/paper__power_likelihood_supplement_code/README.md) | 2 | 0.094236 | 0.12554 | 0.046593 | 0.070238 |
| [paper:cvci_official_linear_bias](designs/paper__cvci_official_linear_bias/README.md) | 5 | 0.90052 | 1.4332 | 0.4303 | 1.2277 |

† M11 means are descriptive only. The Gaussian-noise twin C03 is a separate eligible design. N/A denotes the multivariate Integrative R preset's covariate-dependent trial probabilities, unsupported by the pinned DataFusionPPI adapter; the DGP was not modified to accommodate it.

The catalog includes identification and transport stresses. The normal-target KPS extension extrapolates beyond trial support, and C08 changes target CATE through hidden participation. Their target errors do not validate transport assumptions. Related variants are not independent papers, and fresh sample seeds are sampling replication conditional on inspected mechanisms. Actual IHDP/ACIC files were not evaluated. The earlier [matched-M2](../reference_study/SCIENTIFIC_RESULT.md) and [recruitment](../adaptive_reference/SCIENTIFIC_RESULT.md) studies remain separate evidence with their own seeds and inference.

## Reproducibility evidence and its scope

- **188 tests passed on GSDS.** All 31 separately written target calculations pass fixed absolute tolerance 1e-12; maximum difference 3.552713678800501e-15. CorNet's target checker shares initialized mechanism weights and independently calculates their forward result.
- **309 prespecified first-seed/low-budget cases, each refitted twice: 618 successful refits, zero mismatches.** Two regenerated datasets match for every preset. Direct-dot MSE differs from the original saved score by at most 8.881784197001252e-16 under fixed absolute/relative tolerance 1e-12. Original prediction/data-array hashes were not stored; the checker compares two new repeats and the original scalar score, rather than asserting original-array byte identity or full-study prediction replay.
- **All 128 completed-run resumes preserve records byte-for-byte.** Configuration/source/environment/official implementation identity, record IDs, method seeds, disjoint requested rectangles and the complete 79,104-cell terminal union are authenticated before aggregation.
- **Independent compact-CSV arithmetic passes:** all 1,240 score rows, 488 supported contrast rows, applicability/heavy-tail exclusions, 21 failures and complete-family withholding are verified without importing the producer. This establishes arithmetic consistency, not independent estimator correctness.
- Presentation/identity reporting was clarified after fitting. **All 34 scientific CSVs are byte-identical** to the original analysis; source snapshots and both check records remain preserved.

The [producer identity](producer_identity.json) discloses actual package hashes, numerical versions, official source hashes and resolved controls. The package source is byte-identical to public d613b3f; b77f4dc identifies the protocol/checkers published before fitting. Later presentation/delivery commits do not denote a changed scientific producer. The [full environment lock](evidence/environment-lock-v1.json), [execution record](evidence/execution_record.json), [resume checks](evidence/completed_resume.json), [target check](evidence/target-check-v2.json), [refit check](evidence/reproducibility-v1.json), and [independent arithmetic check](evidence/independent-reaggregation-v2.json) give exact bindings and limits.

The fitting campaign ran from 23:48:07 on 2026-10-25 to 00:12:20 on 2026-10-26 KST: **24m13s elapsed**, including the retained first-seed preflight. CPU-only fitting used one numerical thread per worker, up to 16 concurrent workers, with 1 CPU/4 GiB/2h requested per shard. The first 1,200-fit shard took 3m47s and recorded 358,824 KiB maximum resident size. Complete operational logs, environments, raw full-run JSONL and upstream checkouts remain on GSDS. All per-seed compact scores and selected original first-seed execution inputs are public. The [delivery manifest](DELIVERY_MANIFEST.json) distinguishes published bytes from omitted regenerable manuscript PDFs.

## Reproduce and check

Install the extras/official checkout in the main README and follow the [complete commands](../../docs/CATALOG_STUDY.md#execute-and-reproduce). The exact Linux environment is recorded, including one-thread settings; cross-platform byte identity is not established. The original pip freeze is an execution record, not a portable requirements file: its editable install points to the preserved earlier checkout, while jobs explicitly imported the frozen new snapshot through PYTHONPATH. Actual imported package hashes are verified.

To verify the published arithmetic without fitting models:

```sh
python scripts/verify_catalog_aggregates.py --config configs/catalog_study.toml --analysis results/catalog_study --delivery-manifest results/catalog_study/DELIVERY_MANIFEST.json --output outputs/catalog-arithmetic-check.json
```

The delivery option permits only explicitly listed, hash-bound manuscript budget PDFs to be absent. CSVs, producer identity and every other original analysis artifact remain mandatory. GitHub CI runs this check; it does not refit the full campaign on every push.
