# What the TSR static study shows

All **39,680 fits succeeded**: five recipes × 124 catalog conditions × 64 seeds, with no failures or missing cells. The [protocol](../../docs/TSR_STUDY.md) was fixed before fitting. It asks whether the two-stage residual (TSR) estimator of [Gao et al. (2026)](https://arxiv.org/abs/2603.03785v1) helps on its own, without its acquisition rule, under the catalog's static sampling. TSR here is our reimplementation: a TabPFN v2 offset fitted on OBS plus a Gaussian-process (GP) correction fitted on trial residuals.

## The OBS offset rarely helped the GP

**Family A** compares TSR with the same GP fitted to trial outcomes alone. It has 120 paired tests with one Holm adjustment at .05:

- **TSR lower: 4 conditions.** All four are budgets of the Integrative R normal-target extension, the one preset whose target population deliberately extends beyond the trial's support. There the OBS sample covers covariates that the trial does not.
- **TSR higher: 49 conditions.**
- **Not resolved: 67 conditions.**

The median ratio of TSR's mean MSE to the matched GP's is 1.15 at OBS 512 · RCT 128 and 1.30 at OBS 512 · RCT 512. It is 1.20 at OBS 2,048 · RCT 128 and 1.31 at OBS 2,048 · RCT 512.

The expected direction, recorded before results, was that TSR would beat the matched GP in most RCT-128 conditions. That was not met.

The offset also failed where the OBS stage was accurate. In 53 conditions the OBS-only recipe has a lower mean MSE than the matched GP; TSR beats the GP in only 15 of them. In power likelihood (paper text) at OBS 2,048 · RCT 128, OBS-only's mean MSE is 0.32 times the GP's and TSR's is 2.2 times. The residual correction therefore added error rather than removing it. This comparison is a post-hoc description, not a prespecified test.

## Against existing RCT-only learners

**Family B** has 240 tests, one Holm adjustment:

| TSR compared with | TSR lower | TSR higher | Not resolved |
| --- | ---: | ---: | ---: |
| RCT-only ExtraTrees | 24 | 65 | 31 |
| RCT-only ridge | 10 | 95 | 15 |

All 34 conditions where TSR is lower are paper presets. Against ridge, they are:
- the Integrative R normal-target extension, all four budgets;
- CVCI linear bias, all four budgets;
- Integrative R (KPS code) at RCT 512, two conditions.

## Budget counts (descriptive)

Presets, of 31 per budget, where TSR's mean MSE is lower than each comparator. The comparators are the matched GP, the catalog recipes and all four catalog RCT-only recipes. The catalog's means use the same samples: the rerun reproduced its published per-seed scores exactly.

| TSR mean MSE lower than | OBS 512 · RCT 128 | OBS 512 · RCT 512 | OBS 2,048 · RCT 128 | OBS 2,048 · RCT 512 |
| --- | ---: | ---: | ---: | ---: |
| matched RCT-only GP | 8 | 3 | 11 | 4 |
| RCT-only ExtraTrees | 7 | 10 | 8 | 11 |
| RCT-only ridge | 2 | 5 | 3 | 3 |
| all four catalog RCT-only recipes | 2 | 3 | 3 | 3 |

The matched GP is itself a weak RCT-only learner on this catalog: its mean is lower than the ridge T-learner's in only 3–5 of 31 presets per budget. Family A therefore measures the offset within a learner that most presets do not favor.

## Checks

- **Data identity.** The reruns of the catalog's ExtraTrees and ridge T-learners reproduce all 7,936 published per-seed MSEs each, with maximum absolute difference 0.0.
- **Independent GP implementation.** GPy's coregionalized GP reproduces our marginal likelihood within relative 8.0e-9 and our posterior effect means within 1.8e-7 of the outcome scale, on the first seed of all 31 presets ([report](evidence/gpy-check-5d09194.json)).
- **Run checks.** Every shard's settings equal the declared configuration. Seeds partition the grid. Record identities match the runner's derivation. Every successful Stage 1 record ran on CUDA with the pinned checkpoint.
- **GP convergence.** 7,918 of 7,936 TSR GP fits and 7,896 of 7,936 matched-GP fits converged within 1,000 iterations.
- **Reproduction.** A first run (r01) was stopped after CUDA out-of-memory failures caused by packing twelve processes per GPU. Its successful cells, 25,605, are reproduced exactly by the analyzed run (r02), including the GPU-based Stage 1 fits. r01 was not analyzed; see the protocol's execution record.

## What this does and does not show

In this reimplementation, on the catalog's development-inspected presets, freezing a TabPFN v2 OBS model as an offset did not improve a GP's CATE accuracy. The one exception was a target outside the trial's support. In 49 of 120 tested conditions it made the GP significantly worse.

The study does not test the following:
- the authors' code, TabPFN-2.5 or the original CMGP;
- other Stage 1 learners;
- the R-Design acquisition rule;
- data outside the catalog.

The paper's synthetic designs give the OBS bias a smooth additive form. An earlier exploratory reimplementation on such a design found a small-budget advantage for TSR over the RCT-only GP. The catalog's presets do not share that form.

The same AI agent designed, implemented and analyzed the study. A fresh-context reviewer checked the code before the main run, and its findings led to the recorded revisions. No external human review has taken place.
