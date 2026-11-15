# What wave 2 shows

The [wave-2 plan](../../docs/V1_CORE_WAVE2.md) was frozen at commit `80c9129` before any fit on its seeds. It adds the recipes the v1 core study could not run in one environment: the Integrative R-learner and grf's causal forest (R), Causal-ICM (GPy), and the two-stage residual (TSR) estimator under three TabPFN releases plus the TSR study's registered recipes. They run on the v1 core grid (31 presets at four budgets: OBS 512 or 2,048 rows, crossed with RCT 128 or 512) on the v1 core study's 64 seeds, 70000–70063. Each recipe is paired, within the same shard and environment, with `rct_selected`, the RCT-only reference of the [v1 core study](../v1_core/SCIENTIFIC_RESULT.md).

## Coverage and checks

- **Fits.** 182,336 of 182,528 requested cells succeeded and none are missing. All 192 failures are `integrative_r_quadratic` on the CorNet paper design, at every seed of three budgets; the records keep only the error type. A [trace](evidence/diagnostics/intr-trace-832380.out) of the first seed on a normal node shows the cause there: the fit reaches its 1,200-second deadline during its parameter search. That design has 25 covariates, so the quadratic basis has 350 terms. With 640 training rows (OBS 512 · RCT 128) the same fit takes about 170 seconds; with 1,024 rows or more it times out.
- **Same data as wave 1.** At the first seed, every sample role generated in the `gp`, `tabpfn` and `tsr_legacy` environments was identical to the wave-1 environment's, which the `native_r` group uses. Every `rct_t_ridge` score reproduces wave 1 within 1e-9. In three groups all 7,936 `rct_selected` scores equal wave 1's exactly; in the `tabpfn` group they differ by at most 9e-16. Ratios against `rct_selected` therefore share their denominator with the v1 core study's; comparisons across the two are descriptive.
- **OBS-size identity.** The records of `rct_selected` and the three RCT-only recipes (causal forest, RCT-only Causal-ICM, TSR's RCT-only GP) are identical at both OBS sizes: no mismatch.
- **Reproduction.** On the first seed at OBS 512 · RCT 128, each recipe was refitted twice in its group's environment: 713 cases, 1,426 refits, no mismatch.
- **Arithmetic.** An independent recomputation of every contrast, Holm adjustment and headline count found no mismatch.

Two corrections were made after the fits, both recorded in the plan's [revisions and errata](../../docs/V1_CORE_WAVE2.md#revisions-and-errata):
- **Source check.** Before any metric was read, the analyzer's first run refused the study without writing anything. Each environment had recorded the version of a stale package install, or none, and these differed from the analysis package's, although every run used byte-identical code. The analysis runs the frozen analyzer unchanged through a [runner](evidence/jobs/analyze_erratum.py) that checks the code hashes instead.
- **Slow-storage nodes.** After the first analysis had been read, a rule was added: a shard whose final attempt ran on a slow-storage node and recorded failures is rerun whole on a normal node. Seven shards qualified, all on the CorNet design. The rerun reproduced every record exactly, failures included, so the final analysis is byte-identical to the first one, which [`as_frozen/`](as_frozen/headline.md) keeps.

## Verdicts against the reference

Each family has its own Holm correction. Recipes that use OBS have 120 conditions outside the heavy-tailed M11; RCT-only recipes have 60, one per preset and trial size. Transport conditions are the six presets whose evaluation population differs from the trial's.

| Recipe | Conditions | Better | Worse | Unresolved | Untested | Transport: better / worse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Integrative R, linear | 120 | 62 | 8 | 50 | 0 | 5 / 2 of 24 |
| Integrative R, quadratic | 120 | 43 | 9 | 65 | 3 | 5 / 2 of 24 |
| Causal-ICM with OBS | 120 | 38 | 8 | 74 | 0 | 13 / 0 of 24 |
| OBS-only TabPFN v2.5 | 120 | 33 | 71 | 16 | 0 | 8 / 10 of 24 |
| TSR, TabPFN v2.5 | 120 | 10 | 84 | 26 | 0 | 6 / 5 of 24 |
| TSR, legacy recipe | 120 | 10 | 80 | 30 | 0 | 6 / 1 of 24 |
| Causal forest (RCT only) | 60 | 38 | 17 | 5 | 0 | 2 / 9 of 12 |
| Causal-ICM, RCT only | 60 | 3 | 32 | 25 | 0 | 2 / 2 of 12 |
| TSR's GP, RCT only | 60 | 6 | 30 | 24 | 0 | 2 / 1 of 12 |
| *v1 core: CorNet, raw inputs* | *120* | *88* | *7* | *25* | *0* | *17 / 0 of 24* |

The other TabPFN releases give nearly the same counts ([headline](headline.md)).

## Integrative R: the strongest new recipe that uses OBS, but not CorNet's equal with a small trial

The linear Integrative R-learner is significantly better than `rct_selected` in 62 of 120 conditions and worse in 8.
- **By trial size.** With RCT 128 it is better in 27 of 60 conditions, where raw-input CorNet was better in 51. With RCT 512 it is better in 35, close to CorNet's 37.
- **Where it loses.** Its eight losses are M8 at all four budgets, M9 at RCT 512 and the CorNet paper design at RCT 512. CorNet also lost in M8 and in its own design at RCT 512.
- **The quadratic basis** is better in 43 and worse in 9. It never finishes on the CorNet paper design except at the smallest budget.

## Causal-ICM: OBS helps its GP; the GP is the limit

Adding OBS to Causal-ICM's GP lowers its error significantly in 109 of 120 conditions, and its mean error is lower in all 120. Among this wave's matched contrasts, that is the only clear benefit of OBS within one method. Against `rct_selected` it is better in 32 of 60 conditions with RCT 128 but only 6 of 60 with RCT 512.
- **The gain from OBS shrinks as the trial grows.** In the median condition, OBS cuts the GP's error by 36–40% with RCT 128 and by 12–18% with RCT 512.
- **The GP is weak on its own.** Without OBS it is better than the reference in 3 of 60 conditions and worse in 32. Its median error is 1.42 times the reference's with RCT 128 and 1.23 times with RCT 512.
- **Transport.** It never loses a transport condition.

## TSR: rarely a gain from OBS, under any TabPFN release

- **Against the reference.** TSR is worse than `rct_selected` in 80 to 86 of 120 conditions under every release.
- **Against its own RCT-only GP.** It is better in 5 or 6 and worse in 44 to 48. The legacy recipe gives 6 better and 48 worse, close to the [TSR study](../tsr_study/SCIENTIFIC_RESULT.md)'s 4 and 49 on other seeds. In that study all four wins were in the one preset whose target lies outside the trial's support. Here four of the six are, and two are elsewhere (C04 and Elastic Integrative overlap, one budget each).
- **Release.** TabPFN v2.5, the R-Design paper's version, and v3 change TSR's error in at most 6 of 120 conditions relative to v2. For the OBS-only TabPFN model the release matters: v2.5 is better than v2 in 68 conditions and worse in 24; v3 is better in 56 and worse in 20.

## The RCT-only forest beats the reference

grf's honest causal forest, fitted to trial data alone, is better than `rct_selected` in 38 of 60 conditions and worse in 17. Thirteen of its 17 losses are in the paper designs. This confirms the v1 core study's caution: `rct_selected` is a realistic reference chosen from trial data, not the best available RCT-only learner, so "better than the reference" is not evidence that OBS caused the gain. The matched contrasts above (Causal-ICM and TSR against their own RCT-only fits) are the ones that isolate OBS.

## What this study does not show

- **Fixed settings, not tuned recipes.** Integrative R and the forest run their authors' defaults with per-fit deadlines of 1,200 and 600 seconds. Causal-ICM runs the authors' code with borrowing fixed at ρ = 0.5, not the paper's selection. TSR is reimplemented here. Integrative R's quadratic basis fails on a 25-covariate design under its deadline; a longer one might change those three conditions.
- **Synthetic presets inspected during development.** They do not show how the methods behave on real data.
- **Across groups.** Verdicts never cross environments. Comparisons with the v1 core study's recipes use the same reference scores but different families and are descriptive.
- **The TabPFN release families** compare weights and default settings under one package version (`tabpfn==10.0.0`), not older package versions.
- **No human check.** One AI agent designed, ran and analyzed wave 2. Fresh-context AI reviews checked the plan before the freeze and this write-up against the data. No human has reproduced it.
