# v1 core study: every method against one RCT-only reference chosen from trial data

**Status: analysis plan, not yet run.** This plan fixes the data grid, the reference, the recipes, the scores and the decision rule before any fit on the seeds below.

The freeze is the commit that adds this plan together with the files it names: [configuration](../configs/v1_core_study.toml), [shard writer](../scripts/make_v1_core_configs.py), [analyzer](../scripts/analyze_v1_core.py), [independent checker](../scripts/verify_v1_core.py) and the `rct_selected` and `cornet_raw` recipes in [`methods.py`](../src/obs_rct_benchmark/methods.py). The job record holds that commit's hash before the first fit. Revisions and errata are governed by [the last section](#revisions-and-errata).

## Why this study

The completed studies compare fusion methods with two references, and neither is fair:

- The [catalog](CATALOG_STUDY.md) fixed one RCT-only ExtraTrees T-learner in advance. At RCT 512 it is weak: an RCT-only ridge T-learner has lower mean error in 29 of 31 presets. At RCT 128 ridge is lower in only 10 of 31, so there the fixed choice is reasonable.
- The README also shows the best of four RCT-only learners per condition. Picking that learner needs the true CATE, which no analyst has. Against it, fusion methods look worse than they are.

The catalog's joint test was also withheld entirely, because one method failed in 21 of 79,104 cells.

This study replaces both references with one an analyst could actually use. It also fixes a failure rule that cannot void every decision. It runs the methods with authors' code on one fresh grid; later waves add the remaining recipes, including our labeled reimplementation of TSR.

## Question

In each condition, is a method's error in estimating the target CATE lower or higher than the error of an RCT-only learner chosen with trial data alone?

This is a competitiveness question. It does not ask whether observational data caused the difference. That needs a backbone fitted with and without OBS, which wave 1 does not contain.

## Data

- **Conditions.** The catalog's 124: 31 presets × OBS 512 or 2,048 × RCT training 128 or 512. Each condition also has 32 RCT tuning rows and 64 RCT test rows. No recipe uses either. Each condition has 2,048 evaluation points. Every preset parameter is unchanged.
- **Seeds.** 70000–70063, fresh: 64 seed blocks per condition. No earlier study used them. Methods share the realized samples within a condition and seed.
- **RCT samples do not depend on the OBS size.** In the catalog, every RCT-only recipe has identical results at OBS 512 and 2,048. Its RCT sample and method seed are the same, because method seeds ignore the budget label. So an RCT-only recipe has 60 distinct conditions outside M11, not 120, and the reference is shared by the two OBS sizes. The analyzer verifies this identity.
- **Trial and target can differ.** In six presets the evaluation population is not the trial population:
  - M9 and C08;
  - `experimental_grounding_kps` and the KPS normal-target extension;
  - multivariate Integrative R and Elastic.

  There the reference is chosen for trial-population error but scored on the target. These *transport conditions* stay in every family and are counted separately in every headline.

## The reference: `rct_selected`

A new recipe that chooses among four RCT-only learners using the RCT training rows only.

1. **Candidates.** The catalog's RCT-only recipes with catalog settings, in this order: `rct_dr_extra_trees`, `rct_dr_ridge`, `rct_t_extra_trees`, `rct_t_ridge`.
2. **Folds.** The training rows are split into five folds, stratified by treatment arm, using the recipe's method seed.
3. **Out-of-fold scores.** For each fold:
   - every candidate is fitted on the other four folds;
   - so is a common nuisance `m_hat`, an ExtraTrees regression of Y on X with 100 trees, minimum leaf 5 and all features. Its random state is the method seed plus 11 plus the fold index.

   Every candidate and `m_hat` see a copy of the data in which OBS outcomes and tuning outcomes are zero.
4. **Score.** With known assignment probability `e_i`, each candidate's score is the mean over all training rows of the known-propensity loss fixed in [V03_DESIGN](V03_DESIGN.md#fair-controls-and-model-selection):

   ```text
   L(f) = mean_i [ (Y_i − m_hat(X_i)) − (T_i − e_i) f(X_i) ]² / [ e_i (1 − e_i) ]
   ```

   Each `f(X_i)` and `m_hat(X_i)` comes from the fit that did not see row `i`.

   Given the fits, each candidate's expected score is its trial-population CATE error plus a term common to all candidates. Under randomization the cross terms vanish because `E[T − e | X] = 0` and `E[ε | X, T] = 0`.

   Two caveats follow:
   - the winner's own score is optimistically biased;
   - the error it targets is the trial CATE's, which differs from the target CATE in C08.

   Weights are not clipped. In the multivariate Integrative R preset, `1/(e(1 − e))` can exceed 100, so a few rows can decide the choice. The maximum weight is reported per condition.
5. **Choice.**
   - The lowest score wins; an exact tie goes to the earlier candidate.
   - A candidate that errors, or yields a non-finite prediction or score, is unavailable. If all four are, the recipe fails.
6. **Final fit.** The chosen candidate is refitted on all training rows with the method seed and predicts the evaluation points.
   - For `rct_t_ridge` this fit is identical to the standalone recipe.
   - For the others it differs from the standalone row in its random state: the ExtraTrees seeds and, for the DR learners, the cross-fitting split.
7. **Record.** Every fit, including a failed one, records the chosen candidate, all four scores, the unavailable candidates and the maximum weight.

**Equal labels.** Every wave-1 recipe fits on the same 128 or 512 RCT training rows; the tuning rows go unused, as in the catalog. The reference spends extra computation on selection, but no extra labels.

**How noisy the choice is.** At RCT 128 the candidates' mean errors often differ by only 0.1–0.2. Out-of-fold selection over 128 rows is therefore noisy and may pick a weaker candidate than the catalog's fixed ExtraTrees choice. The reference is what an analyst could do, not the strongest RCT-only learner. Its regret is reported, so a reader can judge it.

**Deviation from V03_DESIGN.** V03 selects on independent tuning rows and runs every method through the same selector in a controlled-tuning lane. This study departs from it in two ways:

- **Selection rows.** Selection uses out-of-fold training rows instead. With 32 tuning rows the choice is close to a coin flip, and the training rows keep the label count equal to the fusion recipes'.
- **Who is selected.** Only the reference is selected. Fusion recipes run at fixed settings: their adapters pass only training rows, and CorNet's own validation-based early stopping is disabled, as in the catalog. A lane where fusion recipes are also tuned remains future work.

**Two descriptive references, never tested.**
- `rct_t_extra_trees`, the catalog reference.
- The best of four: in each condition, whichever candidate has the lowest mean error. Picking it needs evaluation truth.

## Recipes (wave 1)

| Recipe | Data | Settings |
| --- | --- | --- |
| `rct_selected` | RCT | As above |
| 8 catalog baselines | RCT, OBS, or both naively pooled | Catalog settings: T-learner ridge and ExtraTrees for each source; RCT-only DR ridge and ExtraTrees |
| `cornet_official` | OBS + RCT | Catalog settings, standardized inputs |
| `cornet_raw` | OBS + RCT | Identical, including the torch seed, except that inputs are not standardized, which is how the authors' code takes them. Requested by the project lead on 2026-11-08 |
| `datafusionppi` (DR) | OBS + RCT | Catalog settings |

The pinned DataFusionPPI adapter accepts only a constant trial assignment probability. As in the catalog, the four multivariate Integrative R conditions are not applicable to it.

**Grid size.** `120 × 12 + 4 × 11 = 1,484` fits per seed, or 94,976 in total. Shards follow the catalog: one per seed and applicability group, 128 in all.

## Scores and decisions

- **Score.** The per-seed CATE MSE on the 2,048 evaluation points. Root-PEHE is also reported.
- **Primary contrast.** For each recipe and condition, the paired difference `MSE(recipe) − MSE(rct_selected)` over the 64 seeds. Negative favors the recipe.
- **Test.** A two-sided paired t-test per condition. If the 64 differences have zero sample variance, the condition is *unresolved* with p = 1. This happens, for example, when the reference picks `rct_t_ridge` at every seed.
- **Families.** Holm's correction runs within each recipe's family at 0.05. Error control is per recipe only; comparisons across recipes are descriptive.
  - Recipes that use OBS: their 120 conditions outside M11, or 116 for DataFusionPPI.
  - RCT-only recipes: their 60 distinct (preset, RCT size) conditions outside M11, taken from the OBS 512 records.
- **Verdicts.**
  - *Better:* the mean difference is negative and the Holm-adjusted p ≤ 0.05.
  - *Worse:* the mean difference is positive and the Holm-adjusted p ≤ 0.05.
  - *Unresolved:* anything else. It does not mean equal.
- **Failure rule.**
  - A condition where the recipe or the reference lacks a successful fit at any seed is not tested.
  - It stays in the family with p = 1, so Holm's family size never shrinks. It is reported as *untested (failed k/64)*, with its mean over successful pairs.
  - A failure in one recipe never withholds another recipe's decisions.
- **Failure-as-loss count.** Each condition reports the number of the 64 seeds at which the recipe's MSE is strictly below the reference's. A recipe failure or a tie is not a win; a reference failure leaves that seed out.
- **M11.** Its noise has an infinite fourth moment, so it stays descriptive: medians and interquartile ranges, no t-tests.
- **Headline.** For each recipe, the counts of better, worse, unresolved and untested conditions. Counts are given separately for the transport conditions (24 for recipes that use OBS, 20 for DataFusionPPI, 12 for RCT-only recipes) and the rest (96 or 48). These counts replace the current README overview once the results are published.

**Descriptive only.** The following carry no test:

- which candidate `rct_selected` chose, how often, and its maximum weight;
- `rct_selected` against each fixed candidate, and its regret relative to the best of four;
- every recipe against the catalog reference and the best of four;
- CorNet standardized against CorNet raw.

## Execution and checks

**Before results are read.** No metric from the main run is read until every shard has finished and coverage is authenticated. Preflight and monitoring read only status, runtime and memory. Resources follow the catalog: one CPU, 4 GiB and two hours per shard.

**Failures.**
- *Infrastructure failures* end a shard without terminal records. Examples are a scheduler kill, a node fault, a time or memory limit, or an unloadable environment. Before any metric is read, such a shard may be rerun whole, in a fresh output directory, with the old output and logs kept.
- *Method failures* are exceptions inside a fit, recorded as terminal cells. They are never retried or replaced.

**Checks.**
1. **Selector tests.**
   - Candidate and `m_hat` predictions are unchanged when tuning and OBS outcomes are perturbed.
   - On a small closed-form example, the score matches a hand calculation within 1e-12.
   - The fold, tie and failure rules behave as stated.
2. **Independent recomputation.** On the first seed at OBS 512 · RCT 128, a separate script regenerates every preset's data and recomputes `rct_selected`. It calls the public candidate recipes directly and checks the recorded scores (within 1e-12) and the chosen candidate.
3. **Reproducibility.** On the same subset, a separate checker refits every recipe twice. Scores agree with the records within 1e-12. A reproduced failure must keep its stage and type. All wave-1 recipes run on CPU.
4. **Identities.**
   - RCT-only records, including the reference's, must be identical at OBS 512 and 2,048. Any difference is reported, and the OBS 512 record is used: contrasts at OBS 2,048 are always paired with the OBS 512 reference.
   - When the reference chose `rct_t_ridge`, its score must equal the standalone `rct_t_ridge` score. Any difference is reported, because the zero-variance rule relies on it.
5. **Independent arithmetic.** A separate script checks the per-seed table against the run records. From it, the script recomputes all paired means, p-values, Holm adjustments and headline counts against the declared family sizes, without importing the analyzer.
6. **Coverage.** The complete set of requested cells and the not-applicable cells, with their source and environment identities, is authenticated before any aggregate is computed.

## What this study will not show

- **Inspected designs only.** All 31 presets were inspected during development, so the results describe these designs; fresh seeds replicate sampling, not new mechanisms.
- **Not a reproduction of papers.** Recipes use authors' code at disclosed settings. They do not reproduce published tables; the [reported-values packet](../results/reported_values/README.md) handles that.
- **Not OBS contribution.** Beating `rct_selected` does not show that OBS data caused the gain.
- **No independent human check.** One AI agent designed, implemented and will analyze this study. Before the freeze, a fresh-context AI reviewed this plan. No independent human has reproduced it.

## Later waves (separate addenda)

Each addendum is frozen with its exact options, configuration hash and resource limits before its own runs, and uses the same seeds, reference and decision rule. Each recipe is its own family, so no addendum can change a wave-1 decision.

- **Recipes.** Integrative R (linear and quadratic), Causal-ICM with OBS and its RCT-only control, the grf causal forest, and TSR with its two controls. TSR is our labeled reimplementation; the authors released no code.
- **Matched families.** The two matched pairs, Causal-ICM and TSR, form separate families that isolate the effect of adding OBS data. The TSR pair replicates an already published comparison on fresh seeds.
- **Data identity.** Every addendum shard reruns `rct_t_ridge` and must reproduce the wave-1 per-seed MSE exactly. A mismatch blocks pairing with wave-1 records.
- **DataFusionPPI R** joins only after its own admission checks, which have not been run.

After those waves, each item gets its own protocol:

1. **Settings designed to break the winners.**
   - For each recipe that is better than `rct_selected` in many conditions, list the assumptions its paper states, with citations. For each assumption, write a generator dial that violates it, with levels from "holds" to "clearly violated". Fix the dials and levels before any run.
   - Run **all** recipes on the new block with fresh seeds, and publish every result, including a winner that keeps winning.
   - Apply the same to every recipe, not only winners: each also gets the setting its paper was built for, if the catalog lacks one.
   - The output is a regime map: for each method, where along each dial it beats the reference and where it falls behind. No single method is expected to win everywhere, and the map shows where each one stops.
2. **Adaptive designs.** Use the recruitment track: fixed candidate pool, every revealed outcome charged to the budget.
   - Candidates are ABC3, which has official code, and R-Design. R-Design has no released code, so it would be labeled our reimplementation.
   - Policies are crossed with common final estimators, including `rct_selected`. The reference is uniform recruitment followed by `rct_selected`.
   - R-Design chooses arms as well as subjects, so it needs the joint subject-and-arm interface in [V03_DESIGN](V03_DESIGN.md#information-mechanisms-and-acquisition).
3. **More methods with authors' code.** KPS, Elastic Integrative, CVCI and IntegrativeHTEcf, once they pass the admission rules in [V03_DESIGN](V03_DESIGN.md#method-inclusion-decisions).

## Revisions and errata

- **Before results.** A change before any main-run metric is read is recorded here with its date and reason.
- **After results.** Nothing in the plan changes. A bug found later gets a dated erratum that reports the original and the corrected analysis side by side; the original outputs stay published.

**2026-11-14, before the freeze.** A fresh-context AI review led to these changes from the first draft:
- selection moved from the 32 tuning rows to out-of-fold training rows, because selection on 32 rows is close to a coin flip;
- wave 2 moved to separate addenda, because its settings were not fixed;
- infrastructure and method failures are now distinguished, and an erratum rule was added;
- transport conditions are counted separately;
- RCT-only recipes are counted per distinct (preset, RCT size);
- zero-variance contrasts are unresolved;
- `cornet_raw` shares CorNet's torch seed;
- the label count and the claims about ridge and authors' code were corrected.

The same day, an independent code review of the implementation led to these changes, still before any fit:
- the analyzer checks external-method provenance per shard, since the two shard groups request different recipes;
- contrasts at OBS 2,048 always use the OBS 512 reference record;
- the checker fixes family sizes and compares the per-seed table with the run records;
- failed fits keep their recorded diagnostics;
- a mismatch between the chosen ridge refit and the standalone ridge fit is reported;
- the checker reproduces failure stages as well as types.
