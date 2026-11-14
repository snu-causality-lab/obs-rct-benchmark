# What the v1 core study shows

The [plan](../../docs/V1_CORE_STUDY.md) was frozen at commit `a10fe6d` before any fit on its seeds. The study reran the catalog's methods on fresh seeds: 31 presets × 4 budgets × 64 seeds × 12 recipes. Each method is compared with `rct_selected`, an RCT-only learner chosen from four by out-of-fold error on the trial's training rows. Every required check passed.

## Coverage and checks

- **Fits.** 94,962 of 94,976 succeeded, and no cell is missing.
- **Failures.** All 14 failures are DataFusionPPI on the Elastic preset at OBS 512: 5 at RCT 128 and 9 at RCT 512. The catalog failed on the same design. By the declared rule, those two conditions are untested for DataFusionPPI and leave every other decision intact. No failed cell was retried.
- **Checks.**
  - An independent recomputation of the reference's selection matched all 31 cases.
  - Two refits of every recipe on the first seed at OBS 512 · RCT 128 matched the records: 742 refits.
  - An independent recomputation of every contrast, Holm adjustment and headline count found no mismatch. That check first ran with missing arguments and was rerun as a separate job.
- **Identities.** RCT-only records are identical at OBS 512 and 2,048. When the reference chose ridge, its score equalled the standalone ridge score in all 2,444 cases.

## Headline

Each recipe is compared with `rct_selected` in each condition by a paired t-test over 64 seeds, with Holm's correction within the recipe at 0.05. Recipes that use OBS have 120 conditions outside M11, and DataFusionPPI has 116. RCT-only recipes have 60 distinct conditions.

| Recipe | Conditions | Better | Worse | Unresolved | Untested | Transport conditions: better / worse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CorNet, raw inputs | 120 | 88 | 7 | 25 | 0 | 17 / 0 of 24 |
| CorNet, standardized inputs | 120 | 71 | 14 | 35 | 0 | 9 / 4 of 24 |
| DataFusionPPI | 116 | 14 | 43 | 57 | 2 | 8 / 2 of 20 |
| Naive pooled T ridge | 120 | 33 | 65 | 22 | 0 | 4 / 13 of 24 |
| Naive pooled T trees | 120 | 21 | 79 | 20 | 0 | 8 / 9 of 24 |
| OBS-only T ridge | 120 | 27 | 75 | 18 | 0 | 4 / 14 of 24 |
| OBS-only T trees | 120 | 19 | 83 | 18 | 0 | 8 / 10 of 24 |
| RCT-only T ridge | 60 | 26 | 6 | 28 | 0 | 7 / 1 of 12 |
| RCT-only T trees (catalog reference) | 60 | 22 | 30 | 8 | 0 | 2 / 9 of 12 |
| RCT-only DR ridge | 60 | 23 | 9 | 28 | 0 | 6 / 1 of 12 |
| RCT-only DR trees | 60 | 0 | 60 | 0 | 0 | 0 / 12 of 12 |

"Unresolved" does not mean equal. Error control is per recipe; differences between rows are descriptive.

By trial size:
- **CorNet with raw inputs** is better in 51 of 60 conditions at RCT 128 and worse in none. At RCT 512 it is better in 37 and worse in 7.
- **CorNet with standardized inputs** is better in 47 and worse in 4 at RCT 128, and better in 24 and worse in 10 at RCT 512.
- **DataFusionPPI** is worse in 39 of 60 conditions at RCT 128. At RCT 512 it is mostly unresolved: 45 unresolved, 8 better, 4 worse.

## The reference is weaker than a well-chosen fixed learner

The plan warned that selection on 128 rows might be noisy. It was.

- **Selection regret.** In the median preset, `rct_selected` has 1.32 times the mean error of the best of its four candidates at RCT 128, and 1.31 times at RCT 512. "Best" here means the candidate with the lowest mean error, which needs evaluation truth to pick.
- **The best fixed candidate depends on trial size.** At RCT 128 the catalog's ExtraTrees T-learner is the best candidate in 21 of 31 presets. At RCT 512 the ridge T-learner is, in 25 of 31.
- **Consequences.**
  - The catalog reference beats `rct_selected` in 19 of 30 conditions at RCT 128, but loses in 23 of 30 at RCT 512.
  - The ridge T-learner beats `rct_selected` in 19 of 30 conditions at RCT 512.
- **What a no-truth fix would need.** Picking the right learner for each trial size needs either more selection data or a rule that is fixed in advance. Neither is available to an analyst without evaluation truth.

The verdicts above are therefore against a realistic analyst, not the strongest RCT-only learner. A stricter descriptive comparison, against the best of four in each condition, gives this:

| Recipe | Lower mean error than the best of four, RCT 128 | RCT 512 | Median ratio to the best of four, RCT 128 | RCT 512 |
| --- | ---: | ---: | ---: | ---: |
| CorNet, raw inputs | 51 / 60 | 35 / 60 | 0.79 | 0.98 |
| CorNet, standardized inputs | 46 / 60 | 24 / 60 | 0.84 | 1.02 |
| DataFusionPPI | 6 / 57 | 6 / 57 | 2.02 | 1.24 |

Conditions with failed seeds are left out of this table. CorNet's advantage at RCT 128 holds even against this bound. At RCT 512 it is roughly even with the best RCT-only learner.

## CorNet input standardization

The catalog ran CorNet only with the benchmark's input standardization. Without it, CorNet's mean error is lower in 95 of 120 conditions (42 of 60 at RCT 128, 53 of 60 at RCT 512). Its verdicts also improve: 88 better and 7 worse, against 71 and 14. This agrees with the [CorNet Figure 3 replay](../reported_values/replay_cornet_fig3/README.md), where standardization explained most of the gap between the earlier benchmark value and the authors' code.

## Where CorNet falls behind

CorNet with raw inputs is worse than `rct_selected` in seven conditions, all at RCT 512:
- its own paper's design, at both OBS sizes;
- M8, at both OBS sizes;
- power likelihood (paper text), at both OBS sizes;
- C06 at OBS 512.

With standardized inputs it is also worse in its own design and in M8 at RCT 128, and at RCT 512 in Elastic, multivariate Integrative R and C06 at OBS 2,048. It is not worse in power likelihood. These are the starting points for the planned settings designed to break the winners. Those settings must be fixed from CorNet's stated assumptions before any run, not chosen from this list.

## What this study does not show

- **Inspected designs.** All 31 presets were inspected during development. Fresh seeds replicate sampling within these designs, not new mechanisms or real data.
- **Not OBS contribution.** Beating `rct_selected` is competitiveness at equal RCT labels. It does not show that OBS data caused the gain; wave 1 has no matched backbone fitted with and without OBS.
- **Not a reproduction of papers.** Recipes run authors' code at disclosed settings, not the papers' tables.
- **Transport conditions.** In six presets the evaluation population differs from the trial's. There the reference is chosen for trial-population error but scored on the target. Their counts are shown separately above.
- **One AI agent, no human check.** One AI agent designed, implemented and analyzed this study. Fresh-context AI reviewers checked the plan and the code before the freeze. No independent human has reproduced it.

Integrative R, Causal-ICM, the causal forest and TSR join in later waves under separately frozen addenda, on the same seeds and with the same reference and decision rule.
