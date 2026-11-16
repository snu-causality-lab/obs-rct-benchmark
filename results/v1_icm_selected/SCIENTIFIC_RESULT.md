# What the selected-ρ addendum shows

[Wave 2](../v1_wave2/SCIENTIFIC_RESULT.md) and the [stress addendum](../stress_wave2/SCIENTIFIC_RESULT.md) ran Causal-ICM with its borrowing parameter fixed at ρ = 0.5. The paper chooses ρ by cross-validation (Dimitriou et al., CLeaR 2026, Section 3.4.2). This [addendum](../../docs/V1_ICM_SELECTED.md) runs that choice, as `causal_icm_selected`, wherever the fixed recipe ran: every fit selects ρ from its own data, then refits at the chosen value. Each block was frozen before its first main-run fit, the core block at `7758408` and the stress block at `5e48866`. Both reuse the seeds of the earlier waves, whose fixed-ρ results had been read before this plan was written.

Every contrast pairs two recipes fitted in the same shard: a paired t-test over 64 seeds, Holm's correction at 0.05 within each family. Each block has three families:
- **primary:** `causal_icm_selected` against `rct_selected`, the RCT-only reference;
- **matched:** against `causal_icm_rct_only`, the same GP without OBS;
- **selection:** against `causal_icm_fixed`, the same GP at ρ = 0.5, with the same data and method seed.

## Core block: coverage and checks

- **Fits.** All 39,680 requested cells succeeded. 14 shards reached their 3 h 20 min limit on fully loaded nodes and were rerun once, as the plan requires; all finished within an hour.
- **Same data and fits as wave 2.** Every sample role matched wave 2 at seed 70000, byte for byte. The four recipes refitted from wave 2 reproduce its scores exactly in 7,936 of 7,936 cells each.
- **Selection at 0.5.** All 78 fits that chose ρ = 0.5 equal `causal_icm_fixed` exactly.
- **RCT-only identity.** RCT-only records are identical at both OBS sizes.
- **Reproduction.** Every recipe refitted twice at seed 70000, OBS 512 · RCT 128, in all 31 presets: 310 refits, no mismatch.
- **Arithmetic.** The independent checker recomputed every output file without a mismatch.
- **Errata.** Two job-script errors were fixed before any metric was read. The first core analysis read no wave-2 scores because of a path error; its identity check failed, and it is kept, void, in [`core/void_r01/`](core/void_r01/headline.md). Both are in the plan's [revisions and errata](../../docs/V1_ICM_SELECTED.md#revisions-and-errata).

## Core block: selection borrows heavily

The selection chose ρ ≥ 0.9 in 6,522 of 7,936 fits (82%) and ρ = 0 in 291 (4%):

| ρ | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fits | 291 | 40 | 38 | 37 | 92 | 78 | 119 | 227 | 492 | 4,315 | 2,207 |

[`core/rho_selection.csv`](core/rho_selection.csv) gives the choices per condition.

## Core block: verdicts

| Family | Better | Worse | Unresolved | RCT 128: better / worse | RCT 512: better / worse | Transport: better / worse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary: against `rct_selected` | 39 | 14 | 67 | 26 / 7 | 13 / 7 | 12 / 1 |
| Matched: against the RCT-only GP | 72 | 0 | 48 | 42 / 0 | 30 / 0 | 15 / 0 |
| Selection: against ρ = 0.5 | 32 | 17 | 71 | 16 / 14 | 16 / 3 | 4 / 5 |

Each family has 120 conditions and none is untested. "Unresolved" does not mean equal.

- **Selection pays with the larger trial.** With 512 trial outcomes, selection is better than ρ = 0.5 in 16 conditions and worse in 3, and has lower mean error in 33 of 60. With 128 it is better in 16 and worse in 14, with lower mean error in 21 of 60 (M11 excluded from these mean-error counts).
- **Where it gains and where it loses.** It is better than ρ = 0.5 at all four budgets of M0 and C04–C08, where the OBS effect route is unconfounded. With 128 trial outcomes it is worse under localized, sign-changing or subgroup-aligned confounding (M3, M4, M6 and M8 at both OBS sizes, M10 at OBS 512), in the KPS code design, and in the multivariate Integrative R and Elastic transport presets. There it still chose a mean ρ of 0.76 to 0.90: 128 trial rows did not reveal a bias confined to part of the covariate space.
- **Against the reference.** In wave 2, ρ = 0.5 was better than `rct_selected` in 38 conditions and worse in 8: 32 and 2 with 128 trial outcomes, 6 and 6 with 512. Selection moves wins from the smaller trial to the larger. These are separate families, so the comparison is descriptive.
- **OBS still helps the backbone.** Selection never makes the GP significantly worse than the same GP without OBS. In wave 2, ρ = 0.5 was significantly better than that control in 109 of 120 conditions; selection is in 72. Again, separate families: selection adds its own noise to each fit, which widens the paired intervals.

## Stress block: coverage and checks

The stress block runs the [stress addendum](../stress_wave2/SCIENTIFIC_RESULT.md)'s eight dials, built from CorNet's stated assumptions, at four levels and two trial sizes: 64 conditions per family. `trial_shrink` and `target_shift` move the target away from the trial; their 16 conditions are counted separately as extrapolation.

- **Fits.** All 20,480 requested cells succeeded. 16 `obs_size` shards, all on one node, reached the 12-hour limit. The 14 that had not finished were rerun once, as the plan requires, and completed in 4.4 to 5.8 hours. The other two had written every record and their status before the scheduler stopped them, so they stand; the frozen rerun script counts a shard with a status file as finished ([erratum](../../docs/V1_ICM_SELECTED.md#revisions-and-errata)).
- **Same data and fits as the stress addendum.** Every sample role matched at seed 80000, byte for byte. The four recipes refitted from the stress addendum reproduce its scores exactly in 4,096 of 4,096 cells each.
- **Selection at 0.5.** All 17 fits that chose ρ = 0.5 equal `causal_icm_fixed` exactly.
- **RCT-only identity.** In the five dials that change only OBS, every RCT-only record is identical at all four levels.
- **Reproduction.** Every recipe refitted twice at seed 80000, RCT 128, at levels 0 and 3 of every dial: 160 refits, no mismatch. The check ran one condition per job and was merged.
- **Arithmetic.** The independent checker recomputed every output file, including the readout states, without a mismatch.

## Stress block: selection borrows even more

The selection chose ρ ≥ 0.9 in 3,777 of 4,096 fits (92%) and ρ = 0 in 11:

| ρ | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fits | 11 | 1 | 3 | 9 | 9 | 17 | 38 | 70 | 161 | 2,162 | 1,615 |

The mean choice per condition ranged from 0.825 to 0.995 ([`stress/rho_selection.csv`](stress/rho_selection.csv)).

## Stress block: verdicts

| Family | Better | Worse | Unresolved | RCT 128: better / worse | RCT 512: better / worse | Extrapolation: better / worse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary: against `rct_selected` | 32 | 4 | 28 | 22 / 0 | 10 / 4 | 7 / 0 |
| Matched: against the RCT-only GP | 57 | 0 | 7 | 28 / 0 | 29 / 0 | 14 / 0 |
| Selection: against ρ = 0.5 | 35 | 2 | 27 | 13 / 2 | 22 / 0 | 5 / 1 |

Each family has 64 conditions and none is untested. In the stress addendum, ρ = 0.5 was better than `rct_selected` in 27 conditions and worse in 10 (25 and 0 with 128 trial outcomes, 2 and 10 with 512), and better than the same GP without OBS in 57 and worse in none.

Error relative to `rct_selected`, level 0 → 3 (ratio of mean errors, from [`stress/dose_response.csv`](stress/dose_response.csv)); readout state in parentheses:

| Dial | RCT 128: selected ρ | RCT 128: ρ = 0.5 | RCT 512: selected ρ | RCT 512: ρ = 0.5 |
| --- | --- | --- | --- | --- |
| `hidden_modifier` | 0.64 → 0.63 (kept) | 0.73 → 0.75 (kept) | 0.82 → 0.86 | 1.04 → 1.06 |
| `bias_complexity` | 1.07 → 0.27 | 0.79 → 0.68 (kept) | 0.95 → 0.55 | 1.11 → 1.01 |
| `common_shift` | 0.42 → 0.24 (kept) | 0.77 → 0.70 (kept) | 0.79 → 0.54 | 1.03 → 0.98 |
| `obs_selection` | 0.23 → 0.33 (kept) | 0.74 → 0.75 (kept) | 0.56 → 0.70 (kept) | 1.10 → 1.17 |
| `obs_size` | 1.02 → 1.05 | 0.92 → 0.98 | 1.22 → 1.49 | 1.27 → 1.42 |
| `noise_shape` | 0.68 → 0.92 (stopped) | 0.73 → 1.04 (stopped) | 0.92 → 1.44 | 1.19 → 1.72 |
| `trial_shrink` | 0.69 → 0.46 (kept) | 0.76 → 0.65 (stopped) | 0.86 → 0.70 | 1.08 → 0.70 |
| `target_shift` | 0.68 → 1.31 (stopped) | 0.74 → 0.83 (stopped) | 0.89 → 1.27 | 1.11 → 1.36 |

A row without a state had no level-0 win. No dial broke a win in either recipe.

- **Selection pays at both trial sizes here.** Against ρ = 0.5 it is better in 22 of 32 conditions and worse in none with 512 trial outcomes, and better in 13 and worse in 2 with 128. It has lower mean error in 29 and 23 of 32.
- **Against the reference, the larger trial gains most.** With 512 trial outcomes, ρ = 0.5 had 2 wins and 10 losses; selection has 10 wins and 4 losses. The four are levels 1–3 of `obs_size` and level 3 of `noise_shape`, where ρ = 0.5 also lost. With 128 trial outcomes the counts barely move (22 and 0, against 25 and 0).
- **Where it gains.** The largest gains are where the OBS effect is right or its bias averages out: `obs_selection` has no outcome bias (0.23 against 0.74 at level 0), `common_shift` has no CATE bias, and levels 2–3 of `bias_complexity` have a high-frequency bias that smoothing removes, as stress study 1 found.
- **Where it loses.** Both losses to ρ = 0.5 are at RCT 128 with a smooth CATE bias left in place. At level 0 of `bias_complexity` the bias is linear (1.07 against 0.79). At level 3 of `target_shift` the target lies where the trial has few rows, so the borrowed OBS bias goes uncorrected (1.31 against 0.83).
- **OBS still helps the backbone.** Selection is better than the same GP without OBS in 57 conditions and worse in none, the same counts as ρ = 0.5.

## Across both blocks

The selection borrows heavily almost everywhere: ρ ≥ 0.9 in 82% of core fits and 92% of stress fits. That pays when the OBS effect is close to the trial's, or its bias is one the GP smooths away, and when the trial is large enough to correct what remains. It costs with the smaller trial when the bias is smooth, localized or sign-changing, or when the target moves away from the trial. With 512 trial outcomes it is never significantly worse than ρ = 0.5 in the stress block and worse in 3 of 60 core conditions; with 128 it is worse in 14 of 60 core conditions and 2 of 32 stress conditions.

## What this addendum does not show

- **Fresh evidence.** Both blocks reuse seeds whose fixed-ρ results had been read before this plan was written. The selection procedure was fixed by the paper, not chosen from those results, but the decision to run this addendum followed them.
- **Comparisons with the earlier waves.** Only the selection family tests selected against fixed ρ. Counts set beside wave 2's or the stress addendum's come from separate Holm families and are descriptive.
- **Other selection rules.** This is the paper's cross-validation over ρ ∈ {0, 0.1, …, 1}, at the authors' defaults. Another grid, fold scheme or criterion could choose differently.
- **Designs beyond these.** All conditions are synthetic. The stress dials target CorNet's assumptions, not Causal-ICM's, and the core grid's designs were inspected during development.
- **Independent reproduction.** One AI agent designed, ran and analyzed this addendum. Separate AI reviewers checked the plan and code before the freeze; no person has reproduced it.
