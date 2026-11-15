# Addendum: Causal-ICM with the paper's borrowing selection

**Status: draft, not yet fixed; no fit on the study seeds has run.** This addendum to the [v1 core plan](V1_CORE_STUDY.md) and [stress study 1](V1_STRESS_CORNET.md) fixes the recipe, contrasts and resources for `causal_icm_selected` before any of its fits on the study seeds. The freeze is the commit that fills in [Recorded before launch](#recorded-before-launch); earlier commits carry drafts. The job record holds the freeze commit's hash before the first fit.

[Wave 2](V1_CORE_WAVE2.md) and the [stress addendum](V1_STRESS_WAVE2.md) ran Causal-ICM with ρ fixed at 0.5. The paper chooses ρ from data (Dimitriou et al., CLeaR 2026, Section 3.4.2). Version 0.3.0a8 adds that procedure as `causal_icm_selected` ([V03_ADAPTERS](V03_ADAPTERS.md)). This addendum runs it wherever `causal_icm_fixed` ran.

It keeps from those plans:
- **Conditions and seeds.** The v1 core block has 124 conditions on seeds 70000–70063. The stress block has 64 conditions on seeds 80000–80063 with the generator `stress_dgp.py` 2.0.0.
- **Reference.** `rct_selected`.
- **Decision rule.**
  - Holm's correction within each family at 0.05.
  - The failure rule: an incomplete condition gets p = 1 and is untested.
  - In the core block, M11 is descriptive, transport conditions are counted separately, and an RCT-only reference's OBS 512 record serves both OBS sizes.
  - In the stress block, extrapolation conditions are counted separately.

## Recipe and environment

`causal_icm_selected` runs the authors' GP from the fixed recipe and chooses ρ before the final fit:
- **Grid.** ρ ∈ {0, 0.1, …, 1}.
- **Folds.** Five folds over the pooled RCT and OBS rows, shuffled with the fit's method seed.
- **Score.** Each fold's held-out RCT rows are scored by squared error, weighted by 1 / P(trial | x). That probability comes from a default logistic regression on all observed covariates.
- **Choice.** The lowest mean score over folds wins, one ρ for both arms. Ties go to the smaller ρ.
- **Final fit.** The fixed recipe refits all rows at the chosen ρ with the same seed.

The runner gives `causal_icm_selected` the method seed of `causal_icm_fixed`, so a fit that selects 0.5 equals the fixed fit exactly. The selection's departures from the paper (fold-mean criterion, all covariates in the weights) are listed in [V03_ADAPTERS](V03_ADAPTERS.md). The procedure takes no options.

Every shard fits five recipes in wave 2's `gp` environment (Python 3.11, GPy 1.13.2, scikit-learn 1.5.2, numpy 1.26.4, the authors' code at `0f0c6ca`):
- `rct_selected` and `rct_t_ridge`;
- `causal_icm_fixed` and `causal_icm_rct_only`, both at ρ = 0.5, as in wave 2;
- `causal_icm_selected`.

The configurations are [`v1_icm_selected_core.toml`](../configs/v1_icm_selected_core.toml) and [`v1_icm_selected_stress.toml`](../configs/v1_icm_selected_stress.toml). The [shard writer](../scripts/make_icm_selected_configs.py) validates them and writes:
- **Core.** One shard per seed and preset, holding four budget conditions: 1,984 shards.
- **Stress.** One shard per seed, dial and trial size, holding four levels: 1,024 shards. The stress addendum held both trial sizes in one shard. Splitting them halves the longest shard, where 8,192 OBS rows make each selection slow.

## Pairing and identities

Every contrast pairs two recipes fitted in the same shard.

- **Same data and fits as wave 2.** The four recipes refitted from wave 2 run in the same environment on the same seeds and method seeds. `causal_icm_fixed` uses OBS rows, so its agreement covers every sample role, which `rct_t_ridge` alone could not. A block ran on wave 2's data and fits only if all four reproduce wave 2's `gp` scores within 1e-9, with the same success or failure in every cell. Only then are wave 2's `gp` verdicts and this addendum's described as being on the same data. Either way, verdicts here never cross shards.
- **Selection at 0.5.** Every successful selected fit that chose 0.5 must equal `causal_icm_fixed` exactly. A difference is a defect, reported and investigated; no recorded score is replaced.
- **RCT-only recipes.**
  - Core block: their OBS 512 and OBS 2048 records must be equal.
  - Stress block: in the five dials that change only OBS, their records must be equal at every level.

  Mismatches are counted.

## Contrasts

Every contrast is a paired t-test over 64 seeds. Each block has three families, each with its own Holm correction. A family has 120 conditions in the core block and 64 in the stress block.

| Family | Kind | Question |
| --- | --- | --- |
| `gp/causal_icm_selected` | primary | Is the recipe competitive with an analyst who has only the trial (`rct_selected`)? |
| `gp/causal_icm_selected vs causal_icm_rct_only` | matched | Does adding OBS to this backbone help, with ρ chosen from data? |
| `gp/causal_icm_selected vs causal_icm_fixed` | selection | Does the paper's selection beat ρ = 0.5? Same data and method seed; only the choice of ρ differs. |

**Stress readout.** For the primary family, each dial and trial size gets the stress plan's four states, read from the verdicts at levels 0 and 3:
- no level-0 win;
- kept;
- stopped;
- broke.

"Untested" counts as not better, and the readout flags untested levels.

**Descriptive.**
- **Selected ρ.** For each condition, how often each grid value was selected, the mean selected ρ, and the counts of fits with fewer than five usable folds or an unconverged participation model.
- **Condition means.** The mean error of every recipe in every condition.
- **Dose-response (stress block).** Every recipe's error relative to `rct_selected` by dial and level, as in the stress addendum.

## Execution

- **Before results are read.** No metric is read until every shard of a block has finished and coverage is authenticated. Monitoring reads only status, runtime, memory and failure types.
- **Rules.** The execution rules of wave 2, as frozen at `80c9129d0d6046a9e6845c9f0a3580c67ca024d9`, apply with the same shard driver and job scripts, including its [2026-11-15 erratum](V1_CORE_WAVE2.md#revisions-and-errata) on slow-storage nodes:
  - infrastructure failures: a job limit (time or memory), an interruption, a failed environment check, or a final attempt on a slow-storage node that recorded a method failure;
  - such a shard is rerun whole in a fresh directory, and its old output is kept.

  This addendum excludes the slow-storage nodes (b03, b14, b15, b24–b28, b30, b31) from every job.
- **Limits.** The preflight sets one limit per block:
  - time: the larger of 2 hours and 3 times the longest preflight shard, capped at 12 hours;
  - memory: the larger of 4 GiB and twice the largest preflight peak.

  The partition allows at most 12 hours, so a job-limit rerun cannot double a 12-hour limit. Such a rerun runs once more at 12 hours. If it also reaches the limit, its recorded cells stand and its unrecorded cells become `JobLimit` method failures, listed for the analyzer and checker.
- **Deadlines.** No fit has a deadline; the final-attempt rule bounds them.
- **Preflight.** Before the freeze, one seed outside each grid runs every shard of its block: seed 69999 for the core block (31 shards) and seed 79999 for the stress block (16 shards). The jobs delete all records after writing status files.
- **Expected cost.** In a check at one seed, a selected fit took 20 to 27 times as long as a fixed fit: about 19 seconds at OBS 512 · RCT 128 and 350 seconds at OBS 2048 · RCT 512. The preflight times the 8,192-row stress conditions.

## Checks

1. **Coverage.** Every declared shard and every requested cell has a terminal record. All shards of a block share one source, environment and external-method provenance. Every successful selected fit records a completed selection with a ρ on the grid and its own fold seed. The analyzer refuses anything else.
2. **Identities.** The wave-2 reproduction, the selection at 0.5, and the RCT-only identities above.
3. **Reproducibility.** On each block's first seed, every recipe is refitted twice in the `gp` environment and compared with the record within 1e-12:
   - core block: OBS 512 · RCT 128 in all 31 presets;
   - stress block: RCT 128 at levels 0 and 3 of every dial.

   A failure is reported and investigated; no recorded score is replaced.
4. **Independent arithmetic.** A [separate script](../scripts/verify_icm_selected.py) recomputes everything the [analyzer](../scripts/analyze_icm_selected.py) reports from the per-seed records and the declared families, without importing the analyzer:
   - every contrast, Holm adjustment, verdict and count;
   - the headline rows and the readout states;
   - the identity counts and the selected-ρ counts.

## What this addendum will not show

- **Selection per dataset, not per setting.** The authors tuned ρ once per simulation setting and reused it across replications. Here every fit selects its own ρ from its own data, as an analyst with one dataset would. This costs more and makes the result depend on the selection's own noise.
- **Against ρ = 0.5, not the best fixed ρ.** The selection family compares the paper's procedure with the default that wave 2 ran. It does not compare it with the best ρ in hindsight.
- **Inspected designs.** All presets and dials were inspected during development, and wave 2's fixed-ρ results have been read. Fresh seeds replicate sampling within these designs, not new mechanisms or real data.
- **AI verification only.** One AI agent designed, implemented and will analyze this addendum. A second AI model reviewed the selection code. No person has reproduced it.

## Recorded before launch

To be filled in from the preflight before the freeze.

## Revisions and errata

None yet.
