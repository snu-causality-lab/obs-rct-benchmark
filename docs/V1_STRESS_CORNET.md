# Stress study 1: settings that violate CorNet's stated assumptions

**Status: analysis plan, not yet run.** This plan fixes the generator dials, their levels, the recipes and the decision rule before any fit on its seeds. The freeze is the commit that adds it with the files it names. The job record holds that commit's hash before the first fit.

## Why this study

In the [v1 core study](../results/v1_core/SCIENTIFIC_RESULT.md), CorNet with raw inputs was the clear winner: better than the RCT-only reference in 88 of 120 conditions. The benchmark's aim is to show where each method stops working, not only where it wins. So for a method that wins, we build settings that violate the assumptions **its own paper states**, and run every method on them.

The dials come from a list of CorNet's assumptions. A separate agent extracted that list from the paper and code before this plan, without access to benchmark results. They were not chosen from the presets where CorNet lost in v1.

## Relation to the v1 plan

The [v1 plan](V1_CORE_STUDY.md#later-waves-separate-addenda) described this kind of study. This plan departs from that description in three ways:

- **Recipes.** The v1 plan says to run all recipes. This study runs the twelve wave-1 recipes. The wave-2 recipes (Integrative R, Causal-ICM, the causal forest, TSR) need other environments and are still being frozen. They will join this block later, as an addendum on the same seeds, paired within each shard as in [wave 2](V1_CORE_WAVE2.md).
- **Order.** The v1 plan puts stress studies after the recipe waves. This one starts first because its recipes are ready, and adding recipes later cannot change its decisions: each recipe is its own family.
- **Coverage of assumptions.** The v1 plan asks for one dial per stated assumption. The source list has ten items.
  - Two are left out because violating them defeats every method: an unconfounded trial (A2(i)) and no outcome modification between populations (A2(ii)).
  - The remaining eight each get a dial. The "linear sparse bias head" item and the "per-arm bias simpler than outcomes" item get one dial each, `bias_complexity` and `common_shift`.

## Dials

The common data-generating process is written in [`stress_dgp.py`](../src/obs_rct_benchmark/stress_dgp.py):

- six covariates, N(0, I) unless a dial changes a law;
- `mu0(x) = 0.5 x1 + 0.5 sin(x2) + 0.25 x3²`;
- `tau(x) = 1 + 0.5 tanh(x1) + 0.5 x4`, the same in OBS, trial and target;
- unit Gaussian noise unless `noise_shape` sets it;
- trial assignment probability 0.5;
- OBS assignment probability 0.5 unless `obs_selection` sets it.

OBS outcomes add an arm-specific shift `b_t(x)`; trial outcomes do not. Level 0 means the assumption holds, or holds in its mildest form; level 3 means it is clearly violated.

**Bias amount is held fixed.** Where a dial changes the *form* of the OBS bias, its variance is the same at every level. So the dial changes the assumption, not the amount of confounding. The shape `g_c` has `E[g_c²] = 0.1225` at every level:
- at level 0 it is linear, `g = 0.2475 (x1 + x2)`;
- at levels 1–3 it is `A_c sin(c (x1 + x2))`, with c = 1, 2, 4 and `A_c` set to keep the same variance.

| Dial | CorNet assumption (arXiv:2202.12891v1) | Levels 0 → 3 | Target |
| --- | --- | --- | --- |
| `hidden_modifier` | A shared representation learned from OBS outcomes carries what the trial correction needs (eqs. 8–9; §5.1, condition 1) | OBS treated-arm shift `-0.5 (√(1-r) x1 + √r x4)`, r = 0, 1/3, 2/3, 1; variance fixed. At r = 1 the OBS outcomes do not depend on the effect modifier x4 | Trial |
| `bias_complexity` | The bias heads are linear and sparse in the representation (Assumption 3; §6.2) | Arm shifts `b1 = +g_c`, `b0 = -g_c`: a CATE bias of rising frequency, variance fixed | Trial |
| `common_shift` | The per-arm bias heads are simpler than the outcome functions (Theorem 2; §5.3.6) | `b1 = b0 = g_c`: no CATE bias, but per-arm shifts of rising frequency that CorNet must fit separately | Trial |
| `obs_selection` | No treatment selection on covariates in OBS (Remark 3, footnote 15) | No outcome shift; OBS assignment `sigmoid(k x1)`, k = 0, 1, 2, 4 | Trial |
| `trial_shrink` | Bounded density ratio between the populations (Theorem 2, footnote 7). The paper argues that covariate shift *favors* CorNet (§5.3.5) | OBS treated shift `0.5 x1`; trial covariates N(0, s² I), s = 1, 0.75, 0.5, 0.35 | OBS population N(0, I) |
| `obs_size` | Many OBS rows relative to trial rows (§5.3.5) | OBS treated shift `0.5 x1 + 0.25 x2²`; OBS rows 8,192, 2,048, 512, 128, each a prefix of the 8,192-row draw | Trial |
| `target_shift` | The OBS population is the target (A2(iii); eq. 16) | OBS treated shift `0.5 x1`; target covariates N(m e1, I), m = 0, 0.5, 1, 1.5 | Shifted population |
| `noise_shape` | Gaussian noise with constant variance (eqs. 8–9) | OBS treated shift `0.5 x1`; noise standard deviation `exp(γ x2 / 2 - γ²/4)` in OBS and trial, mean variance 1, γ = 0, 0.5, 1, 1.5 | Trial |

**Identification.** Every dial keeps the trial-population CATE identified by the randomized trial alone. Two dials are *extrapolation* dials: `trial_shrink` and `target_shift`. In each, the target population is wider than, or shifted from, the trial's. A Gaussian trial covers every point in principle, but the targets increasingly fall where the trial has few rows. Every method, including RCT-only learners, finds these harder. Their counts are reported separately.

In `trial_shrink`, levels 1–3 all have an unbounded density ratio, so that dial grades severity rather than moving from "holds" to "violated".

**What the paper predicts.** Recorded here as the paper's claims, not as our hypotheses:
- CorNet's advantage should shrink as `hidden_modifier`, `bias_complexity`, `common_shift` or `obs_size` rises.
- `obs_selection` and `noise_shape` should affect only its rate.
- `trial_shrink` should help CorNet relative to RCT-only learners.

## Design

- **Conditions.** 8 dials × 4 levels × 2 trial sizes (RCT training 128 or 512) = 64 conditions.
  - OBS has 2,048 rows except in `obs_size`.
  - Each condition also has 32 RCT tuning rows, 64 RCT test rows and 2,048 evaluation points.
- **Seeds.** 80000–80063, fresh: 64 per condition.
  - Within a seed, the levels of a dial share their random draws, so differences between levels come from the dial.
  - Level 0 of different dials is not the same data, because the bias form differs.
  - One shard per seed runs all 64 conditions: 49,152 fits in total.
- **Recipes.** The twelve wave-1 recipes with wave-1 settings, including `rct_selected` as the reference and CorNet with standardized and raw inputs. [Configuration](../configs/v1_stress_cornet.toml).

## Decisions

- **Primary contrast.** Each recipe that uses OBS data against `rct_selected`, in each of the 64 conditions, by a paired t-test over 64 seeds. Holm's correction runs within each recipe's family of 64 at 0.05.
  - The recipes that use OBS are CorNet in both input forms, DataFusionPPI, and the four OBS-only and naive pooled T-learners.
  - Verdicts and the failure rule (p = 1 for an incomplete condition, family size fixed) are as in the [v1 core plan](V1_CORE_STUDY.md#scores-and-decisions).
- **RCT-only recipes are descriptive.** Their data do not change with most dials, so they carry no family.
- **Main readout.** For each dial, each CorNet form and each trial size, the four verdicts from level 0 to level 3 give one state:
  - *no level-0 win:* CorNet is not better at level 0;
  - *kept:* better at level 0 and at level 3;
  - *stopped:* better at level 0, unresolved or untested at level 3;
  - *broke:* better at level 0, worse at level 3.
- **Counts.** Verdict counts per recipe are given separately for the 16 extrapolation conditions and the other 48.
- **Descriptive.** Every recipe's error relative to `rct_selected` by dial and level, as dose-response curves. The curves show the ratio of mean errors and the geometric mean of the seedwise error ratios, with a pointwise 95% interval on the latter.

## Execution and checks

- **Before results are read.** No metric is read until every shard has finished and coverage is authenticated. Monitoring reads only status, runtime and memory.
- **Resources.** One CPU, 8 GiB and 4 hours per shard, in the wave-1 environment.
- **Failures.**
  - *Infrastructure failures* end a shard without terminal records. Before any metric is read, such a shard is rerun whole, in a fresh output directory, with a doubled limit and the old output kept.
  - *Method failures* are terminal cells, never retried.

**Checks.**
1. **Data.** A separate script regenerates every role's covariates, assignments and outcomes on the first seed from this plan's formulas and sampling order, without calling the generator. It compares them with the generator's output at 1e-12, together with the evaluation targets.
2. **Coverage.** Every requested cell has a terminal record from one source and environment.
3. **Reproducibility.** The first seed's fits are refitted twice by a separate checker and agree within 1e-12.
4. **Independent arithmetic.** A separate script checks the per-seed table against the records. It then recomputes every contrast, Holm adjustment, verdict count and readout state.

## What this study will not show

- **A designed construction.** These are synthetic constructions built to test stated assumptions, not real data. A method that fails here may still work where its assumptions hold.
- **No method is favored by design.** Every dial keeps the target identified from the trial, and the same data stress OBS-only learners, pooling and DataFusionPPI as well.
- **One construction per assumption.** Each dial is one way to violate an assumption. Other violations of the same assumption may behave differently.
- **Same agent, no human check.** One AI agent designed the dials from an AI-extracted list of assumptions and will analyze the results. A fresh-context AI review shaped the dials before the freeze.

## Revisions and errata

**2026-11-14, before the freeze.** A fresh-context AI review led to these changes from the first draft:
- `hidden_modifier` and `bias_complexity` now hold the bias variance fixed across levels;
- three dials were added, `common_shift`, `target_shift` and `noise_shape`, and the relation to the v1 plan is stated;
- the readout has four states;
- extrapolation counts are separate;
- the data check regenerates outcomes independently;
- `obs_size` samples are nested;
- execution rules were added.
