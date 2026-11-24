# Sequential recruitment with a fixed candidate pool

This track evaluates a two-stage recruitment decision separately from the static `TrainingView` benchmark. A policy receives the observational training sample, the candidate pool's covariates and only the trial outcomes already purchased. The evaluator keeps unrevealed treatments/outcomes and all target effect truth. Querying a row costs one trial outcome and cannot be repeated.

The [reference configuration](../configs/adaptive_reference.toml) fixes three structural mechanisms (M0, M3 and M9), 32 independent sample seeds, a 2,048-row candidate pool, OBS size 2,000, target evaluation size 2,048 and trial budgets 128/512. Each policy starts from the same uniformly sampled 32-row pilot. It orders the remaining candidates once after seeing that pilot. The two budgets are fixed prefixes of that order, not selected stopping times. All policies share the same pool, realized potential trial observations and independent target evaluation rows within a mechanism/seed.

| Recruitment policy | Information used after the pilot | Declared rule |
| --- | --- | --- |
| `uniform` | Remaining row indices | Uniform permutation without replacement |
| `geometry` | Pool/pilot covariates | Prioritize candidates far from the pilot in standardized covariate distance |
| `outcome_residual` | Pool/pilot covariates, OBS data and pilot treatment/outcome | Prioritize local squared disagreement between pilot randomized pseudo-outcomes and an OBS-fitted contrast, with a small geometry component |

The outcome rule is an intentionally simple, public heuristic. Its score can reflect noise, baseline variation, OBS bias or treatment-effect variation. It does not identify causal trust, estimate the value of acquiring an outcome, or guarantee an improvement. Geometry is computed against the fixed pilot; no later outcomes enter recruitment. Policies do not inspect target evaluation covariates.

Every final fit uses a declared public baseline. The reference compares `rct_t_extra_trees` with `naive_pooled_t_extra_trees`: the latter pools OBS and acquired RCT rows without correcting confounding and is deliberately labeled naive. This comparison studies the recruitment contract and naive borrowing; it does not reproduce an adaptive fusion method. Both use the existing fixed tree settings, with no truth-based tuning.

The first 16 pilot rows form the charged `rct_tune` sample, and the other acquired rows form `rct_train`. Thus a budget of 128 means 112 training outcomes and 16 tuning outcomes; 512 means 496+16. The built-in final T-learners do not use tuning labels, but custom final methods can. The recruitment policy has already seen the entire pilot, including these 16 labels, so `rct_tune` is not an untouched validation sample for policy selection. No held-out diagnostic trial labels are exposed or charged to a policy. Outcome-free pool covariates are available without charge; recruitment, screening and compute costs are not modeled.

Run the track with:

```sh
python -m obs_rct_benchmark.adaptive_runner --config configs/adaptive_reference.toml --output outputs/adaptive-reference
```

The [support configuration](../configs/adaptive_support.toml) runs two support-stress mechanisms, M12 and M13, both built on M0 (no hidden confounding; same target CATE). **M12 (deficient support)** removes M0's propensity floor and multiplies its observed treatment logit by 10, so about 30% of the target has true OBS treatment probability below 0.01 or above 0.99: OBS sees essentially one arm there. **M13 (absent from OBS)** suppresses OBS sampling, with acceptance `expit(-8)`, in a localized region holding 30% of the target; about 0.3 of 2,000 OBS rows fall there, while the trial pool still covers it. In both, only trial outcomes inform the CATE in that region. Each scenario declares an evaluator-only `evaluation_region`: `deficient_support` (true OBS treatment probability outside [0.01, 0.99]) or `obs_absent` (structural truth `region_score == 1`, the region M13's OBS sampling avoids). Successful records then carry `region_metrics` with CATE MSE and root-PEHE inside and outside the region; `summary.json` and the run README add descriptive region means. Headline metrics, paired comparisons and the hashes of configurations without regions are unchanged. Policies and estimators never see the region.

Add `--resume` only for the identical configuration, package source and numerical environment. The runner records requested cells, exact query indices, charged pilot/train/tune counts, deterministic policy/fitting seeds, source hashes and terminal failures. Failed cells remain terminal on resume. Missing or failed pairs withhold paired comparisons. Summaries use independent sample-seed blocks, with scenario/budget averages formed inside each block; budgets and evaluation rows are not extra independent replications.

The protocol and configuration precede result inspection. Existing mechanisms were inspected during benchmark development; fresh sample seeds establish sampling replication conditional on those mechanisms, not unseen-mechanism generalization. All estimates and adverse outcomes are retained. Intervals are descriptive paired Monte Carlo summaries, with no prespecified multiplicity-adjusted superiority claims.

The library entry points are `RecruitmentSession`, `RecruitmentView`, `recruit`, `run_adaptive_problem` and `replay_adaptive_records` in [`adaptive.py`](../src/obs_rct_benchmark/adaptive.py). A custom policy implements `select(view, n, *, seed)` and returns exactly `n` distinct, still-hidden candidate indices. Read-only copies reduce accidental mutation; this is an information contract rather than an operating-system sandbox. Replay regenerates the trial pool from the stored configuration/seed, checks every query and budget, reconstructs policy orders and refits/scorers from observed acquired rows. The tests additionally use independent small examples for query costs, target MSE and outcome-dependent selection.
