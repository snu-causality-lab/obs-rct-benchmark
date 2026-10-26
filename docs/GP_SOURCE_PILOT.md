# A controlled GP source-access pilot

**Prospective development protocol, 2026-10-26.** The [configuration](../configs/gp_source_pilot.toml), native source and numerical environment are fixed before comparative fitting. This is a small fixed-recipe experiment; the previous catalog, its failure records and its withheld Holm decisions remain unchanged. It adds no claim about unseen mechanisms, optimal borrowing, or reproduction of a source-paper table.

## Question and three procedures

At the same randomized-population CATE target, measure what adding observational data changes relative to an RCT-only fit, and what adding randomized outcomes changes relative to the same model's OBS-conditioned starting prediction.

| Slot | Included fitting rows | Prediction and interpretation |
| --- | --- | --- |
| `full` | OBS 512 plus RCT B | Existing `causal_icm_fixed`: source-task0 treated mean minus source-task0 control mean. |
| `rct_only` | RCT B; zero OBS rows | Same native GP with truly empty observational arrays. OBS outcomes cannot optimize kernel or noise parameters. |
| `obs_start` | OBS 512; zero RCT rows | Same native GP conditioned only on OBS, queried at trial task0 in both arms. This is an OBS-informed trial prediction under the fixed prior. |

All procedures use rank2, all two observed covariates, rho0.5, native RBF/source coregionalization, fixed kappa1e-9, optimized Gaussian noise and native optimizer defaults. There are no additional restarts, external tuning candidates, fitted input scaling or query adaptation. Different hyperparameter estimates may arise from the different allowed fitting data. The contrasts concern the complete procedures, including that refitting; they do not isolate a fixed-hyperparameter posterior update.

The pilot passes only the included SourceSample to each control through fit_predict_sample; actual excluded-source labels never enter its method arguments. Both controls call the unchanged pinned author function with correctly shaped empty arrays for the excluded source. The common TrainingView entry point remains a compatibility delegator for other runners. The author revision, notebook and cell hashes are in [sources.lock.toml](../sources.lock.toml). Neither replaces exclusion with rho0. RCT-only and full fits charge the same B randomized outcomes. The OBS-start uses zero randomized outcomes; it is the zero-RCT baseline, not an equal-total-cost competitor.

The OBS-start is **not a naive OBS arm contrast or an optimal OBS-only estimator**. With this zero-mean two-task prior and the same fitted OBS-only model, its task0 posterior mean is `alpha = rho/(1+kappa)` times the task1 posterior mean, arm by arm. Thus its shrinkage is part of the recipe and of any measured improvement over it.

## Observed laws and causal target

All fitting and query covariates are independently Uniform[-1,1]^2. Let

```text
m(x)   = 0.5*x0 + 0.25*x1
tau(x) = 1 + 0.5*sin(pi*x0) + 0.25*cos(pi*x1)
b(x)   = cos(pi*x0)*sin(pi*x1).
```

Trial treatment is Bernoulli0.5 and its observed response is `m(X)+(T-0.5)*tau(X)+epsilon`. Observational treatment is marginally Bernoulli0.5 and its observed response is `m(X)+(T-0.5)*(tau(X)+delta*b(X))+epsilon`. Each conditional observed arm response has independent Gaussian noise variance1. Bias levels are delta0,1,2. The trial law, its CATE, its difficulty and the query population are identical across delta; only the observational conditional-mean discrepancy changes.

A structural interpretation is `U_OBS~Bernoulli0.5, T_OBS=U_OBS`, `U_RCT=0.5, T_RCT~Bernoulli0.5` and

```text
Y_s(t) = m(X)+(t-0.5)*tau(X)+delta*(U_s-0.5)*b(X)+epsilon.
```

Both populations' causal CATE is tau(X); observational conditioning yields contrast `tau+delta*b`. Observed treatment positivity is0.5. Conditional positivity given latent U fails in OBS, and the latent prognostic distribution differs between sources. This is a deliberately controlled latent-confounding/selection stress, not an assertion of a realistic empirical population or satisfaction of every native modeling assumption. No effects are extrapolated outside trial covariate support.

Under the declared target, E[tau]=1, E[b]=0, E[b²]=1/4, E[tau*b]=0 and E[tau²]=1.15625. The observational contrast bias changes CATE while leaving ATE unchanged. If the OBS conditional means were known exactly, the fixed-prior starting prediction would be alpha*(tau+delta*b), with target MSE `(1-alpha)^2*1.15625 + alpha^2*delta^2/4`. This algebra explains prior shrinkage; it is not a fitted procedure, a tuning rule or an accuracy threshold.

## Pairing, information and budgets

Use OBS512, RCT128/512, query1024 and32 fixed seeds30000–30031: six conditions,192 replicate-condition cases and576 fits. The independent seed block is the Monte Carlo unit. Within a seed, separate RNG streams generate OBS covariates/assignment/noise, RCT covariates/assignment/noise, and independent query covariates. OBS base arrays/noise and query rows are common across delta and B; RCT128 is a prefix of RCT512. Treatment is not balanced by rejection sampling; fewer than two included observations in either treatment arm is an input failure and remains recorded.

Only observed X,T,Y from the declared sources are forwarded to native GP fitting. The full procedure receives the common TrainingView; the pilot controls receive only their included sample. The compatibility TrainingView entry point also neither inspects nor hashes excluded observations. Truth is evaluator-only; query covariates are prediction-only. Known trial propensity0.5 is provided by the interface; the native algorithm does not use it for fitting. For the full procedure only, the interface's nonempty `rct_tune` field aliases the first two already charged fitting observations and is ignored. Controls receive no tuning object. It is a compatibility placeholder, **not held-out validation**. No additional unique outcomes are revealed, no candidate is selected, and no final validation refit occurs. Target-reference rows are absent. Perturbing this placeholder or an excluded source must not change predictions.

Each case uses the same estimator seed for all slots and the same seed across delta/B. This permits two exact source-access checks: RCT-only prediction hashes must be identical across delta at fixed B/seed, and OBS-start hashes must be identical across B at fixed delta/seed. No result is accepted merely because these controls improve risk.

## Required checks and execution limits

Before comparative fits, [verify_gp_source_pilot.py](../scripts/verify_gp_source_pilot.py) checks direct native empty-source calls against fresh adapters, excluded-source/tuning/reference mutations, repeated seeds, RCT-only rho invariance, and the same-model OBS-start posterior scaling. Independent data checks cover hand-evaluated target/bias values, potential-outcome contrasts, common trial law, nested samples, shared inputs and unique-label accounting. Fixed float64 fidelity tolerances are absolute1e-10/relative1e-8; analytic identities use absolute1e-12. Failed attempts remain retained; thresholds are not relaxed after inspecting outputs.

Substantial work runs on GSDS CPU-only P2 with one numerical thread. The observed earlier d5/eval2048 native GP profiles took2.7s at OBS512/RCT128 and16.4s at OBS2048/RCT512. This pilot has d2/eval1024 and no more than1024 fitting rows, but its exact costs are unmeasured before execution. Each fit has an outer180s process-group limit. Shards request1CPU/4GiB; up to four run concurrently. The allocation is conservative relative to the prior largest GP worker high-water mark588036KiB, which is not a simultaneous whole-job peak. Exact job IDs, wall limits and shared paths are operational records, not claims about final runtime.

An exclusive manifest binds configuration/driver/native/package/environment identities before shards start. Stable case IDs and immutable terminal records permit exact-identity resumes. A completed failure or timeout remains terminal; no selective retry, seed replacement, favorable score imputation or silently omitted recipe is allowed. Required child failures propagate to a nonzero shard exit after recording the requested cases. Analysis authenticates the full expected record union and saved input/prediction hashes before reporting; raw arrays and logs remain on the server.

## Outcomes and interpretation

Report each procedure's trial-population CATE MSE, fit-plus-prediction time, charged OBS/RCT labels and failure counts separately for all six conditions. Per-case MSE is mean((prediction_i-tau(query_X_i))^2) over the1024 common query rows. The fit timer includes adapter/native loading, provenance inspection, fitting and prediction inside the method call; a separate launcher timer includes interpreter startup, input preparation and artifact handling. Compare full-minus-RCT-only and full-minus-OBS-start within each of32 shared seeds. Negative differences favor full. Both contrasts concern the same target and observed queries.

For each of the12 condition/contrast summaries, report the paired mean, MCSE and approximate pointwise95% t31 interval only when all32 required pairs succeed. With an incomplete pair set, retain successful-only means labeled conditional on success and withhold that contrast's interval/MCSE. Query rows and conditions are not independent replicates. No global raw-MSE average, p-values, joint superiority decision or new Holm family is produced. Intervals describe Monte Carlo uncertainty on this development panel and are not simultaneous. The earlier catalog's472 withheld decisions remain withheld.

Independent post-fit verification recomputes targets and scalar scores from saved query/prediction arrays outside the producer, authenticates all terminal records and source-access identities, and reaggregates the12 contrasts. The first fixed seed30000 in every condition/slot (18 fits) is freshly replayed once, with predictions checked at absolute1e-10/relative1e-8 and scalar MSE at absolute1e-12/relative1e-11; no score is replaced. A separate execution smoke outside the comparison seed panel must preserve its three terminal records byte-for-byte on resume.

The pilot is complete when all576 terminal records and both paired contrasts per condition are accounted for, controls obey their source-exclusion invariants, independent reconstruction passes, and results/costs/failures are readable. Favorable performance is not a completion criterion. Broader method coverage, controlled tuning and an independently specified mechanism holdout remain subsequent questions.
