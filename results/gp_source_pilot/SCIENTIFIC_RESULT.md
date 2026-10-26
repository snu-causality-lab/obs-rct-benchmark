# Observational information can hurt this fixed GP procedure

**Development pilot; frozen producer 0.3.0a2.** Adding OBS data lowered the full procedure's mean trial-CATE error at discrepancy levels delta0 and delta1, but raised it at delta2 relative to the matched RCT-only GP. All **576 requested fits succeeded**, with **zero failures and zero missing records**. The 32 sample-seed blocks, six conditions and three source-access procedures were declared before comparative fitting. These are conditions of one designed mechanism, not six independent mechanism classes.

## The three-way comparison changes the interpretation

At delta2 and RCT128, full-minus-RCT-only mean MSE is **+0.0399937**, with approximate pointwise 95% interval **[0.0111285, 0.0688589]**. At RCT512, it is **+0.0172488 [0.00888523, 0.0256123]**. In both conditions, the complete procedure using OBS is worse on average than the same native GP fitted only to the trial. This is negative transfer for this fixed recipe and target.

Full-minus-OBS-start means are negative in every condition. That comparison alone would therefore miss the adverse OBS contribution relative to the matched RCT-only alternative. The two comparisons answer different questions: adding OBS at the same B randomized outcomes, and adding randomized outcomes relative to a zero-RCT starting prediction.

The OBS-start is a **prior-shrunken trial-task prediction conditioned on OBS only**. With rho0.5 and kappa1e-9, its conditional mean has factor rho/(1+kappa) relative to the same fitted model's OBS-task mean. It is neither the unshrunk OBS arm contrast nor an optimal OBS-only estimator; it can be a weak starting reference. Improvements over it do not establish superiority over stronger OBS-only procedures.

## All six conditions

Negative differences favor full. Intervals below are approximate **pointwise, unadjusted paired t31 intervals** over the 32 shared sample-seed blocks. They are descriptive Monte Carlo summaries, not simultaneous intervals or joint superiority decisions. No p-values, new Holm decisions or global raw-MSE ranking are introduced.

| Delta | RCT rows | Full MSE | RCT-only MSE | OBS start MSE | Full−RCT [pointwise 95% CI] | Full−OBS start [pointwise 95% CI] |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 128 | 0.143201 | 0.212625 | 0.3409 | -0.0694242 [-0.0971763, -0.041672] | -0.197699 [-0.229091, -0.166308] |
| 0 | 512 | 0.0647022 | 0.0712551 | 0.3409 | -0.00655291 [-0.00958201, -0.00352382] | -0.276198 [-0.290495, -0.261901] |
| 1 | 128 | 0.164762 | 0.212625 | 0.361698 | -0.047863 [-0.0784679, -0.0172582] | -0.196936 [-0.23115, -0.162723] |
| 1 | 512 | 0.0638492 | 0.0712551 | 0.361698 | -0.00740584 [-0.0115086, -0.00330305] | -0.297849 [-0.31583, -0.279868] |
| 2 | 128 | 0.252619 | 0.212625 | 0.511709 | 0.0399937 [0.0111285, 0.0688589] | -0.259091 [-0.29848, -0.219701] |
| 2 | 512 | 0.0885038 | 0.0712551 | 0.511709 | 0.0172488 [0.00888523, 0.0256123] | -0.423205 [-0.446028, -0.400383] |

The RCT-only mean is identical across delta because the trial law, realized trial sample, method seed and evaluation queries are held fixed. The OBS-start mean is identical across B because the included OBS sample, method seed and queries are identical. The corresponding **64 RCT-only and 96 OBS-start source-exclusion groups** pass exact prediction-hash checks.

## What was held fixed

All source/query covariates are Uniform[-1,1]^2. The trial target is `tau(x)=1+0.5*sin(pi*x0)+0.25*cos(pi*x1)`. Bias levels delta0,1,2 change only the OBS conditional arm-mean discrepancy through `b(x)=cos(pi*x0)*sin(pi*x1)`. The trial law, target, observed conditional noise variance and query population are unchanged. OBS512 and RCT128/512 share paired base observations across conditions; RCT128 is a prefix of RCT512. Each replicate uses 1,024 common independent query covariates.

The documented SCM has common causal CATE across sources but confounded OBS treatment: U_OBS is Bernoulli0.5 with T_OBS=U_OBS, whereas U_RCT=0.5 and trial treatment is randomized. Latent-stratum positivity in OBS is absent; positivity given observed X is 0.5. This designed stress does not represent every observational population or guarantee all native modeling assumptions. See the [prospective protocol](../../docs/GP_SOURCE_PILOT.md) and [frozen configuration](../../configs/gp_source_pilot.toml).

All three procedures use the pinned native two-arm GP, all two covariates, rank2, fixed rho0.5, fixed kappa1e-9 and native kernel/noise optimization defaults. RCT-only excludes OBS rows from fitting and shared-hyperparameter optimization; OBS-start excludes RCT rows and predicts task0. Each control receives only its included SourceSample, with no excluded-source labels in its method arguments. Full uses B randomized and512 observational outcomes; RCT-only B and0; OBS-start0 and512. Full's two tuning-role rows alias already charged fitting rows and are unused. The complete-procedure contrasts include different fitted hyperparameters, so they do not isolate a fixed-hyperparameter posterior update or explain which internal component caused harm.

## Independent reconstruction and its limits

The producer/configuration were published at [a645821](https://github.com/snu-causality-lab/obs-rct-benchmark/commit/a645821ea187cea2bb95de535d2b1bee5564bb41) before comparative fitting. [producer_binding.json](producer_binding.json) preserves the pre-fit binding; its `not_started_when_bound` field describes when that record was made. [manifest.json](manifest.json), [summary.json](summary.json) and [the independent verification](verification/verification.json) retain the actual source/configuration identities, all terminal-record hashes and coverage.

Independent verification reconstructs all576 scalar risks from saved query/prediction arrays with separately written math.sin/cos target arithmetic, checks all18 recipe means and12 paired summaries, and authenticates the160 source-exclusion groups. A fixed subset of **18 first-seed fits** is freshly replayed once; every maximum absolute prediction difference is **0**. That is the [replay's stated subset](verification/replay.json), not a refit of all576 fits. Raw observed/evaluation/prediction arrays, complete process logs and unsuccessful-attempt diagnostics remain on GSDS. This producer does not save fitted GP model pickles or complete optimized kernel state.

The [execution summary](checks_summary.json) also records371 passed frozen-producer tests and one external-CorNet skip, with that integration separately passing in its configured environment. Native/data preflight passed13 tiny fits. A completed-run resume preserved every one of the576 terminal record files byte-for-byte without refitting; this claim concerns terminal records, not newly appended shard-status metadata. One outside-panel helper-smoke verifier invocation failed because its configuration was not the frozen comparison panel; its24 numerical fits, analysis and rendering had succeeded. That orchestration mismatch is retained and was not introduced as a planned negative test. Main-panel verification subsequently passed with unchanged scientific source.

## Scope of the result

This is a development diagnostic on a specified generator and a fixed borrowing recipe. It is not an unseen-mechanism evaluation, a source-paper table reproduction, a study of selected/optimal rho, or a result about all fusion methods. The observed positive/negative transfer pattern supplies a concrete controlled example; changing the generator, prior, model or selection rule can change it. The existing31-preset catalog, its21 terminal PPI failures and its472 withheld Holm decisions remain unchanged.

The [figures](rendered/README.md), [18 recipe-condition rows](rendered/condition_scores.csv), [12 paired contrasts](rendered/paired_contrasts.csv), [576 compact scalar records](rendered/scalar_records.csv) and [execution costs/label counts](rendered/cost_summary.csv) retain every condition. Cost summaries distinguish successful-fit timing from all-attempt launcher timing and separate process RSS marks from simultaneous whole-job memory. Their renderer/source identities are in [figure_source.json](rendered/figure_source.json).
