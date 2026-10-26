# Data and evaluation protocol

This is the implemented protocol for the current package. The [v0.3 comparison design](V03_DESIGN.md) specifies future native-target comparisons and selection rules; it does not modify this API or reinterpret earlier results.

The estimand for each scenario is the target-population conditional average treatment effect, `tau_target(x)=E_target[Y(1)-Y(0)|X=x]`. The target population is part of the scenario definition. A randomized trial identifies effects in its enrolled population; random treatment assignment alone does not establish transport to another population. This distinction matters in the hidden-participation scenarios.

## Roles and information boundary

Each replicate generates independent role draws from a fixed mechanism. A `SourceSample` contains observed `X`, binary `T`, continuous `Y`, source and role labels, and known per-row trial randomization probabilities when applicable. OBS propensity is evaluation truth and is not supplied as a known probability to estimators.

| Role | Estimator access | Purpose |
| --- | --- | --- |
| `obs_train` | Yes | Observational fitting data |
| `rct_train` | Yes | Trial fitting data with known randomization probability |
| `rct_tune` | Yes | Independent trial data for declared tuning rules |
| `target_reference` | Optional, covariates only | Independent target-law covariates when `n_target_reference>0` in a supporting structural, Gamma or empirical configuration; absent by default |
| `rct_test` | No | Reserved for held-out factual diagnostics; unused by the current CATE score and never estimator or hyperparameter selection |
| `eval_X` | Covariates passed for prediction | Independent target-law evaluation rows |
| `eval_tau`, `SimulationTruth` | No | Evaluator-only CATE and mechanism truth |

`problem.training_view()` returns only `TrainingView(obs_train, rct_train, rct_tune, target_reference)`. A method implements `fit_predict(training, eval_X, *, seed)` and returns one finite prediction per evaluation row. Arrays are copied and read-only. This API prevents accidental access through the supplied objects; it is not an operating-system sandbox for arbitrary plugin code. Do not capture a `SimulationProblem` or truth in an adapter closure.

Query `eval_X` is prediction-only. Do not fit preprocessing, estimate target densities or adapt model parameters on those evaluation rows. Use observed training data or an explicitly enabled independent `target_reference` for those operations. This rule makes the evaluation distribution available through a declared sample rather than through evaluation queries. A future transductive protocol would need its own declared information contract.

Use fixed declared defaults, an internal split of training data, or a disclosed tuning rule based on observed training/tuning data. A trial factual prediction loss can tune outcome models, but it is not an oracle CATE loss. Keep each method's data use visible: the optional official DataFusionPPI adapter uses `rct_train` and `obs_train`, and the upstream method makes its own internal splits; it does not consume `rct_tune`.

## Repeated settings and sample budgets

A scenario's `kind` and `id` choose its generator preset. Its optional `label` names a particular setting, allowing the same preset to appear at several sample sizes or parameter values in one run. Labels contain ASCII letters, digits and underscores; the resulting key is `kind:id@label`. An unlabelled scenario retains its original `kind:id` key and configuration representation. Every key within a run must be unique. See the [configuration example](EXTENDING.md#comparing-settings-of-the-same-generator).

Changing a label alone does not change the generated observations, mechanism seed or method seed. Changing sample sizes or parameters does change the experiment. Matching sample seeds pairs the random streams across settings; it does not guarantee that every smaller sample is an exact prefix of a larger one. Labels are conditions, not independent replicates, so repeated copies of one setting must not increase the reported number of seed blocks.

The report shows sample sizes separately for OBS training, RCT training, RCT tuning, evaluator-only RCT test, target reference and evaluation. It also shows `RCT train + tune`, the randomized-outcome budget made available to methods. This is an access budget: a fixed recipe may leave tuning data unused. Neither unused tuning data nor evaluator-only test data silently becomes additional fitting data.

## Confounding and source outcome shift

For arm `a`, let `m_obs,a(x)=E[Y|T=a,X=x,OBS]`, `mu_obs,a(x)=E[Y(a)|X=x,OBS]`, and `mu_target,a(x)=E_target[Y(a)|X=x]`. The evaluator distinguishes:

```
confounding_bias_a       = m_obs,a - mu_obs,a
source_shift_a           = mu_obs,a - mu_target,a
regression_discrepancy_a = confounding_bias_a + source_shift_a
```

Thus the OBS regression contrast minus target CATE is `(confounding_bias_1-confounding_bias_0) + (source_shift_1-source_shift_0)`. A discrepancy may come entirely from source-specific outcomes even with randomized treatment. CorNet and the CVCI linear translation illustrate this case. Conversely, arm-common source shifts affect outcome regressions while canceling from the causal contrast.

Structural truth also separates trial participation effects from OBS confounding. In C07 a hidden prognostic participation variable shifts the trial baseline while preserving the CATE; in C08 that variable modifies treatment effects, so trial and target CATE differ. Covariate overlap and conditional-effect transport are different assumptions.

Truth is derived from the declared structural equations, not from any method's fit. Some hidden conditional moments use deterministic Gaussian quadrature. Finite quadrature order is a numerical approximation: extreme parameter settings require a stability check against higher orders or an independent calculation. The known CATE must not be used to choose a favorable mechanism, method, or hyperparameter after observing comparisons.

Some structural covariate-density ratios use normalization over the supplied evaluation sample. Treat these as numerical diagnostic quantities; do not mistake them for exact, independently normalized population densities or expose them to ordinary estimators.

## Scores, pairing, and failures

For one replicate and target evaluation rows `i=1,...,n_eval`:

```
cate_mse = mean_i((predicted_tau_i - true_tau_i)**2)
root_pehe = sqrt(cate_mse)
```

`mean_root_pehe` averages replicate root-PEHE values; `root_mean_mse` takes the square root after averaging replicate MSE. They are different summaries and are named separately in output. These are finite evaluation-sample estimates under the declared target law, not exact population integrals.

All methods in a scenario/seed receive the same generated samples and evaluation covariates. Pairwise differences are method minus reference, so negative values favor the method. For an overall comparison, scenario differences are averaged within each seed before uncertainty is computed across seed blocks. This gives each requested scenario equal weight and preserves the within-seed pairing. Declare any other scenario weighting before running it.

The summary reports a mean and approximate two-sided Student `t` interval across independent seed blocks. The implementation uses a conservative critical-value table; a single seed block has no uncertainty interval. Evaluation rows, scenarios, and methods are not treated as independent replicate blocks. The interval describes simulation-seed variability conditional on the selected mechanisms and settings; it does not account for how those scenarios were selected.

The generated Markdown report shows each scenario's scores, failed/missing cells, and paired CATE-MSE differences against the declared reference. These intervals are pointwise and have no multiplicity adjustment. An aggregate table remains available as a descriptive equal-cell average on the original outcome scales; large-scale scenarios can dominate it. Use the condition-specific results to interpret failures and borrowing benefits rather than treating the raw-scale aggregate as a universal ranking. `summary.json` additionally contains paired root-PEHE comparisons and the overall seed-block summaries.

Each requested `(scenario, seed, method)` cell must have one success or failure record. Missing and failed cells are counted explicitly, never scored as zero. Descriptive means of successful cells carry an incomplete-coverage label. Paired inferential comparisons are withheld unless both methods cover the entire requested grid. Duplicate or unrequested records are rejected.

## Reproducibility and interpretation

Declare sample budgets separately for each role, the number of mechanisms and sample seeds, each method's tuning budget, and compute resources before a substantive comparison. Record CPU/thread use, memory and time limits, and any upstream internal splits. Comparable observed data access does not imply equal optimization cost. The runner records outcomes and provenance; external job controls remain responsible for enforcing compute limits.

Record the complete configuration, mechanism seed, sample seeds, package and source versions, method options, optional-method provenance, and command. Mechanism seeds fix random coefficients or representations; sample seeds change draws conditional on that mechanism. Changing both answers a broader question and should be reported as such. Same-seed reproducibility applies to this package version and its numerical environment, not to an upstream experiment's RNG stream.

Set `[run].purpose` to `"smoke"` for execution checks or `"reference"` for a declared comparison study. The default is `"smoke"`; omitting it preserves the original configuration hash. A reference purpose is recorded in the manifest and report without automatically asserting adequate Monte Carlo precision, confirmatory design or scientific validity. State those choices and limitations in `[run].scope` and the study protocol before inspecting its results. Changing the purpose changes run identity; resume still requires the exact configuration and source/environment fingerprint.

Report results by scenario and distinguish published translations from structural mechanisms. The default collection is a chosen set of stressors, not a sampled population of all causal problems. A smoke run checks the software path; it does not establish scientific performance, reliable uncertainty, or universal method superiority. Prespecify the comparison grid and reference method before running a substantive study.
