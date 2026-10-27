# Native/default trial-CATE panel

This is a prospective development comparison on six already inspected mechanisms. It compares fixed algorithms at equal randomized-outcome access and separately compares OBS access within the same GP. It is not a reproduction of a published performance table, an unseen-mechanism evaluation, or the controlled-tuning campaign.

The executable specification is [native_panel.toml](../configs/native_panel.toml). Its resource status must be `preflight_reviewed` and its reviewed evidence digest bound before `init` accepts it. Review authenticates execution, costs and classified terminal outcomes; it does not promise every estimator completes within budget. The immutable campaign manifest then binds configuration, resolved cases, recipe options, protocol, execution/checker/analysis source and runtime identities **before comparative fitting**. A source or resource change requires a new campaign directory and manifest. Existing studies keep their original protocols and evidence.

## Conditions and target

Each condition scores trial-population CATE on 2,048 independent prediction queries drawn from that trial's covariate law. Assignment is randomized with known probability 0.5. Native dimensions, equations and fixed mechanism seeds are preserved. The fixed grid has six mechanisms × two RCT budgets × 64 sample seeds × eleven recipes = **8,448 requested fits**, with no scientific N/A rows.

| Mechanism | Trial covariates | Role and restrictions |
| --- | --- | --- |
| M0 | Normal, five covariates | Clean OBS control with a shared causal effect. |
| M2 | Normal, five covariates | Hidden treatment confounding; shared effect, nonlinear discrepancy. Linear/quadratic effect bases are misspecified. |
| M10 | Normal, five covariates | Hidden confounding and an OBS causal-effect shift. Shared-source-effect restrictions are deliberately false. |
| C08, **new trial target** | Normal, five covariates | Hidden study participation changes the trial effect. OBS is unconfounded for another causal contrast. |
| Integrative R native KPS code | Uniform on [-1,1], one covariate | Native quadratic effect; the quadratic effect basis is correctly specified and the linear basis is misspecified. OBS covariates are normal. |
| CorNet native `n_conf` | Normal, 25 covariates | Nonlinear source-effect shift, fixed mechanism seed 2025; neural construction favors a representation used by one comparator. No hidden treatment confounding. |

C08 is explicitly named `addendum:C08@trial_target`. Its scored effect is `tau_trial(X) = tau_target(X) + 2 E[Z | participant]`, where `Z` is standard normal and participation has probability `sigmoid(2Z)`. The canonical catalog C08 scores `tau_target`, so its old scalar scores cannot be substituted. An independent adaptive integral checks the producer's selected-normal moment. The panel binds 256 Hermite nodes and absolute tolerance 1e-12; the factory's original 64-node participation moment failed this comparison by 5.73e-10. Only evaluator precision changes: native observations and frozen studies remain unchanged. The other five native evaluation targets and laws already coincide with their trial targets. The independent checker reconstructs all six effect maps without calling producer truth helpers; sampler checks establish configured laws and numerical/support sanity, not a statistical proof from empirical moments.

The [data resolver](../scripts/native_panel_data.py) calls the unchanged [structural definitions](../src/obs_rct_benchmark/structural_dgp.py) and [paper definitions](../src/obs_rct_benchmark/paper_dgp.py). M0/M2/M10 and canonical C08 have effect `1 + 0.5 tanh(X[:,1]) + 0.25 sin(X[:,2])`, using zero-based coordinates. Native KPS has `1 + 2x + 0.75x²`; native CorNet uses its fixed-seed base-plus-correction networks. Mechanism/signal seeds stay 2025. The campaign manifest includes all twelve resolved factory configurations, trial-target/access metadata and the enumerated 72 comparisons. [Independent equations](../scripts/verify_native_panel.py) provide separate implementations, including CorNet initialization and forward arithmetic.

M0/M2/M10 retain different observation mechanisms even where trial effect surfaces coincide. Conditions are selected to represent clean borrowing, confounding, effect disagreement, hidden participation, polynomial specification and a high-dimensional native construction. The remaining catalog mechanisms, poor trial-support/external-target settings, heavy tails, variable randomization, real data and new method families remain extensions. Six mechanisms do not constitute comprehensive coverage of those questions.

## Recipes and information

| Role | Fixed recipe IDs |
| --- | --- |
| RCT references | `rct_causal_forest`, `rct_t_extra_trees`, `rct_dr_extra_trees`, `causal_icm_rct_only` |
| Fusion recipes | `cornet_official`, `datafusionppi`, `integrative_r_linear`, `integrative_r_quadratic`, `causal_icm_fixed` |
| Diagnostics | `obs_t_extra_trees`, `naive_pooled_t_extra_trees` |

These are eleven recipes, not eleven distinct fusion approaches. Integrative R contributes two effect bases. CorNet retains 500 source and 200 correction epochs; PPI uses its admitted DR recipe and native internal splitting; Integrative R retains its native SCAD search; the GP fixes rho at 0.5 without selection. Causal forest retains its admitted honest native defaults, including 2,000 causal trees and native nuisance fitting. All options are bound in the configuration and native adapter provenance. No benchmark-specific search, restarts, outcome-dependent recipe selection, fallback estimator or refit is allowed. Internal cross-fitting, OOB fits and model selection reuse charged training labels.

For each seed, every applicable algorithm receives identical bytes for its allowed samples and prediction queries. RCT references receive only RCT observations; the OBS diagnostic receives only OBS observations; fusion and pooling receive both. Strict source controls use their source-sample API. Compatibility fields required by generic wrappers contain aliases of permitted observations, never the excluded source. An ignored `rct_tune` field aliases the first two charged RCT training rows for fusion recipes. It supplies no extra outcomes and performs no selection. This is an explicit study-driver exception to the generic runner's independent-tuning semantics.

| Outcome access | Fusion / pooling | RCT references | OBS diagnostic |
| --- | --- | --- | --- |
| Unique OBS training outcomes | 512 | 0 | 512 |
| Unique RCT training outcomes | 128 or 512 | 128 or 512 | 0 |
| Additional external tuning outcomes | 0 | 0 | 0 |
| Target-reference sample | 0 | 0 | 0 |
| Evaluator queries / analytic truth | 2,048 queries at prediction / hidden truth | Same | Same |

The existing factories require four RCT tuning and four RCT test rows. They may generate these eight extra outcomes internally; the driver discards them and never supplies them to a worker or selector. Distinguish generated counts from supplied unique outcomes. Trial truth, discarded outcomes and evaluator-only metadata stay outside method payloads. Native internally consumed row counts and process artifacts are retained where exposed; the interface proves permitted access, not a universal trace of every arithmetic operation in a dependency.

Algorithm-versus-forest comparisons measure competitiveness at equal RCT access. Only **GP Full versus GP RCT-only** isolates access to OBS within an admitted common backbone. Both use the same prespecified GP seed, but shared kernel/noise hyperparameters are fitted using their own permitted data; this is not a posterior-update comparison at fixed fitted hyperparameters. CorNet, PPI and Integrative R lack admitted matched source-excluded controls in this panel. In particular, native Integrative R's `rlasso` uses L1 whereas the fusion recipe uses SCAD. Neither forest nor that L1 recipe would isolate the fusion component.

## Replication and analysis

Sample seeds **50000–50063** are reserved before fitting; profile seed **49999** is excluded. Method seeds are deterministic functions of scenario and sample seed. GP Full and RCT-only share the `causal_icm_fixed` seed label. Conditions and budgets may share seeds, so comparisons across conditions are dependent. Each condition's 64 seed blocks supply the replication unit. The inference model assumes independent identically distributed draws of these blocks, including method randomness and fresh evaluation queries. A deterministic pseudorandom implementation does not itself prove that model.

Primary outputs are condition-specific CATE-MSE, root-PEHE, seed distributions and paired mean differences. Mean-risk standard errors, t intervals, bootstrap intervals and mean-risk significance decisions are **withheld**: the fitted-risk moment and precision conditions are not established for these native recipes. No global ranking averages raw MSE across different target scales. Incomplete paired means are withheld. Successful-only method means are explicitly conditional on success.

The secondary inferential endpoint is the probability that a method has **strictly smaller finite-query CATE-MSE** than its reference on a seed block. It concerns the 2,048-query score, not exact population risk or expected MSE improvement. An exact numerical tie is a nonwin; wins, losses and ties are reported separately. Many ties can reduce strict-win probability without establishing harm.

One family contains **72 fixed comparisons**: five fusion recipes versus causal forest in twelve conditions (60), plus GP Full versus GP RCT-only in twelve conditions (12). For each, test `H0: p_win <= 0.5` against `p_win > 0.5` with the exact upper binomial tail at 0.5. Apply Holm's step-down adjustment over all 72 at familywise alpha 0.05. A rejected null supports a majority of strict wins, not a lower expected error or a universal superiority claim. The implementation uses [SciPy's exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

Report two-sided pointwise 95% [Clopper–Pearson intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html). Separately, Hoeffding's inequality and a union bound give simultaneous intervals `p_hat ± 0.25`, clipped to [0,1], because `2 × 72 × exp(-2 × 64 × 0.25²) < 0.05`. This is deliberately coarse precision for large probability differences. It supplies no mean-risk precision guarantee. There is no adaptive seed extension, early significance stopping or change of endpoint after looking at results. A narrower comparison requires a separately frozen study.

If any required family pair lacks a successful fit, **all 72 Holm decisions and the simultaneous precision statement are withheld**. Complete pointwise estimates can remain descriptive; the family is never shrunk. Failures unrelated to family contrasts still remain visible in full coverage and method summaries. No failed method becomes scientifically inapplicable after execution.

## Execution and retained evidence

The study driver has separate `profile`, `init`, `run` and verification stages. Profiling attempts one observed-data-only, unscored maximum-size native CorNet case per recipe, preserving d25 for every algorithm. It may expose a cost limitation before the campaign, never an accuracy-based exclusion. Production caps and allocation sizes are fixed only after this check. The compact [preflight record](../results/native_panel_preflight/README.md) records measured scope and remaining resource uncertainty. A single profile is a cap check, not a full-grid runtime forecast or proof of worst-case memory use.

**Observed limitation:** the 25-dimensional Integrative R quadratic profile reaches its original 1,200-second native deadline. Its 350-column basis entails a 702-column joint design and 44 SCAD cross-validations. Preserve the default basis, search and deadline, retain this budget-limited row, and record any comparative timeouts as failures. Do not turn it into a scientific N/A, drop its comparisons, enlarge its budget after seeing comparison scores, or assert that preflight established numerical completion. The primary descriptive comparison remains defined; this known resource risk may prevent the entire secondary family from receiving decisions.

The fixed GSDS plan has **128 shards**, six seed/condition cases per shard, one CPU/thread and **8 GiB requested per job**, a six-hour job limit and at most eight simultaneous jobs. Submit waves of at most eight jobs. Per-method outer limits are in the configuration; their sum across six cases is below the job deadline with setup margin. The preflight records cgroup counters, but a `memory.max=max` value does not establish that GSDS enforces the requested memory. No GPU is requested. The raw-artifact storage planning allowance is 10 GiB; actual size remains to be measured. These are resource ceilings and planning values, not a full-campaign completion-time forecast.

Use one fresh worker process and one numerical thread per fit, with explicit Slurm CPU/RAM/time allocation and per-fit timeout. The adapter's internal time limits remain 600 seconds for causal forest and 1,200 seconds for Integrative R; they constrain its native R subprocess. The outer per-recipe worker limits in `[resources.method_fit_timeout_seconds]` include startup, verification, fitting and prediction. Whichever limit occurs first applies. An inner limit is a terminal failure with native timeout provenance; an outer limit is a terminal `timeout`. Both are reported and withhold a required family contrast. Kill the complete worker process group after completion or timeout, including remaining R children. Bound source and runtime identities before fitting; route GP and other recipes through their recorded compatible interpreters without modifying retained environments. Bind Python executables, package versions, numerical runtime information and adapter source/build evidence to the manifest. These checks do not authenticate an entire operating system or prove compiler correctness.

Retain observed arrays once per case, separate evaluator truth, hashes, predictions, native provenance, worker logs, wall/CPU times and clearly labeled process RSS measurements on the server. Integrative R's complete native stdout/stderr are retained as diagnostics; its temporary bridge files are reconstructed from the source and input arrays. Causal forest additionally retains the native artifacts configured by `GRF_ARTIFACT_ROOT`. GP fitted model objects are not retained. Per-fit terminal JSON files are atomic and exclusive. Resume authenticates the immutable manifest and artifacts, skips every terminal success/failure/timeout, and rejects duplicate, extra, truncated or inconsistent records. Interrupted nonterminal attempts remain preserved. No terminal failure is retried or replaced.

Independent verification reconstructs trial truth and score arithmetic from retained predictions; it is distinct from native adapter fidelity. Analytic truth uses absolute tolerance 1e-12 and zero relative tolerance. Independent score reductions use `math.isclose` with absolute and relative tolerances both 1e-12, fixed before comparative scoring; the relative bound accommodates floating reduction error when a method has large finite loss. These verification tolerances do not change the strict numerical-win definition. The analyzer requires the complete requested terminal union and performs those artifact checks before aggregation. Full arrays, environments and logs stay on research storage; GitHub receives source/configuration, compact preflight evidence and eventual compact study summaries. Dropbox and Overleaf receive manuscript assets only. The existing integrated report is the four completed studies, not prospective panel results.

## Reproduce the frozen execution plan

Prepare a private runtime JSON with `schema_version=1` and `methods`, mapping **all eleven recipe IDs** to an absolute `python` interpreter and an `env` object. A generic entry is `"rct_t_extra_trees": {"python": "/absolute/shared/environment/bin/python", "env": {}}`. CorNet needs `CORNET_SOURCE`; Integrative R needs `INTEGRATIVE_R_SOURCE` and `INTEGRATIVE_R_RSCRIPT`; GP needs `CAUSAL_ICM_SOURCE` and its pinned numerical environment; forest needs `GRF_SOURCE`, `GRF_LIBRARY`, `GRF_BUILD_MANIFEST`, `GRF_RSCRIPT` and an existing `GRF_ARTIFACT_ROOT`. The [adapter instructions](V03_ADAPTERS.md) and [forest instructions](CAUSAL_FOREST.md) specify these dependencies. Use shared server paths, no credentials. The driver checks the actual runtime bindings; a mapping file alone is not proof of installation.

Run on an allocated GSDS CPU node with `GSDS_WORK_ROOT` set to its declared shared work root, one CPU and one thread for OMP/MKL/OPENBLAS/NUMEXPR, sufficient explicit RAM, and caches/temporary files on shared research storage. First profile every declared method using the reserved native CorNet d25 case obtained from `profile_cases(config)`, with a separate new output directory per method:

```bash
python scripts/run_native_panel.py profile --config configs/native_panel.toml \
  --runtimes "$PANEL_RUNTIMES" --protocol docs/NATIVE_PANEL.md \
  --output "$PANEL_PROFILE_OUTPUT" --case-json "$PANEL_PROFILE_CASE" \
  --method "$PANEL_METHOD"
python scripts/run_native_panel.py verify --output "$PANEL_PROFILE_OUTPUT"
```

Review retained costs and failures, publish the compact preflight summary, set final resource caps and bind that summary's digest. An unexplained implementation/provenance failure blocks initialization. The reviewed proof explicitly distinguishes successful profiles from the retained native Integrative R quadratic timeout; it authorizes that disclosed resource limitation, not an all-success claim. The published evidence supplies this completed stage. Changing budgets or rerunning profiles creates new evidence and requires a new resource binding before a new campaign. Then initialize from a source checkout whose bytes remain fixed:

```bash
python scripts/run_native_panel.py init --config configs/native_panel.toml \
  --runtimes "$PANEL_RUNTIMES" --protocol docs/NATIVE_PANEL.md \
  --output "$PANEL_OUTPUT" --shard-count "$PANEL_SHARD_COUNT"
python scripts/run_native_panel.py run --output "$PANEL_OUTPUT" \
  --shard-index "$PANEL_SHARD_INDEX" --resume
python scripts/run_native_panel.py verify --output "$PANEL_OUTPUT"
python scripts/analyze_native_panel.py --output "$PANEL_OUTPUT" \
  --analysis "$PANEL_ANALYSIS/summary.json"
```

Shard count, source bindings and caps belong to the immutable plan; restarting a shard does not permit changing them. Preflight verification is not permission to imply this comparative campaign has already run. The [broader design](V03_DESIGN.md) retains remaining method families, matched controls, external targets and the separate observable-selection lane.
