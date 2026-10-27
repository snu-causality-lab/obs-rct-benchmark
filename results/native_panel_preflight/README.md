# Native/default panel preflight

**Protocol and execution preparation, without comparative scores.** The [six-mechanism trial-CATE panel](../../docs/NATIVE_PANEL.md) requests 8,448 fits from eleven fixed recipes, two RCT budgets and 64 reserved seeds. This packet records correctness, input isolation and unscored resource checks. It is not another accuracy study or a performance update to the integrated report.

The [machine-readable evidence](summary.json) distinguishes **ten successful profiles and one native deadline failure**. All eleven terminal attempts authenticate against their original source, runtime, observed-input and artifact bindings, with no scores persisted and no surviving worker descendants. Execution-contract review passes; all-profile numerical completion does not. The [configuration](../../configs/native_panel.toml) binds this evidence by SHA256. The [current source freeze](freeze_v2.json) binds the protocol, all runtime identities, twelve resolved conditions, 72 comparisons and zero comparative fits started. The [current GSDS checks](checks_v2.json) record **563 passing tests**, no failures or skips, with the retained log hash and a reproduced deadline-cleanup regression. The [initial unexecuted manifest](freeze.json) and its [559-test checks](final_checks.json) remain preserved.

## Resource measurements

One fixed native CorNet fixture per recipe: d25, OBS512, RCT512, 2,048 queries and reserved profile seed 49999. Each task requests one CPU/thread and 8 GiB; at most two profile tasks run concurrently. Times include fresh worker startup, verification, fitting and prediction, but exclude the parent driver's initial runtime inspection. These are single-fixture costs, not campaign forecasts.

| Recipe | Launcher wall seconds (`launcher_seconds`) | Outcome |
| --- | ---: | --- |
| RCT causal forest | 19.158 | Success |
| RCT ExtraTrees T | 2.353 | Success |
| RCT ExtraTrees DR | 2.357 | Success |
| OBS ExtraTrees T | 2.083 | Success |
| Naive pooling ExtraTrees T | 2.024 | Success |
| Official CorNet | 7.732 | Success |
| DataFusionPPI DR | 2.300 | Success |
| Integrative R linear | 23.971 | Success |
| Integrative R quadratic | 1,205.255 | Native 1,200-second deadline; terminal failure |
| Fixed-rho GP Full | 23.352 | Success |
| Matched GP RCT-only | 3.950 | Success |

The quadratic basis expands 25 covariates to 350 columns. The author procedure uses a 702-column joint design and 44 SCAD cross-validations, including a branch with 351 unpenalized effect coordinates. Code inspection found no repeated basis expansion. The original search, basis and native 1,200-second inner deadline are preserved; the final outer launcher allowance is 1,320 seconds. That larger outer allowance does not extend the native deadline. A separate diagnostic attempt also reached the same deadline; neither attempt is replaced or presented as an accuracy score.

The maximum recorded enclosing cgroup peak is 500,813,824 bytes. Process RSS and cgroup charged-memory peaks measure different things; separate parent/child high-water marks cannot be summed into a simultaneous peak. Cgroup `memory.max` reports `max`, so these records do not establish enforcement of the requested 8 GiB. A single fixture does not bound remaining seed/stage memory. The final plan conservatively retains 8 GiB per job.

The fixed execution plan has 128 shards with six cases each, at most eight simultaneous single-CPU jobs, and six-hour job limits. Every method has a prespecified outer deadline. The quadratic R native timeout remains an eligible **budget-limited failure**, not scientific N/A. Any such comparative failure withholds all 72 secondary Holm decisions under the unchanged family rule. Complete primary descriptions remain available with their missing/success conditioning disclosed. Initialization does not imply this limitation has been solved.

## Correctness and source scope

The recorded profile-source suite passes **555 tests**, no failures or skips, including all six independent trial targets, exact outcome-access projections, shared GP seeds, atomic terminal records, failure/timeout resume and descendant reaping. A first C08 check failed at 5.72653e-10 against the declared 1e-12 analytic tolerance. Panel-only Hermite 256 evaluation now passes the independent adaptive integral at the same tolerance; native observations and frozen catalog truth remain unchanged. The initial 559-test source adds approved resource evidence, JSON-stable resolved conditions, and separate scale-aware score arithmetic without loosening analytic truth tolerance.

A later review found that a worker may publish its result during termination after an outer deadline. Four regressions reproduced rejection of that late metadata. The current source authenticates and retains it while preserving terminal `timeout`, its original cause and null scores; even a late successful prediction cannot rescue an expired attempt. All four regressions and the full **563-test** suite pass. The source was refrozen before any comparative fit, preserving the initial manifest and the same configuration, targets, recipes, seeds and budgets.

The retained profile source precedes explicit resolved-condition/analysis manifests, review of the known native timeout, separate relative score-reduction tolerance and the deadline-cleanup fix. Package estimator/generator source is unchanged. The original profile directories are verified using their own frozen source, not resumed under the later driver. Raw arrays, predictions, native artifacts, logs and complete runtime inventories stay on GSDS; this packet exports only compact identities and measurements.

The freezes record zero comparative fits at initialization for sample seeds 50000–50063. The [campaign is now running](../native_panel/README.md); its execution and eventual comparative evidence are separate from these preflight checks. No unseen-mechanism or source-paper performance claim follows from preflight. See the [protocol](../../docs/NATIVE_PANEL.md), [configuration](../../configs/native_panel.toml) and [verification scopes](../../docs/VERIFICATION.md).
