# Full-catalog static reference study

This protocol and its [nominal configuration](../configs/catalog_study.toml) are fixed before fitting or inspecting the new evaluation-seed comparisons. It extends coverage to every canonical catalog preset with the existing public recipes. It introduces no new estimator and does not tune recipes using evaluation truth. The earlier [M2 study](REFERENCE_STUDY.md) and [two-stage recruitment study](ADAPTIVE.md) remain separate evidence.

## Question and scope

How do the fixed public estimators behave across the declared differences in confounding, source outcomes, trial coverage, transport, effect geometry and sample budget? The catalog comprises 12 structural presets M0–M11, 10 paired variants C01–C10, and nine source-related variants associated with six research papers. Related variants are not independent papers or independently discovered mechanisms. See [coverage](COVERAGE.md) and [paper cards](PAPER_DGPS.md).

Every preset is used unchanged apart from its supported sample-role sizes. Structural/addendum configurations explicitly set the six roles below; paper configurations support the first five and have no target-reference role, so no unsupported `n_target_reference` field is passed to them. Preset dimensions, mechanism seeds, covariate laws, outcome/assignment equations, noise, trial participation and target populations are preserved. In particular, the CorNet translation remains 25-dimensional, the three KPS-related designs remain one-dimensional, multivariate Integrative R remains nine-dimensional, Elastic and power-likelihood designs remain two-dimensional, CVCI remains five-dimensional, and C09/C10 remain six-dimensional. No blanket `d=5` override is applied.

The same sample-seed grid, **20000–20063**, gives 64 independent seed blocks within each condition. Mechanism seeds stay at their canonical values; this is sampling replication conditional on development-inspected designs, not unseen-mechanism generalization. Methods share the realized OBS/RCT/evaluation samples within a condition/seed. Seeds pair settings, but differently sized samples are not asserted to be nested. Cross-condition dependence is retained; conditions and evaluation rows are not counted as independent replications.

## Fixed sample and optimization budgets

| Role | Rows | Information and accounting |
| --- | ---: | --- |
| OBS training | 512 or 2,048 | Available to observational/fusion recipes |
| RCT training | 128 or 512 | Available to trial/fusion recipes |
| RCT tuning | 32 | Available but unused by these fixed recipes |
| RCT factual test | 64 | Evaluator-only, unused by fitting and the CATE score |
| Target evaluation | 2,048 | Prediction queries; causal truth stays evaluator-owned |
| Target reference | 0 / absent | Explicitly zero for structural/addendum; absent in paper designs. No adaptation on target-reference or evaluation covariates |

The two OBS sizes and two fitting sizes make four conditions per preset, **124 conditions** in total. Scenario labels are `o512_r128`, `o512_r512`, `o2048_r128` and `o2048_r512`; the preset ID remains unchanged. Available RCT-label budgets are 160/544, including the unused tuning labels; actual training uses 128/512. This is a static study, not a recruitment policy, and its budgets must not be equated to the separate sequential track's charged acquisition budgets.

Ten fixed recipes are requested nominally:

- RCT-only, OBS-only and naive pooled T-learners, each with standardized Ridge or ExtraTrees: six recipes.
- RCT-only two-fold DR-learners with Ridge or ExtraTrees: two recipes. The nuisance folds are stratified by treatment and use known trial assignment probabilities.
- Official CorNet with 500 source epochs, 200 correction epochs, width 25, three source representation layers, learning rate .1, source outcome penalty .1, correction penalty zero, and OBS-fitted standardization. There is no early stopping or search.
- Official DataFusionPPI with its DR option and covariate-shift handling enabled. Upstream rotated nuisance splits use RCT training only.

Ridge uses penalty one and an intercept with armwise/training-fitted standardization. ExtraTrees uses 100 trees, minimum leaf size five, all features and one worker. The nominal configuration explicitly sets penalties, tree/leaf counts, CorNet epochs/width/learning rate/penalties/standardization, and DataFusionPPI learner/shift options. Three CorNet representation layers, two DR nuisance folds, ExtraTrees all-feature selection and one worker, and Ridge intercept/standardization behavior are fixed implementation invariants rather than additional TOML options. The adapters verify upstream revisions and actual source hashes. These are disclosed fixed recipes, not optimized reproductions of source-paper tables. Equal data access does not imply equal statistical modeling flexibility or optimization effectiveness.

## Applicability is part of the design

The canonical `integrative_r_learner_multivariate_paper` design has covariate-dependent known trial treatment probabilities. The locked official DataFusionPPI adapter accepts only a constant probability. Its four condition rows are therefore **not applicable** for DataFusionPPI at all 64 seeds: **256 nominal cells**. No treatment probability or source equation is changed to make the method run.

The nominal product is `31 × 4 × 64 × 10 = 79,360` cells. Removing the 256 predeclared unsupported cells gives **79,104 requested fits**. Unsupported cells are neither successes nor failures, and are not scored as zero. Any additional unexpected simulation or fitting error is a terminal failed cell, retained with its stage/type and coverage. It does not become a post-hoc applicability exemption.

The [sharding helper](../scripts/make_catalog_configs.py) writes ordinary runner-compatible configurations with two disjoint condition groups per sample seed:

| Group | Conditions | Recipes | Cells per seed |
| --- | ---: | ---: | ---: |
| `common` | 120 | 10 | 1,200 |
| `variable_propensity` | 4 | 9 | 36 |

The groups intentionally share the same sample-seed labels while their requested condition/method rectangles are disjoint. Sixty-four seeds produce 128 shard configurations. `study_manifest.json` records their exact file/configuration hashes, the applicability exclusions and full requested-cell counts. The nominal configuration is a design declaration; execute the generated compatible shards rather than passing the nominal product directly to the runner.

Some settings are deliberate identification or transport stress tests. The KPS normal-target extension evaluates beyond trial support; C08 has hidden participation that changes trial versus target CATE. A method's target error there is not evidence that trial randomization alone identifies that target, or that a required transport assumption holds. These settings remain visibly separate from ordinary overlap/transport conditions.

## Scores, precision and the prespecified primary comparisons

The primary score is per-replicate target CATE MSE, with root-PEHE named separately. Report all condition/method means and coverage. Different outcome scales and different target laws prevent interpreting one unweighted raw-MSE average as a universal ranking. The paired C comparisons are useful diagnostics at their canonical settings; arbitrary M-to-M differences need not isolate a single cause.

Generic runner shard summaries and READMEs are execution diagnostics; the catalog-specific analyzer is authoritative for applicability and M11 inference, and public exports must omit generic pooled raw-MSE rankings and invalid M11 mean intervals.

For each applicable condition, compare both official fusion recipes with both nonlinear RCT-only controls, `rct_t_extra_trees` and `rct_dr_extra_trees`. Differences are fusion minus reference, so negative favors fusion. Other generic baselines remain fully reported descriptive references; selecting the best observed RCT control does not create a new primary test.

There are 496 nominal fusion/control/condition contrasts. Eight DataFusionPPI contrasts are inapplicable. Before fitting, 16 additional M11 contrasts are excluded from mean-t inference for the heavy-tail reason below. The resulting **472 prespecified two-sided paired mean-t tests** receive one joint Holm adjustment at .05. Pointwise approximate 95% Student-t intervals and Monte Carlo standard errors across the 64 seed blocks remain pointwise, not simultaneous. A favorable Holm decision means a negative paired mean and adjusted p ≤ .05; unresolved does not imply equivalence.

If any of the 472 required primary comparisons lacks a successful pair at any requested seed, family-level Holm decisions are withheld rather than silently testing a smaller family. Descriptive means of successes retain their failed/missing counts. No failed case is assigned a winning score. Report achieved Monte Carlo precision; do not extend selected conditions after inspecting their means, p-values or uncertainty.

### M11 remains a descriptive heavy-tail stress test

M11's canonical noise is Student t with three degrees of freedom, scaled to variance one. Its second moment exists but its fourth moment does not: the [Student-t density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) at three degrees of freedom decays as `|x|^-4`, so integrating its fourth-moment tail diverges. Squared-loss variance can therefore be infinite; 64 seeds alone do not justify a finite-variance mean-MCSE or mean-t interval. Cancellation in some paired comparisons is not assumed to rescue every method pair.

All M11 fits, raw per-seed errors and descriptive means remain in the study. Report medians and interquartile ranges as additional empirical descriptions, while **withholding M11 mean MCSE, t intervals and mean-t p-values**. Do not encode the omitted tests as p=1 or count them as unresolved Holm discoveries. No median/sign-test superiority claim is introduced. Its Gaussian-noise twin C03 remains eligible for the declared mean-t family; the two noise settings are not silently merged or substituted.

## Reproducibility and independent checks

Before submitting, freeze a complete code/configuration/protocol snapshot and bind its file hashes. Use a fixed numerical environment, pinned official adapter revisions and verified imports; record Python, NumPy, SciPy, scikit-learn and PyTorch versions. Keep method seeds separate from generator seeds, and use a single numerical-library thread per worker. Parallelism is across disjoint immutable shard configurations. Code, environments, complete outputs and operational logs stay on the execution server; compact scientific records and selected figures may be published.

The first fixed seed may be used for resource and execution preflight and then retained as part of the declared study. It is not an accuracy-tuning pilot. The frozen [target checker](../scripts/verify_catalog_targets.py) checks all 31 canonical designs at the smaller budget against separately written analytic target formulas with absolute tolerance 1e-12. Its CorNet manual forward pass shares the initialized mechanism weights, so it checks the target calculation rather than independently validating mechanism initialization or the estimator. Generator/target consistency checks are distinct from fitted-method performance, and a design inspected during development is not an external holdout merely because it uses a new seed.

Each requested applicable `(condition, seed, method)` has one terminal record. Resume must require identical configuration, scientific source, numerical environment and upstream source identity; failed records remain terminal. Check completed-run resume byte-for-byte and retain any unsuccessful attempts. A partial final JSONL line, duplicate cell, overlapping requested shard rectangle or changed fingerprint must fail closed.

Separately check every applicable method for all 31 designs at the first seed and OBS512/RCT128: **309 cases, each with two fresh refits, or 618 fit invocations**. Regenerate each dataset twice and compare its observed roles, evaluation rows and target arrays. For each successful refit, recompute MSE with a direct dot product, compare against its saved score at a fixed 1e-12 absolute/relative tolerance, and bind checker source/input hashes. For terminal failed cases, retain and compare the reproduced failure stage/type instead of omitting them or inventing a score. This deterministic reconstruction tests execution and score reproducibility on the declared subset. It is not an independent empirical replication, an independently implemented estimator oracle, or full prediction replay of all 79,104 fits. If the tolerance fails, preserve the evidence and investigate; do not loosen it after reading discrepancies.

Before aggregation, authenticate the exact union of all 79,104 requested applicable cells and the 256 declared N/A cells, every shard configuration/resolved preset, source/environment/optional-method fingerprints, record/method seeds and metric identities. Matching the collection's own summary JSON is insufficient by itself. Independently compute aggregate arithmetic from terminal records and protect that analysis with hand-calculated small examples, explicit heavy-tail exclusions and corrupted-input regressions.

## Execute and reproduce

Install the public package and documented optional official checkouts in a controlled environment. Analysis uses `.[analysis]`. On the chosen compute service, request CPU/RAM/time explicitly, limit worker/thread concurrency and use immutable paths for all jobs. Exact job IDs, allocations, observed runtime/memory and any repairs belong with the execution record rather than being inferred from a successful dry run.

```sh
python scripts/make_catalog_configs.py --config configs/catalog_study.toml --output outputs/catalog-configs
obs-rct-benchmark run --config outputs/catalog-configs/group-common-seed-20000.toml --output outputs/catalog/group-common-seed-20000
obs-rct-benchmark run --config outputs/catalog-configs/group-variable_propensity-seed-20000.toml --output outputs/catalog/group-variable_propensity-seed-20000
```

Execute each of the 128 generated configurations once; add `--resume` to the identical configuration/output for a completed-run check. The helper's `study_manifest.json` gives the deterministic order and complete list. Frozen analysis/checking tools must use the same canonical configuration and applicability/heavy-tail rules. Publish every condition's result and uncertainty limits, including failures and adverse outcomes. The study does not include new fusion methods, actual IHDP/ACIC validation, a multi-source contract or an extension of recruitment to the nine paper-related generators.


After executing all 128 configurations, authenticate and summarize the complete grid:

```sh
python scripts/analyze_catalog_study.py --config configs/catalog_study.toml --runs outputs/catalog/group-* --output outputs/catalog-analysis --paper-assets
python scripts/verify_catalog_targets.py --config configs/catalog_study.toml --output outputs/catalog-target-check.json
python scripts/verify_catalog_reproducibility.py --config configs/catalog_study.toml --runs outputs/catalog/group-common-seed-20000 outputs/catalog/group-variable_propensity-seed-20000 --output outputs/catalog-reproducibility.json
python scripts/verify_catalog_aggregates.py --config configs/catalog_study.toml --analysis outputs/catalog-analysis --output outputs/catalog-arithmetic-check.json
```

An analysis with terminal failed cells writes authenticated reports and exits nonzero. Preserve that status and inspect coverage; do not call it a fully successful fit campaign or retry failed cells to obtain scores. A failure-free pipeline dependency must not silently prevent reporting retained scientific failures. The independent arithmetic check can pass for a complete, honestly reported grid containing terminal failures.

[Completed execution and results](../results/catalog_study/SCIENTIFIC_RESULT.md) report the realized coverage, diagnostics and reconstruction limits. To check its compact public export, use `verify_catalog_aggregates.py` with `--delivery-manifest results/catalog_study/DELIVERY_MANIFEST.json`; only explicitly hash-bound manuscript PDFs may be omitted. Full per-seed compact CSVs remain mandatory. The selected original first-seed manifests/records are in `results/catalog_study/evidence/runs/`; the frozen full-run protocol can regenerate every input.
