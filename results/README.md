# Results and execution examples

The [AI-generated technical report](../docs/REPORT.md) gives one integrated reading account. The result packets below retain their own frozen producer, configuration, source/environment identities and uncertainty rules. Package **0.3.0a4** registers 16 recipes, including an admitted RCT-only causal forest, and applies [metric-specific inference permissions](../docs/INFERENCE.md) to new generic runs; this is not a count of 16 distinct fusion algorithms or a claim that every recipe has run on every design.

| Study | Coverage | Question and readout |
| --- | ---: | --- |
| [Full catalog comparison](catalog_study/SCIENTIFIC_RESULT.md) | 79,083 successful / 79,104 applicable; 21 failed, 0 missing; 256 N/A | All 31 native presets × four budgets × ten recipes × 64 seeds; all 472 eligible Holm decisions withheld because four contrasts lack complete pairs |
| [GP source-access experiment](gp_source_pilot/SCIENTIFIC_RESULT.md) | 576 / 576; 0 failed or missing | Three procedures × three OBS discrepancies × two trial budgets × 32 paired seeds. Fixed native GP: adding OBS helps mean risk at discrepancy 0/1 and hurts at 2 relative to its matched RCT-only fit at both budgets. Both contrasts and all six conditions are retained; intervals are pointwise and unadjusted. |
| [Matched-M2 comparison](reference_study/SCIENTIFIC_RESULT.md) | 3,072 / 3,072 | Four matched M2 variants × four OBS/RCT budget pairs × six public recipes × 32 seeds; full official training, nonlinear RCT controls and all 64 primary contrasts with Holm adjustment |
| [Recruitment comparison](adaptive_reference/SCIENTIFIC_RESULT.md) | 1,152 / 1,152 | Three mechanisms × two outcome budgets × three recruitment policies × two final estimators × 32 seeds; same-estimator policy contrasts, exact acquisition costs and complete deterministic replay |

The GP experiment is one additional controlled mechanism in its dedicated driver, not six new canonical presets or a 31-preset comparison of the new recipes. Its OBS start is a prior-shrunken trial-task prediction, not an optimal OBS-only estimator. The [native implementation checks](native_adapter_admission/README.md) separately cover three recipes, 304 tests, direct/repeated calls, independent numerical identities and six bounded CPU profiles; they are not a comparative accuracy study.

These studies provide sampling replications conditional on development-inspected mechanisms. Actual IHDP/ACIC evaluation and unseen-mechanism validation remain open. The catalog, GP, matched-M2 and recruitment analyses keep separate seed collections and inference. Pointwise GP/recruitment intervals are descriptive and have no multiplicity adjustment. See [verification scopes](../docs/VERIFICATION.md) and each linked frozen protocol before reproducing a study; the current package must not resume older result trees with different source fingerprints.

The [0.3.0a3 inference-policy checks](inference_policy_checks/README.md) separately record 428 passing tests, smoke/replay and tutorial execution. They check the shared reporting rules and preserve every scientific study above.

The [0.3.0a4 causal-forest admission](causal_forest_admission/README.md) records direct-native and independent prediction-map checks, two bounded cost profiles and 475 passing tests. It adds a comparator candidate, with no new performance ranking or source-paper table reproduction.

## Original alpha execution examples

Compact results from public methods, executed with package `0.1.0a1`. These small-sample, two-seed examples check execution rather than method ranking. Their result packets preserve per-record metrics/status, aggregate JSON, configuration and source/environment provenance. The file-fixture run uses its linked Python recipe and resolved manifest. Raw arrays, trained models and machine-specific paths are excluded from these compact examples.

| Example | Coverage | Methods | Interpretation |
| --- | ---: | --- | --- |
| [Quick start](quickstart/README.md) | 16 / 16 | Four generic ridge baselines | Two designs, two seeds |
| [Full catalog smoke](catalog_smoke/README.md) | 248 / 248 | Four generic ridge baselines | All 31 fixed designs, two seeds |
| [Official adapter smoke](official_smoke/README.md) | 12 / 12 | RCT ridge, DataFusionPPI, CorNet | Two designs, two seeds; CorNet uses only 5+5 epochs |
| [Gamma factory smoke](gamma_smoke/README.md) | 16 / 16 | Four generic ridge baselines | Constant and logistic Gamma(X), two seeds |
| [Synthetic file-format smoke](empirical_fixture_smoke/README.md) | 16 / 16 | Four generic ridge baselines | Invented IHDP/ACIC-format records; [reproduction recipe](../examples/run_synthetic_file_smoke.py) |

All **308 cells** completed without failures or missing records; completed-run resume preserved the records. The bundles overlap and are not 308 independent scientific tasks. File fixtures use no real IHDP/ACIC data. Their numerical results and original manifests remain unchanged.
