# Results and execution examples

The [AI-generated technical report](../docs/REPORT.md) connects five studies. Each packet below is the authoritative record of its frozen producer, configuration, source/environment identities, coverage and inference.

| Study | Coverage | Question and readout |
| --- | ---: | --- |
| [Native/default trial-CATE panel](native_panel/SCIENTIFIC_RESULT.md) | 8,350 successes / 8,448 requested; 64 failed, 31 timed out, three missing | Six inspected mechanisms × two trial budgets × eleven recipes × 64 seeds. Matched GP OBS access and complete-procedure comparisons with causal forest; all 72 joint decisions and simultaneous precision withheld. |
| [Full catalog comparison](catalog_study/SCIENTIFIC_RESULT.md) | 79,083 successful / 79,104 applicable; 21 failed, zero missing; 256 N/A | All 31 native presets × four budgets × ten recipes × 64 seeds; all 472 eligible Holm decisions withheld because four contrasts lack complete pairs |
| [Controlled GP experiment](gp_source_pilot/SCIENTIFIC_RESULT.md) | 576 / 576; zero failed or missing | Three procedures × three OBS discrepancies × two trial budgets × 32 paired seeds. Fixed native GP: OBS helps descriptive mean risk at discrepancy 0/1 and hurts at 2 versus matched RCT-only at both budgets. All six conditions and both contrasts are retained; intervals are pointwise and unadjusted. |
| [Matched-M2 comparison](reference_study/SCIENTIFIC_RESULT.md) | 3,072 / 3,072 | Four matched M2 variants × four OBS/RCT budget pairs × six recipes × 32 seeds; full official training, nonlinear RCT controls and all 64 primary contrasts with Holm adjustment |
| [Recruitment comparison](adaptive_reference/SCIENTIFIC_RESULT.md) | 1,152 / 1,152 | Three mechanisms × two outcome budgets × three recruitment policies × two final estimators × 32 seeds; same-estimator policy contrasts, charged acquisition and complete deterministic replay |

These are sampling replications within inspected synthetic mechanisms, not empirical-data or unseen-mechanism validation. Study-specific seed collections, targets, budgets and inference remain separate. The controlled GP's six conditions belong to one construction, not six canonical presets. Its OBS start is a prior-shrunken trial prediction, not an optimal OBS-only estimator. Native C08 uses a trial target different from canonical catalog C08.

Package **0.3.0a4** registers 16 recipes, including variants and controls; not every recipe runs in every study. [Native implementation admission](native_adapter_admission/README.md), [causal-forest admission](causal_forest_admission/README.md), [inference-policy checks](inference_policy_checks/README.md) and [panel preflight](native_panel_preflight/README.md) provide implementation/resource evidence rather than accuracy rankings. The [verification guide](../docs/VERIFICATION.md) records their exact scopes.

Reproduce each study with its frozen source and environment. Source fingerprints prevent resuming a retained tree under different code. The native panel's three unresolved attempts stay missing; its [packet](native_panel/README.md) exposes all 132 method rows, 72 comparisons and requested per-seed identities without retrying or replacing outcomes.

## Original alpha execution examples

These package `0.1.0a1` examples check small-sample execution. Each packet retains metrics/status, aggregate JSON and configuration/source/environment provenance. Raw arrays, trained models and machine-specific paths remain outside these compact examples.

| Example | Coverage | Methods | Interpretation |
| --- | ---: | --- | --- |
| [Quick start](quickstart/README.md) | 16 / 16 | Four generic ridge baselines | Two designs, two seeds |
| [Full catalog smoke](catalog_smoke/README.md) | 248 / 248 | Four generic ridge baselines | All 31 fixed designs, two seeds |
| [Official adapter smoke](official_smoke/README.md) | 12 / 12 | RCT ridge, DataFusionPPI, CorNet | Two designs, two seeds; CorNet uses 5+5 epochs |
| [Gamma factory smoke](gamma_smoke/README.md) | 16 / 16 | Four generic ridge baselines | Constant and logistic Gamma(X), two seeds |
| [Synthetic file-format smoke](empirical_fixture_smoke/README.md) | 16 / 16 | Four generic ridge baselines | Invented IHDP/ACIC-format records; [reproduction recipe](../examples/run_synthetic_file_smoke.py) |

All **308 cells** completed without failures or missing records, and completed-run resume preserved the records. The bundles overlap; they are not 308 independent scientific tasks. File fixtures use no real IHDP/ACIC data. Original numerical results and manifests remain unchanged.
