# Reference studies and execution examples

Version `0.2.0a1` adds two fixed reference studies, executed on 2026-10-25. All requested cells completed; original alpha examples remain unchanged below.

| Study | Coverage | Question and readout |
| --- | ---: | --- |
| [Static reference comparison](reference_study/SCIENTIFIC_RESULT.md) | 3,072 / 3,072 | Four matched M2 variants × four OBS/RCT budget pairs × six public recipes × 32 seeds; full official training, nonlinear RCT controls, all 64 primary contrasts with Holm adjustment |
| [Adaptive recruitment comparison](adaptive_reference/SCIENTIFIC_RESULT.md) | 1,152 / 1,152 | Three mechanisms × two outcome budgets × three recruitment policies × two final estimators × 32 seeds; same-estimator policy contrasts, exact acquisition costs and complete deterministic replay |

These studies give sampling replications conditional on development-inspected mechanisms. They do not establish unseen-mechanism generalization or validate identification assumptions in a real population. Pointwise intervals in the adaptive study are descriptive and have no multiplicity adjustment. See the frozen [static](../docs/REFERENCE_STUDY.md) and [adaptive](../docs/ADAPTIVE.md) protocols and [verification](../docs/VERIFICATION.md).

## Original alpha execution examples

Compact results from public methods only, executed with package `0.1.0a1`. These small-sample, two-seed examples check execution; they are not a scientific method-ranking study.

Each result set contains per-record metrics/status, an aggregate JSON summary, a Markdown report and source/environment provenance. Configured runs also include the executed TOML; the file-fixture run is reproduced by its linked Python recipe and fully resolved manifest. Raw arrays, trained models and machine-specific paths are excluded.

| Example | Coverage | Methods | Interpretation |
| --- | ---: | --- | --- |
| [Quick start](quickstart/README.md) | 16 / 16 | Four generic ridge baselines | Two designs, two seeds |
| [Full catalog smoke](catalog_smoke/README.md) | 248 / 248 | Four generic ridge baselines | All 31 fixed designs, two seeds |
| [Official adapter smoke](official_smoke/README.md) | 12 / 12 | RCT ridge, DataFusionPPI, CorNet | Two designs, two seeds; CorNet uses only 5+5 epochs |
| [Gamma factory smoke](gamma_smoke/README.md) | 16 / 16 | Four generic ridge baselines | Constant and logistic Gamma(X), two seeds |
| [Synthetic file-format smoke](empirical_fixture_smoke/README.md) | 16 / 16 | Four generic ridge baselines | Invented IHDP/ACIC-format records; [reproduction recipe](../examples/run_synthetic_file_smoke.py) |

All **308 cells** completed without failures or missing records; resume left records unchanged. The file-format example uses no real IHDP/ACIC data and makes no empirical-dataset performance claim. Numerical results appear in the linked reports and JSON. Averaging raw errors over designs with different outcome scales is descriptive. See [verification](../docs/VERIFICATION.md) and the main README for reproduction and limits.
