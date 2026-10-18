# Example results

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
