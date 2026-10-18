# Example results

This directory contains compact results from public methods only. The initial examples are execution checks with small samples and two replicate seeds, not a scientific method-ranking study.

Each result set includes its executed TOML configuration, per-record metrics and status, an aggregate JSON summary and a Markdown report. Raw simulated arrays, trained models and machine-specific paths are excluded. The runner records code and environment provenance and rejects mixed-configuration resumes.

| Example | Coverage | Methods | Interpretation |
| --- | ---: | --- | --- |
| [Quick start](quickstart/README.md) | 16 / 16 cells | Four generic ridge baselines | Two designs, two seeds |
| [Full catalog smoke](catalog_smoke/README.md) | 248 / 248 cells | Four generic ridge baselines | All 31 designs, two seeds |
| [Official adapter smoke](official_smoke/README.md) | 12 / 12 cells | RCT ridge, DataFusionPPI, CorNet | Two designs, two seeds; CorNet uses only 5+5 epochs |

Each example completed with no missing or failed cells, and resuming it produced the same records. The numerical results are in the linked Markdown reports and aggregate JSON. Averaging raw errors over designs with different outcome scales is descriptive; do not use these tiny examples as a cross-design leaderboard. See [verification](../docs/VERIFICATION.md) and the repository README for reproduction commands.
