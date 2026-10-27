# Inference-policy execution checks

Package **0.3.0a3** separates complete coverage from declared uncertainty permission in generic static and adaptive reports. Complete M11 mean-MSE differences and mixed overall differences remain descriptive; their standard errors and intervals are withheld. Root-PEHE has an independent policy. Explicitly eligible Gaussian controls retain their intervals. [Policy and API](../../docs/INFERENCE.md) · [source/environment and check identities](verification.json).

The recorded GSDS checks pass **90 focused tests and all 428 full-suite tests**, with no skips. CorNet and the R namespace probe were configured; this cohort does not perform native GPy fitting. Hand-calculated fixtures check arithmetic, resolved noise overrides, separate metric declarations, unknown/custom defaults, failure handling and report/manifest behavior. These checks establish enforcement rather than finite moments or confidence-interval coverage.

All 248 static smoke cells pass, with byte-identical records after resume. All 24 adaptive smoke cells pass and replay. Gamma and synthetic-file examples each complete 16 cells. All five tutorials execute 47 code cells and 11 figures. The custom-method tutorial now displays descriptive means when uncertainty is withheld; its previous source and outputs remain in the [original snapshot](https://github.com/snu-causality-lab/obs-rct-benchmark/tree/2fe9c483b2edfc90ee1abeafc1250aba3e1782e5/notebooks).

Independent reaggregation still checks all 1,240 catalog score rows and 488 supported contrasts. The catalog's 21 terminal failures and 472 withheld Holm decisions remain unchanged. These execution examples are not a new performance study. Existing study protocols, results, and the integrated report are preserved.

Run the focused checks on an allocated compute node:

```bash
python -m pytest tests/test_inference_policy.py tests/test_adaptive_inference.py tests/test_evaluation.py tests/test_runner.py tests/test_adaptive_analysis.py
```

The full suite, release checker, independent catalog arithmetic, smoke/replay commands and notebook execution follow [verification](../../docs/VERIFICATION.md). Full logs and failure records remain on GSDS. An expected baseline regression failure, an environment-bootstrap failure and the subsequently fixed notebook-consumer failure are retained separately from successful checks.
