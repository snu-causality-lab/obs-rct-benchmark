# Learn the benchmark with notebooks

Start with **01**. The notebooks contain explanations, editable controls, plots and small executable checks. Stored outputs let you read the default walkthroughs without running them. They use public code and synthetic data; no dataset download or optional official upstream is needed.

| Notebook | Question it answers | What you can change or reuse |
| --- | --- | --- |
| [01 · First comparison](01_first_comparison.ipynb) | How do I generate OBS/RCT data and interpret baseline CATE errors? | Sample budgets, confounding, trial coverage; source plots, CATE curves and region errors |
| [02 · Gamma and assumptions](02_gamma_and_assumptions.ipynb) | Why can the same Gamma yield different bias? | Hidden selection, outcome loadings and source shifts; independent Bayes checks |
| [03 · Empirical covariates](03_empirical_covariates.ipynb) | What does hiding a variable change about the target? | Masking/coarsening, displayed-X CATE and optional IHDP/ACIC local-file loading |
| [04 · Custom methods and reproducibility](04_custom_method_and_reproducibility.ipynb) | How do I add a method and preserve a fair, resumable comparison? | Observed-data interface, registration, paired seed blocks, failure/missing accounting |
| [05 · Multi-source identification](05_multisource_identification.ipynb) | Do complementary RCT covariates identify joint CATE? | Exact counterexample enumeration; distinguish identification from estimation |

Notebooks **01–04** teach the implemented package. **05** is a mathematical tutorial, not a multi-source API. For source-attributed paper DGPs and the full fixed catalog, use the [paper cards](../docs/PAPER_DGPS.md), [coverage](../docs/COVERAGE.md) and [catalog configuration](../configs/smoke.toml). These tutorials do not train or tune official CorNet/DataFusionPPI; their separate dependencies and execution recipes are in the main README.

## Run from a cloned checkout

Use Python 3.11 or newer and an environment/kernel with the benchmark installed:

```bash
python -m pip install -e '.[baselines,notebooks]'
python -m jupyterlab
```

Open a notebook in JupyterLab, select that environment's Python kernel, then **Restart Kernel and Run All Cells**. Custom registration in 04 is intentionally unique within a kernel; restart before rerunning the full walkthrough. The small default settings are for learning and execution verification. Substantial sweeps need an appropriate compute server and a declared protocol.

The configuration cells expose the intended controls. Make one change at a time and state the comparison before reading its scores. Truth plots and target effects are explicitly evaluator-only; they are not inputs for fitting, tuning or method selection.

## Headless execution

From the repository root, execute every default tutorial in a fresh kernel:

```bash
python scripts/execute_notebooks.py --output-dir outputs/executed-notebooks
```

The script uses its current Python interpreter for each kernel, runs from a temporary working directory, stops on cell errors, and writes new notebooks rather than overwriting the sources. It clears stale outputs first and requires every code cell to execute and every notebook to produce a figure. The output `execution.json` records code-cell counts, figure counts and hashes of the executed source cells. Cell timeouts and the release file-size limit bound this check. CI runs the same command.

[Stored execution evidence](execution.json) describes the default tutorial run. It does **not** cover the disabled external-data branches in 03. To use those, obtain the dataset separately, read its applicable [distribution terms](../docs/DATASETS.md), set the documented environment paths and explicitly enable the external section. No raw rows, local paths or actual IHDP/ACIC performance outputs are stored here.

## What the results mean

The notebooks illustrate mechanisms with tiny, declared runs. They do not establish a method ranking or real-population validity. Notebook 04's independent replicate count is its seed-block count, not its number of methods, scenarios or evaluation rows. Its malformed-prediction example deliberately produces failure records and keeps them out of scores. Notebook 05 proves equality of entire observable distributions by finite enumeration, rather than fitting competing models and interpreting a tie as nonidentification.

Use the [evaluation protocol](../docs/PROTOCOL.md) and [extension guide](../docs/EXTENDING.md) when moving from these lessons to a substantive comparison or a new contribution.
