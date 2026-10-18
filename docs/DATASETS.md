# Dataset sources and terms

The adapters read files that the user obtains separately. This repository contains no IHDP or ACIC records, downloads no dataset automatically, and grants no rights over external data. Its MIT license covers original benchmark code. Source and license metadata were checked on 2026-10-18; the exact distribution matters.

| Distribution | Primary evidence | Reuse and redistribution status |
| --- | --- | --- |
| Original IHDP study archive, ICPSR 9795 v2, dated 2024-02-14 | [Study page and recommended citation](https://www.icpsr.umich.edu/web/HMCA/studies/9795/versions/V2), [ICPSR terms](https://www.icpsr.umich.edu/sites/icpsr/about/policies/terms-of-use), [redistribution policy](https://www.icpsr.umich.edu/sites/icpsr/about/policies/redistribution) | Public-use access does not require institutional membership. The terms limit use to research/statistical purposes and require written agreement for redistribution, apart from the specified institutional-representative and authorized-collaborator exceptions. The public-use label does not authorize a public GitHub mirror. |
| Semi-synthetic IHDP NPZ files linked by CFRNet, including `ihdp_npci_1-100.{train,test}.npz` and the 1000-replication ZIP releases | [Johansson's Software & Data page](https://www.fredjo.com/), [CFRNet README](https://github.com/clinicalml/cfrnet#examples), [CFRNet software license](https://github.com/clinicalml/cfrnet/blob/master/LICENSE) | The author provides downloads and describes the array fields. No data-specific redistribution grant was found on the inspected download page or README. CFRNet's MIT grant describes its software and associated documentation; it does not establish rights over separately hosted NPZ records. |
| Semi-synthetic IHDP CSV files distributed with CEVAE | [CEVAE repository at the inspected revision](https://github.com/AMLab-Amsterdam/CEVAE/tree/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd) | Public example CSVs are present, but no license file or data-specific redistribution grant was found at that revision. Public availability is not used as permission to relicense or bundle them. |
| NPCI IHDP example data | [Package DESCRIPTION](https://github.com/vdorie/npci/blob/ba08244f35575831d81990c505e70f4fc2e69d80/DESCRIPTION), [example README](https://github.com/vdorie/npci/blob/ba08244f35575831d81990c505e70f4fc2e69d80/examples/ihdp_sim/README), [build exclusions](https://github.com/vdorie/npci/blob/ba08244f35575831d81990c505e70f4fc2e69d80/.Rbuildignore) | The R package declares GPL (>= 2). The IHDP files are in `examples/`, which is excluded from the built package. No separate data-specific grant or underlying-archive permission was found for those example files; the package declaration alone does not resolve that scope. |
| Organizer's `aciccomp2016` R package, version 0.1-0, at `282d26659b2d3d6fd060dde6d32feeb8f1e8ab5a` | [DESCRIPTION](https://github.com/vdorie/aciccomp/blob/282d26659b2d3d6fd060dde6d32feeb8f1e8ab5a/2016/DESCRIPTION), [package README](https://github.com/vdorie/aciccomp/blob/282d26659b2d3d6fd060dde6d32feeb8f1e8ab5a/2016/README.md), [`input_2016` documentation](https://github.com/vdorie/aciccomp/blob/282d26659b2d3d6fd060dde6d32feeb8f1e8ab5a/2016/man/input_2016.Rd) | This is an explicitly GPL (>= 2) distribution containing the input data and simulator. The README identifies `input_2016` as the raw data supplied to contestants; the data documentation identifies the Collaborative Perinatal Project source. No separate data-license exception was found in the inspected package. Obtain this distribution separately and retain its GPL and attribution obligations if redistributing package material. It is not MIT-licensed by this benchmark. |
| Full ACIC2016 archive linked by the official competition site, released 2017-05-22 | [Organizer's competition page](https://jenniferhill7.wixsite.com/acic-2016/competition) | The page explicitly releases covariates, treatment assignments and potential outcomes through a Google Drive link. No redistribution license is stated on the inspected page. The archive was not downloaded for this review, so any terms inside it remain unverified. |
| Other ACIC2016 mirrors | Distribution-specific terms must be checked at the source | The organizer package's GPL declaration is evidence for that package, not a license conclusion about every CSV archive or later mirror. No other ACIC archive is bundled or automatically fetched here. |

The absence of an identified grant is an unresolved permission question, not a claim that every research use is prohibited. Conversely, a synthetic outcome does not itself establish redistribution rights for the underlying empirical covariates. Using a loader does not waive the terms attached to the user's copy.

## Benchmark provenance

The common semi-synthetic IHDP benchmark derives from Hill's [Bayesian Nonparametric Modeling for Causal Inference](https://doi.org/10.1198/jcgs.2010.08162) (2011). The original study archive is Gross's [Infant Health and Development Program (IHDP): Enhancing the Outcomes of Low Birth Weight, Premature Infants in the United States, 1985–1988](https://doi.org/10.3886/ICPSR09795.v2), distributed by ICPSR, version 2 (2024-02-14). These are distinct sources and should be cited according to the data actually used.

ACIC2016 is described by Dorie, Hill, Shalit, Scott and Cervone in [Automated versus Do-It-Yourself Methods for Causal Inference: Lessons Learned from a Data Analysis Competition](https://doi.org/10.1214/18-STS667), *Statistical Science* 34(1):43–68 (2019). The organizer's [package documentation](https://github.com/vdorie/aciccomp/blob/282d26659b2d3d6fd060dde6d32feeb8f1e8ab5a/2016/man/input_2016.Rd) traces its covariates to the Collaborative Perinatal Project and the [National Archives source description](https://www.archives.gov/research/electronic-records/nih.html).

Keep local data outside the repository. Record the exact source URL, release or revision, file hash, applicable terms, replication identifier, covariate mapping and hidden-variable selection alongside a run. Share those compact provenance records and aggregate results only where the source terms permit.

## Local-file adapters

`obs_rct_benchmark.datasets` provides `load_ihdp_npz`, `load_ihdp_csv`, `load_acic2016`, `EmpiricalPopulation`, `EmpiricalDGPConfig` and `generate_empirical_problem`. These loaders validate the source layout and retain covariates and potential-outcome means for the evaluator. They discard the original treatments and noisy outcomes. The generated benchmark draws fresh treatments and outcomes; it is a new OBS/RCT task, not a reproduction of the original IHDP or ACIC experiment.

| Loader | Required layout | Explicit behavior |
| --- | --- | --- |
| `load_ihdp_npz(train_path, test_path=None, replication=0)` | CFR-style keys `x,t,yf,ycf,mu0,mu1`; 25 covariates; a common replication axis | Selects a zero-based replication and pools supplied train/test rows into one empirical population. Original split membership is discarded. |
| `load_ihdp_csv(path)` | Headerless numeric CSV: `t,yf,ycf,mu0,mu1,x1,...,x25` | Retains supplied feature values, including historical `x14` coding; does not apply CEVAE's recoding or standardization. |
| `load_acic2016(covariates_path, response_path, ...)` | A covariate CSV and a response CSV with headers `z,y0,y1,mu0,mu1` | An explicit `column_map` supports dotted R names. Text covariates receive deterministic one-hot encoding; numeric-coded categories can be specified with `categorical_columns`. |

For organizer R exports, use `column_map={"y0": "y.0", "y1": "y.1", "mu0": "mu.0", "mu1": "mu.1"}` and export numeric `z` in `{0,1}`. ACIC rows must already be aligned. `row_id_columns=("x_id", "response_id")` checks identical unique IDs in the same order and excludes the covariate ID column. Without IDs, the loader checks equal row counts and records positional alignment; it cannot establish record linkage. Missing values require explicit preprocessing before loading. The loader accepts arbitrary covariate counts rather than assuming every distribution has the same columns.

The target population gives equal mass to every loaded row. `hidden_columns` names original source columns; hiding a categorical source column removes all its one-hot indicators. Optional `visible_binning` coarsens selected numeric columns by empirical quantiles. The evaluator computes

```text
CATE(displayed X) = E[mu1(row) - mu0(row) | displayed X]
```

by averaging over rows with identical displayed covariates. In real arithmetic, OBS assignment preserves the configured marginal propensity within every group and obeys the requested Γ odds bound. Groups with hidden variation attain Γ; groups without it have realized Γ=1. The float64 implementation verifies both treatment-arm masses and the realized odds bound, and rejects settings whose numerical representation violates these constraints. Conditional outcome means use the represented arm masses. All OBS, RCT training, RCT tuning, RCT test and evaluation roles sample rows independently with replacement. Repeated row identities across roles are possible under this declared finite population. The fixed covariate encoding and coarsening define that population before role sampling and use no outcomes.

With continuous visible features, every row may be unique, so masking another column can leave no hidden variation conditional on displayed `X`. The generator rejects this case unless `allow_no_hidden_ambiguity=True` explicitly requests a labeled negative control. Coarsening changes the observable target and must be declared. Γ controls selection; actual CATE bias still depends on the loaded or configured potential means. Diagnostics report hidden ambiguity, singleton rows, requested/realized Γ and exact confounding bias.

This illustrative IHDP configuration displays a two-bin version of `x1` and hides the other 24 covariates. Choose and report the representation appropriate to the scientific comparison.

```python
from obs_rct_benchmark.datasets import (
    EmpiricalDGPConfig, generate_empirical_problem, load_ihdp_npz,
)

population = load_ihdp_npz("/path/to/train.npz", "/path/to/test.npz", replication=0)
config = EmpiricalDGPConfig(
    name="ihdp_two_bins", dataset_id="ihdp",
    hidden_columns=tuple(f"x{i}" for i in range(2, 26)),
    visible_binning=(("x1", 2),), gamma=2.0,
    n_obs=120, n_rct_train=60, n_rct_tune=30, n_rct_test=30, n_eval=128,
)
problem = generate_empirical_problem(population, config, seed=17)
training = problem.training_view()
```

`TrainingView` exposes displayed covariates, observed treatments/outcomes and known RCT propensities. Full covariates, potential means, OBS oracle probabilities, target CATE and exact biases stay in evaluator-owned objects. Each loader parses and hashes the same captured file bytes. Run metadata records those hashes and transformations, excluding local file paths and raw rows; the runner rejects loaded inputs that differ from its initial snapshot. The [runnable example](../examples/empirical_benchmark.py) supports these local-file formats.

The runner accepts `kind="empirical"`, with `id="ihdp"` or `id="acic2016"`. Set `OBSRCT_IHDP_TRAIN` to the local IHDP NPZ or CSV file and optionally `OBSRCT_IHDP_TEST` for a second NPZ split; a CSV has neither a replication axis nor a separate test file. For ACIC, set `OBSRCT_ACIC_X` and `OBSRCT_ACIC_RESPONSE`. These paths stay in environment variables, while run provenance uses fixed file-role labels and SHA-256 hashes. The runner's ACIC route uses canonical response headers; use the Python loader's `column_map` for dotted R exports. Put hiding, binning, Γ and other `EmpiricalDGPConfig` controls in scenario `parameters`; input files are obtained separately under their own terms.
