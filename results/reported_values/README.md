# Paper-reported values: CorNet and Integrative R

This packet holds the reported side of a planned comparison between published results and benchmark results. It covers the two papers for which the benchmark has both a DGP translation and a method adapter: Hatt et al. (2022), which introduces CorNet, and Wu and Yang (2022), which introduces the Integrative R-learner. Each reported number has its setting, metric definition, replication count and a note on unclear or inconsistent source descriptions. Two replays rerun each paper's released code and the benchmark at the paper's settings.

The extraction and both replays are complete. The combined reported-versus-benchmark view has not been built yet; these files are its inputs.

## Files

| File | Content |
| --- | --- |
| `experiments.csv` | One row per reported experiment: source location, code reference, design, sizes, evaluation, replications, metric, uncertainty, benchmark preset, whether the benchmark can express it, method mapping and notes |
| `reported_values.csv` | One row per reported number, in long format, with panel, method, sizes, metric, value, uncertainty and source type |
| `raw/` | Per-page output of the figure digitizer before labelling |
| `scripts/digitize_page.py.txt` | Recovers means and SDs from vector error-bar figures. It needs PyMuPDF, which is not a benchmark dependency, so it is stored as text; rename it to `.py` to run it on a separately obtained PDF |
| `scripts/build_tables.py` | Labels the digitized points, adds the transcribed tables, checks counts and writes both CSVs |
| [`replay_cornet_fig3/`](replay_cornet_fig3/README.md) | Protocol, scripts, execution record and compact results of the CorNet Figure 3 replay |
| [`replay_intr_tables/`](replay_intr_tables/README.md) | Protocol, scripts, execution record and compact results of the Integrative R Tables 1 and 2 replay |

Rebuild from this folder with:

```sh
python -I scripts/build_tables.py .
```

## Sources

| Source | Version and identity |
| --- | --- |
| Hatt, Berrevoets, Curth, Feuerriegel and van der Schaar (2022), *Combining Observational and Randomized Data for Estimating Heterogeneous Treatment Effects* | [arXiv:2202.12891v1](https://arxiv.org/abs/2202.12891v1), the only version. The PDF read has SHA-256 `ecac5aa6b71ec1839c97d042e03bdceacb883afee7f6d29a2139319c4492ee84` |
| Wu and Yang (2022), *Integrative R-learner of heterogeneous treatment effects combining experimental and observational studies*, CLeaR, PMLR 177:904-926 | [PMLR PDF](https://proceedings.mlr.press/v177/wu22a/wu22a.pdf). The PDF read has SHA-256 `663ec4356f283bb9110d664c0b0dc036a2d72a8942f6dc96627b847991aaca2f` |
| CorNet code | [`tobhatt/CorNet` at `38b9c37`](https://github.com/tobhatt/CorNet/tree/38b9c37d5a86bc87950945d6ddb1ceb5354a45ba). `simulation_studies/simdata.py` matches the hash pinned in [`sources.lock.toml`](../../sources.lock.toml) |
| Integrative R code | [`lwu9/intRlearner` at `a46ce2c`](https://github.com/lwu9/intRlearner/tree/a46ce2c1cc814865898b863d6a8e130bbc2c802b). `R/intRlearner.R` matches the pinned hash |

No paper and no upstream code is copied here; see [the third-party notices](../../THIRD_PARTY_NOTICES.md).

## How the values were extracted

**Tables.** An AI agent transcribed Wu and Yang Tables 1 and 2 and Hatt et al. Table 3 into `scripts/build_tables.py` and checked each against a 200 to 220 dpi rendering of its page. A separate AI agent then transcribed the same tables from renderings up to 400 dpi without seeing these CSVs. All 92 values and uncertainties agree, and the second agent independently flagged the same duplicated print in Table 2.

**Figures.** All CorNet simulation results appear only as figures. Figures 3 to 10 are vector graphics, so their points were recovered from the PDF drawing paths rather than read by eye.
- Grey dashed grid lines and their numeric tick labels give a per-panel affine map from page to data coordinates.
- Each series is a solid stroke colour. Polyline vertices give the means, and the half-length of each vertical error bar gives the SD.
- In every panel, marker centres agree with the polyline means to within 1.4e-6. Error-bar centres agree with the means to within 1.2e-6. Every point has exactly one error bar.
- The digitized x positions match the grids in the authors' scripts to within 3.3e-4.
- Series that the code computes once per replication are constant across x in every panel. These are tau_Unc in Figures 3 to 6 and 8 to 10, and tau_Conf in Figure 7.

Values are stored to four decimals. Digitization error is far below that, so the stored numbers are effectively the authors' plotted floats.

## Coverage

| Experiment | Reported form | Values | Benchmark can express it |
| --- | --- | ---: | --- |
| CorNet Figure 3: OBS size × RCT size | vector figure | 90 | yes |
| CorNet Figure 4: RCT covariate scale | vector figure | 60 | partial |
| CorNet Figure 5: bias complexity | vector figure | 60 | approximate |
| CorNet Figure 6: when CorNet beats RCT-only | vector figure | 90 | partial |
| CorNet Figure 7: when CorNet beats OBS-only | vector figure | 90 | partial |
| CorNet Figure 8: non-shared representation | vector figure | 90 | no |
| CorNet Figure 9: confounding strength | vector figure | 90 | approximate |
| CorNet Figure 10: limited overlap | vector figure | 180 | no |
| CorNet Table 3: STAR, ACTG, NSW | table | 27 | no |
| CorNet real-data Tables 4 to 6 and Figures 11, 13, 14 | tables and figures | 0, listed only | no |
| Integrative R Table 1: Simulation I | table | 27 | yes |
| Integrative R Table 2: Simulation II | table | 38 | yes, with fixed RCT size |
| Integrative R STAR application | text and figure | 3 | no |

## Notes that change how the numbers can be compared

`experiments.csv` holds every note. These are the ones that matter most for a comparison:

- **The CorNet figures are conditional on one unrecorded network.** Each script draws its DGP network once, without a seed, and shares it across all replications. Replications redraw data only.
- **"Unregularized" CorNet still has a penalty.** The text sets lambda_d = lambda_delta = 0, but step 1 applies an L2-norm penalty of 0.1 to the OBS outcome heads. The benchmark adapter uses the same 0.1, but it also standardizes by default, which the authors' scripts do not.
- **Figure 6 cannot come from the released script.** That script runs one replication, which would give zero-length error bars, but the published bars are non-zero.
- **Figure 7 uses a different metric.** Its axis is PEHE, while the other simulation figures use sqrt PEHE.
- **The CorNet plotting scripts need arrays that were not released.** They load the plotted values from `.npy` files that are not in the repository.
- **CorNet Table 3 disagrees with Table 4.** For unregularized CorNet on STAR, Table 3 gives an SD of 0.01 and Table 4 gives 0.10 for the same configuration. The NSW description is also inconsistent: 465 randomized subjects, of whom 297 are treated and 425 are controls.
- **Integrative R's released generator returns a zero CATE for Simulation I.** Its `tau` output is 0 in that branch, so RMSE must use 1 + 2x + 0.75x^2 computed separately. The README also swaps the Simulation I and II labels.
- **Integrative R Simulation II has a larger RCT than stated.** The selection model gives an expected 243.7 units with SD 15.6, while the text says around 200.
- **Integrative R's README cannot produce the paper's J=9 setting.** Its p0 options give J=8 or J=1.
- **Integrative R Table 2 has one duplicated print.** In the J=1, b=2.5 row, the os rlearner standard-error line repeats its value.

## Nearest existing benchmark runs

These benchmark numbers come from the [native panel](../native_panel/SCIENTIFIC_RESULT.md) and the [full catalog](../catalog_study/SCIENTIFIC_RESULT.md), whose settings differ from the papers'. They show where a gap may exist; they are not a reproduction test. All entries are mean sqrt PEHE, averaged per replication or seed in both sources.

| Quantity | Paper | Benchmark |
| --- | --- | --- |
| CorNet on its own DGP | 0.646, SD 0.055, Figure 3 at OBS 500 and RCT 100 | 1.246, MCSE 0.023, native panel at OBS 512 and RCT 128 |
| CorNet, larger budgets | same | 1.192, MCSE 0.024, catalog at OBS 2048 and RCT 512 |
| OBS-only error level on the CorNet DGP | 2.093, Figure 3 tau_Conf | 2.042, catalog OBS-only ridge at OBS 512 |
| RCT-only on the CorNet DGP | 1.009, Figure 3 neural tau_Unc at RCT 100 | 0.721, catalog RCT ridge T-learner at RCT 128 |
| Integrative R, Simulation I | 0.23 and 0.11, Table 1 at OBS 2000 and RCT 200 or 500 | 0.306 and 0.138, native panel quadratic recipe at OBS 512 and RCT 128 or 512 |

The benchmark's OBS-only error matches the paper's bias level, so the CorNet DGP translation reproduces the stated discrepancy. A different bias magnitude therefore does not explain the CorNet gap; the replay below does. The benchmark's Integrative R errors are higher because the native panel has fewer OBS rows and, at RCT 128, fewer RCT rows; the replay below confirms this.

## CorNet Figure 3 replay

The [replay](replay_cornet_fig3/README.md) ran the authors' script over ten seeded network draws. It also ran the benchmark at the Figure 3 settings over ten mechanism seeds, with and without the adapter's standardization. It found three things.

- **The benchmark's default standardization explains most of the native-panel gap.** It raises CorNet's error by about 0.25 in all ten mechanisms. At the native panel's mechanism it moves the error from 0.81 to 1.26, which reproduces the native-panel value.
- **The benchmark's DGP translation and adapter match the authors' code.** Without standardization they agree at all 30 Figure 3 points, within the spread across networks.
- **The authors' code reproduces Figure 3's baselines but not its CorNet curve.** The published CorNet values are better than all ten network draws from OBS 236 upward. At OBS 500 and RCT 100 the paper reports 0.646, while the released code gives 0.748 to 2.057 with a mean of 1.275. A favorable network or an unreleased code difference could explain this; the replay cannot tell which.

## Integrative R Tables 1 and 2 replay

The [replay](replay_intr_tables/README.md) ran the released R code at the settings of Tables 1 and 2 over 100 seeds. It also ran the benchmark's translation and adapter at the same settings and at the native-panel sizes. It found four things.

- **The released code reproduces the paper's R-learner results, except in the J = 9 rows.** It agrees with 20 of the 24 published R-learner values, including all seven integrative R-learner values. Three of the four that differ are in the J = 9 rows of Table 2, where the replay is higher; the fourth is a 0.01 gap in Table 1.
- **The README example reproduces exactly.** Run once in the replay's R environment, it gives the README's printed RMSEs to seven decimals.
- **The benchmark's DGP translation and adapter match the released code.** For the integrative R-learner they agree in all seven paper cells.
- **The native-panel values differ from the paper because of sample size.** At the native-panel sizes the replay reproduces 0.306 and 0.138. At the paper's sizes the benchmark gives 0.234, 0.134 and 0.088, against the paper's 0.23, 0.11 and 0.09.

## Open items

Both replays are complete; their execution records hold job IDs, server paths and results.

1. The planned view shows CorNet both ways: with the authors' raw inputs (`standardize=False`) and with the benchmark default (`standardize=True`). The CorNet replay already holds both at the Figure 3 settings and the two bridge cells, over ten mechanisms. The completed benchmark studies that include CorNet ran only the default; adding the raw-input variant there would be a new run beside them. The `cornet_official` recipe already accepts the option. CorNet-DGP entries should also span several mechanism seeds, because mechanism 2025 is one of the easiest networks for CorNet.
2. Whether the paper's J = 9 rows used the README's p0 = 1, which gives J = 8, is untested. Rerunning the released code's two J = 9 cells with p0 = 1 would test it.
3. The reported-versus-benchmark view for both papers is to be assembled from `reported_values.csv` and the two replays' analysis outputs.
