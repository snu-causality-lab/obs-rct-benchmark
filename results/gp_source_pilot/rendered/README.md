# Three-source GP development pilot: figures and scalar tables

Complete terminal coverage: 576 successful, 0 failed, zero missing fits.

The two figures show condition-specific trial-target CATE risk and paired full-minus-control differences. Negative differences favor the full OBS+RCT recipe. The 32-seed pilot is a development diagnostic, not unseen-mechanism evaluation or a source-paper table reproduction.

OBS-conditioned start predicts the trial task after conditioning only on observational outcomes. Under the fixed zero-mean two-task GP, its cross-task conditional mean is shrunk by rho/(1+kappa), with rho=0.5 and kappa=1e-9. It is not the unshrunk OBS task contrast.

MSE bars are one Monte Carlo standard error only for complete success groups. Paired bars are the analysis's approximate pointwise 95% t intervals over sample-seed blocks; incomplete intervals are withheld, and conditional means are marked. No p-values or multiplicity decisions are introduced. Separate paired-figure rows use separate vertical scales.

CSV floats retain 17 significant digits; blanks indicate unavailable values, and rounding appears only in figure/Markdown display. Label columns give the frozen procedure's allowed source outcomes; successful records are checked against actual supplied/consumed counts. Full uses B RCT and 512 OBS outcomes, RCT-only uses B and 0, OBS-conditioned start uses 0 and 512. The full procedure's two tuning-role rows alias charged fitting rows and supply no extra outcomes.

Fit-time summaries condition on successful fits; launcher-time means include all attempts and exclude queue delay. Worker self and child RSS columns are separate high-water marks, not simultaneous process-group or whole-job peaks. Concurrent jobs were not a controlled runtime-ratio study.

The renderer authenticates the saved terminal-record collection and its scalar agreement with the analysis; it does not refit models or independently reconstruct target/prediction arithmetic. Raw fitting arrays, query truth, predictions and full logs remain server evidence. figure_source.json binds the input hashes and exported files.

| Delta | RCT rows | Full MSE | RCT-only MSE | OBS start MSE | Full−RCT [pointwise 95% CI] | Full−OBS start [pointwise 95% CI] |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 128 | 0.143201 | 0.212625 | 0.3409 | -0.0694242 [-0.0971763, -0.041672] | -0.197699 [-0.229091, -0.166308] |
| 0 | 512 | 0.0647022 | 0.0712551 | 0.3409 | -0.00655291 [-0.00958201, -0.00352382] | -0.276198 [-0.290495, -0.261901] |
| 1 | 128 | 0.164762 | 0.212625 | 0.361698 | -0.047863 [-0.0784679, -0.0172582] | -0.196936 [-0.23115, -0.162723] |
| 1 | 512 | 0.0638492 | 0.0712551 | 0.361698 | -0.00740584 [-0.0115086, -0.00330305] | -0.297849 [-0.31583, -0.279868] |
| 2 | 128 | 0.252619 | 0.212625 | 0.511709 | 0.0399937 [0.0111285, 0.0688589] | -0.259091 [-0.29848, -0.219701] |
| 2 | 512 | 0.0885038 | 0.0712551 | 0.511709 | 0.0172488 [0.00888523, 0.0256123] | -0.423205 [-0.446028, -0.400383] |
