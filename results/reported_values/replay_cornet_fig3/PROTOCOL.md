# CorNet Figure 3 replay protocol

Fixed on 2026-11-08 before any replay output was produced. Preflight outputs check execution only and do not change this protocol.

## Question

On the CorNet DGP, the benchmark's CorNet recipe has mean sqrt PEHE 1.246 in the native panel at OBS 512 and RCT 128. Figure 3 of Hatt et al. (2022) reports 0.646 at OBS 500 and RCT 100. This replay asks which of three explanations hold:

1. The published figure does not reproduce with the released code.
2. The benchmark's DGP translation or adapter differs from the released code at matched settings.
3. The benchmark adapter's default input standardization raises the error.

## Arms

**A. Released code.** Execute lines 3 to 247 of `simulation_studies/sim_n_conf.py`, SHA-256 `9d99f06ae92650114dc8ad314a93446703491b4d4502d4af8922eba993e581c3`, from `tobhatt/CorNet` at `38b9c37d5a86bc87950945d6ddb1ceb5354a45ba`. This is everything except the IPython reset in lines 1 and 2 and the plotting and file-loading tail from line 248. The deviations are listed below.
- NumPy, PyTorch and Python seeds are set to s for s = 0, ..., 9, one job per seed. Each job therefore draws one network, as one run of the script does.
- PyTorch uses one thread, and matplotlib uses the Agg backend.
- Placeholder modules stand in for `pandas` and `seaborn`. The executed lines import both but never use them. Every library the script imports is loaded before the placeholders become importable, so scikit-learn sees an environment without pandas, as installed. This loading order was added after the first preflight, before any replay output existed.

Each job runs the script's own 10 replications, three RCT sizes and ten OBS sizes, and keeps every per-replication MSE.

**B. Benchmark.** Use the public package at `173af2a` with the same CorNet checkout.
- Build `make_paper_config(paper_id="cornet_official_n_conf", mechanism_seed=ms, n_obs=n, n_rct_train=m, n_rct_tune=4, n_rct_test=4, n_eval=10000)`, then `generate_paper_problem(cfg, seed=s)`. Four is the generator's minimum role size, and CorNet uses neither role.
- Fit `OfficialCorNet(standardize=std).fit_predict(view, eval_X, seed=s)`, with std = False and True on the same problem.
- Mechanism seeds are ms = 2025, ..., 2034. 2025 is the default used in earlier studies.
- Sample seeds are s = 0, ..., 9.
- The grid is Figure 3's ten OBS sizes, 25, 77, 130, 183, 236, 288, 341, 394, 447 and 500, crossed with RCT sizes 25, 50 and 100.
- Two bridge cells repeat the native-panel sizes: OBS 512 with RCT 128, and OBS 512 with RCT 512.

## Metric and summaries

The metric is sqrt PEHE per replication. Within one network it is averaged over replications: in Arm A over the script's 10 replications per seed, in Arm B over the 10 sample seeds per mechanism seed. Across the 10 networks of each arm, the summaries are mean, SD, minimum and maximum.

## Readouts

All readouts are descriptive; no tests.

- **R1, reproduction.** At each of the 30 Figure 3 points for tau_CorNet, check whether the paper's value lies within the minimum to maximum of the 10 Arm A network means. Report the count and the point OBS 500 with RCT 100. Repeat for tau_Conf and tau_Unc.
- **R2, translation and adapter.** At the same 30 points, compare Arm B without standardization against Arm A. Report the difference of the across-network means and count the points where it is smaller than the larger of the two across-network SDs.
- **R3, standardization.** Compute the paired difference, standardized minus unstandardized, for each mechanism seed, sample seed and cell, then average within mechanism. At the bridge cells and at OBS 500 with RCT 25, 50 and 100, report three numbers: the mean over mechanisms, the across-mechanism SD, and how many mechanisms show a positive difference.
- **Bridge.** Report Arm B with standardization at OBS 512 with RCT 128 and with RCT 512, next to the native-panel values 1.2463 and 1.2158.

Failures are recorded and never retried. No setting changes after outputs are seen. Any further run is a new, separately labeled step.

## Resources

Every job runs on GSDS `P2` with no GPU, one CPU and 4 GiB of memory. Time limits are one hour for Arm A tasks, two hours for Arm B tasks, and 15 minutes each for the preflight and the aggregation. All paths are under `/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02`. The `-r01` folder keeps only the first preflight, which failed on the placeholder loading order, and the scripts it ran.
