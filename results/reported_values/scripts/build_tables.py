"""Build experiments.csv and reported_values.csv from digitized figures and transcribed tables.

Usage: python -I scripts/build_tables.py REPORTED_VALUES_DIR

Inputs: REPORTED_VALUES_DIR/raw/hatt2022_page*_digitized.csv, written by scripts/digitize_page.py
from the arXiv v1 PDF of Hatt et al. (2022). Table values below were transcribed by hand and
checked against high-resolution renderings of the source pages.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])

CORNET_CODE = "tobhatt/CorNet@38b9c37d5a86bc87950945d6ddb1ceb5354a45ba"
INTR_CODE = "lwu9/intRlearner@a46ce2c1cc814865898b863d6a8e130bbc2c802b"
CORNET_SRC = "Hatt et al. (2022), arXiv:2202.12891v1 (only version)"
INTR_SRC = "Wu and Yang (2022), CLeaR, PMLR 177:904-926"

FIG_COMMON = (
    "DGP network fixed within a script run and shared by all replications; drawn without a seed "
    "(see notes). Training per script: 3-layer width-25 representation, 500 OBS epochs, "
    "200 correction epochs, SGD lr argument 0.1, full batch, head-norm penalty 0.1, correction penalty 0."
)

EXPERIMENTS = [
    dict(
        experiment_id="hatt2022_fig3",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Figure 3, PDF p.29; Section 7.1.1(i)",
        result_form="figure (vector)",
        code_reference=f"{CORNET_CODE} simulation_studies/sim_n_conf.py, simdata.py:sim_data_n_conf",
        design=(
            "X~N(0,I) in both sources; T~Bernoulli(0.5) in both; outcome heads on a shared random tanh "
            "representation; RCT/target heads add a sparse correction delta that OBS lacks; no hidden "
            "treatment confounding. " + FIG_COMMON
        ),
        d="25",
        n_obs="int(linspace(25,500,10)) = 25,77,130,183,236,288,341,394,447,500",
        n_rct="25 | 50 | 100 (panels; first m rows of a 1,000-row RCT draw)",
        evaluation="10,000 fresh X~N(0,I) per replication; target = RCT CATE",
        replications="10 (caption and code)",
        metric="sqrt PEHE: per replication sqrt(mean_test (tau_hat - tau)^2), then mean over replications",
        uncertainty="SD over replications (numpy std, ddof=0)",
        methods="tau_CorNet; tau_Conf (OBS heads of the same CorNet fit); tau_Unc (same architecture fit on RCT only)",
        benchmark_preset="paper:cornet_official_n_conf",
        benchmark_expressible="yes",
        benchmark_gap_or_overrides=(
            "Set n_obs, n_rct_train and n_eval to the paper grid. Preset parameters equal the script "
            "(d=25, outcome norm 10, correction RMS 2, sparsity 0.666, noise SD 0.5). Network differs: "
            "benchmark mechanism_seed=2025 vs one unrecorded draw; run several mechanism seeds to show "
            "between-network spread."
        ),
        benchmark_method_mapping=(
            "tau_CorNet -> cornet_official twice: standardize=False (the authors' raw inputs) and the default standardize=True (benchmark adaptation); "
            "tau_Conf, tau_Unc -> no matching recipe"
        ),
        extraction="digitized from PDF vector paths",
        notes=(
            "1) Paper text omits d, the representation architecture, noise SD and beta; all come from the script. "
            "2) The DGP network is created once before the replication loop without a seed, so replications "
            "redraw data only and the curve is conditional on one unrecorded network. "
            "3) The text calls this the unregularized variant (lambda_d = lambda_delta = 0), but step 1 still "
            "applies an L2-norm penalty of 0.1 to the OBS outcome heads (lambd_h_c), which the text does not mention. "
            "4) The scheduler steps at epoch 0, so the effective learning rate is 0.1 for one epoch, 0.01 "
            "to epoch 250 and 0.001 after. "
            "5) Plotted arrays are loaded from .npy files absent from the repository; the save block is commented out. "
            "6) sim_data_n_conf ignores its noise_scale and cov_shift arguments: noise SD is fixed at 0.5 and "
            "both sources use N(0,I). "
            "7) tau_Unc is fit once per replication and repeated across nConf; its digitized values are constant "
            "within each panel, which matches the code. "
            "8) The OBS-only error plateaus near 2.07-2.09, consistent with the script's root mean squared "
            "OBS-RCT CATE gap of 2. "
            "9) A replay of this script over ten seeded network draws (replay_cornet_fig3) reproduces tau_Conf and tau_Unc "
            "at all 30 points but not tau_CorNet: from OBS 236 upward the published values are below the best of the ten draws "
            "(OBS 500, RCT 100: paper 0.646; released code 0.748 to 2.057, mean 1.275)."
        ),
    ),
    dict(
        experiment_id="hatt2022_fig4",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Figure 4, PDF p.31; Section 7.1.1(ii)",
        result_form="figure (vector)",
        code_reference=f"{CORNET_CODE} simulation_studies/sim_d_infty.py, simdata.py:sim_data_d, sim_data_d3",
        design="As Figure 3 with nUnc=50; RCT covariates N(0, sigma_u^2 I) with sigma_u=1 (left) or 0.1 (right). " + FIG_COMMON,
        d="25",
        n_obs="int(linspace(25,500,10))",
        n_rct="50",
        evaluation="10,000 rows per replication (generator constant)",
        replications="10 (caption and code)",
        metric="sqrt PEHE, mean over replications",
        uncertainty="SD over replications (ddof=0)",
        methods="tau_CorNet; tau_Conf; tau_Unc",
        benchmark_preset="paper:cornet_official_n_conf",
        benchmark_expressible="partial",
        benchmark_gap_or_overrides="sigma_u=1 panel: as Figure 3 at n_rct=50. sigma_u=0.1 panel: the preset has no RCT covariate-scale option.",
        benchmark_method_mapping="tau_CorNet -> cornet_official twice: standardize=False and the default standardize=True",
        extraction="digitized from PDF vector paths",
        notes=(
            "1) The sigma_u=1 panel repeats the Figure 3 design at nUnc=50 with a separate network draw and run. "
            "2) The internals of the shifted-covariate generators were not re-derived here; settings follow the paper text and script header."
        ),
    ),
    dict(
        experiment_id="hatt2022_fig5",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Figure 5, PDF p.31; Section 7.1.1(iii)",
        result_form="figure (vector)",
        code_reference=f"{CORNET_CODE} simulation_studies/sim_bias.py, simdata.py:sim_data_bias",
        design=(
            "As Figure 3 with nUnc=50; delta rescaled so the root mean squared OBS-RCT CATE gap is 1. "
            "'C_B small' uses OBS outcomes from the base heads (gap delta); 'C_B large' uses a second OBS "
            "sample whose outcomes subtract delta (gap 2*delta). " + FIG_COMMON
        ),
        d="25",
        n_obs="int(linspace(25,500,10))",
        n_rct="50",
        evaluation="10,000 rows per replication",
        replications="10 (caption and code)",
        metric="sqrt PEHE, mean over replications",
        uncertainty="SD over replications (ddof=0)",
        methods="tau_CorNet; tau_Conf; tau_Unc",
        benchmark_preset="paper:cornet_official_n_conf",
        benchmark_expressible="approximate",
        benchmark_gap_or_overrides="cornet_delta_rms=1 (small) and 2 (large); in the large setting the OBS head is base minus delta rather than a norm-10 head.",
        benchmark_method_mapping="tau_CorNet -> cornet_official twice: standardize=False and the default standardize=True",
        extraction="digitized from PDF vector paths",
        notes="1) Raising 'complexity' here also doubles the bias magnitude; the paper's text acknowledges this for tau_Conf.",
    ),
    dict(
        experiment_id="hatt2022_fig6",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Figure 6, PDF p.33; Section 7.1.2",
        result_form="figure (vector)",
        code_reference=f"{CORNET_CODE} simulation_studies/sim_condition.py, simdata.py:sim_data_condition3",
        design=(
            "Rows: d_inf=1 and C_B small; d_inf>1 and C_B small; d_inf=1 and C_B large. x = sqrt(nConf/nUnc) "
            "with nUnc=50 and nConf = round(x^2*50). Series tau_CorNet and tau_Unc. " + FIG_COMMON
        ),
        d="25",
        n_obs="round(x^2*50) for 15 x values; 10 ... 300",
        n_rct="50",
        evaluation="10,000 rows",
        replications="unknown (released code runs 1; published error bars are non-zero)",
        metric="sqrt PEHE, mean over replications",
        uncertainty="SD over replications (ddof=0)",
        methods="tau_CorNet; tau_Unc",
        benchmark_preset="paper:cornet_official_n_conf",
        benchmark_expressible="partial",
        benchmark_gap_or_overrides="d_inf=1 rows via sizes and correction scale; the d_inf>1 row needs an RCT covariate-scale option.",
        benchmark_method_mapping="tau_CorNet -> cornet_official twice: standardize=False and the default standardize=True",
        extraction="digitized from PDF vector paths",
        notes=(
            "1) The released script loops `for k in range(1)`, which gives zero-length error bars, but the published "
            "bars are non-zero (SD 0.06-0.22). The figure therefore came from a different run configuration, and "
            "the replication count is unknown. "
            "2) The dashed 'improvement point' lines are drawn at hand-chosen grid indices (t[3], t[1], t[4]), not computed. "
            "3) The paper text states neither nUnc, the nConf range nor the replication count."
        ),
    ),
    dict(
        experiment_id="hatt2022_fig7",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Figure 7, PDF p.34; Section 7.1.2",
        result_form="figure (vector)",
        code_reference=f"{CORNET_CODE} simulation_studies/sim_condition_conf.py, simdata.py:sim_data_condition_conf",
        design=(
            "Rows: d_inf=1 and Delta=1; d_inf>1 and Delta=1; d_inf=1 and Delta=1.5. x = nUnc, nConf=300. "
            "Step 1 is fit once per replication; only step 2 is refit for each nUnc. Series tau_CorNet and tau_Conf. " + FIG_COMMON
        ),
        d="25",
        n_obs="300",
        n_rct="int(x): 3,4,5,6,7,8,9,10,11,12,15,17,20,22,25",
        evaluation="10,000 rows",
        replications="10 (code; caption silent)",
        metric="PEHE (MSE scale, not square-rooted)",
        uncertainty="SD over replications (ddof=0)",
        methods="tau_CorNet; tau_Conf",
        benchmark_preset="paper:cornet_official_n_conf",
        benchmark_expressible="partial",
        benchmark_gap_or_overrides="d_inf=1 rows via sizes and correction scale; the d_inf>1 row needs an RCT covariate-scale option; tiny RCT sizes can lack an arm.",
        benchmark_method_mapping="tau_CorNet -> cornet_official twice: standardize=False and the default standardize=True",
        extraction="digitized from PDF vector paths",
        notes=(
            "1) Unlike the other simulation figures, the y-axis is PEHE, not sqrt PEHE (code label epsilon_PEHE; rendered axis confirms). "
            "2) At nUnc <= 5 an RCT prefix lacks one arm with probability 2*0.5^n (25% at n=3). The authors' code does not check; "
            "the benchmark CorNet adapter rejects such samples as failures. "
            "3) Dashed lines are hand-placed (t[3]+0.5, t[4]+0.004, t[2]+0.05). "
            "4) Script constants are beta=1 and beta_2=sqrt(2); their mapping to the labels Delta=1 and Delta=1.5 was not re-derived."
        ),
    ),
    dict(
        experiment_id="hatt2022_fig8",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Figure 8, PDF p.35; Section 7.1.3(i)",
        result_form="figure (vector)",
        code_reference=f"{CORNET_CODE} simulation_studies/sim_assum_fail_shared_rep_n_conf.py, simdata.py:sim_data_assum_shared_rep",
        design="OBS and RCT outcomes use different representations; last-layer distance beta_phi = 0, small, large. nUnc=50. " + FIG_COMMON,
        d="25",
        n_obs="int(linspace(25,500,10))",
        n_rct="50",
        evaluation="10,000 rows",
        replications="10 (code; caption silent)",
        metric="sqrt PEHE, mean over replications",
        uncertainty="SD over replications (ddof=0)",
        methods="tau_CorNet; tau_Conf; tau_Unc",
        benchmark_preset="none",
        benchmark_expressible="no",
        benchmark_gap_or_overrides="The CorNet preset has one shared representation; source-specific representations need a DGP extension.",
        benchmark_method_mapping="tau_CorNet -> cornet_official twice: standardize=False and the default standardize=True",
        extraction="digitized from PDF vector paths",
        notes="1) Script constants infty_norm_1=1.5 and infty_norm_2=2.0 appear to set 'small' and 'large'; not re-derived. 2) y-axis range 0.5-4.5 differs from the other figures.",
    ),
    dict(
        experiment_id="hatt2022_fig9",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Figure 9, PDF p.36; Section 7.1.3(ii)",
        result_form="figure (vector)",
        code_reference=f"{CORNET_CODE} simulation_studies/sim_fail_os_bias.py, simdata.py:sim_data_bias, sim_data_n_conf",
        design="Delta = 4, 1, 0 (mean squared OBS-RCT CATE gap). nUnc=50. " + FIG_COMMON,
        d="25",
        n_obs="int(linspace(25,500,10))",
        n_rct="50",
        evaluation="10,000 rows",
        replications="10 (code; caption silent)",
        metric="sqrt PEHE, mean over replications",
        uncertainty="SD over replications (ddof=0)",
        methods="tau_CorNet; tau_Conf; tau_Unc",
        benchmark_preset="paper:cornet_official_n_conf",
        benchmark_expressible="approximate",
        benchmark_gap_or_overrides="cornet_delta_rms = 2, 1, 0 (0 is accepted as no discrepancy).",
        benchmark_method_mapping="tau_CorNet -> cornet_official twice: standardize=False and the default standardize=True",
        extraction="digitized from PDF vector paths",
        notes="1) Uses the same generator as Figure 5: Delta=4 and Delta=1 are its doubled and base OBS samples, and Delta=0 uses an RCT-law sample as OBS.",
    ),
    dict(
        experiment_id="hatt2022_fig10",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Figure 10, PDF p.38; Section 7.1.3(iii)",
        result_form="figure (vector)",
        code_reference=f"{CORNET_CODE} simulation_studies/sim_assum_fail_positivity.py, simdata.py:sim_data_assum_positivity",
        design=(
            "RCT covariates Uniform[-a,a]^d with a = 3, 1, 1/2 (top) or N(0, sigma_u^2 I) with sigma_u = 1, 1/3, 1/6 "
            "(bottom); OBS covariates N(0,I). nUnc=50. Representation width 20."
        ),
        d="10",
        n_obs="int(linspace(25,500,10))",
        n_rct="50",
        evaluation="10,000 rows",
        replications="10 (code; caption silent)",
        metric="sqrt PEHE, mean over replications",
        uncertainty="SD over replications (ddof=0)",
        methods="tau_CorNet; tau_Conf; tau_Unc",
        benchmark_preset="none",
        benchmark_expressible="no",
        benchmark_gap_or_overrides="The CorNet preset has no RCT covariate-law option.",
        benchmark_method_mapping="tau_CorNet -> cornet_official twice: standardize=False and the default standardize=True",
        extraction="digitized from PDF vector paths",
        notes=(
            "1) d=10 and width 20 here versus 25 elsewhere; the paper text does not state d. "
            "2) Footnote 17 says sigma_u gives P(X in [-a,a]^d) = 0.9973. With a/sigma_u = 3 that holds per coordinate; "
            "for d=10 the joint probability is 0.9973^10 = 0.973."
        ),
    ),
    dict(
        experiment_id="hatt2022_table3",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Table 3, PDF p.42; Sections 7.2.1-7.2.6",
        result_form="table",
        code_reference=f"{CORNET_CODE} real_world_exp/real_exp.py, real_actg.py, real_jobs.py",
        design=(
            "Real experiments STAR, ACTG 175 and NSW. Small RCT selected on one covariate (birthday, gender, age); "
            "OBS built by outcome-based selection (controls below mean - c*SD, treated above mean + c*SD; c = 1, 0, 0.25)."
        ),
        d="8 | 12 | 8",
        n_obs="643 | 552 | 482",
        n_rct="2d = 16 | 24 | 16",
        evaluation="Original data minus the RCT sample; ground truth = TARNet fit to the full experiment",
        replications="10",
        metric="sqrt empirical PEHE against the estimated ground truth",
        uncertainty="SD over runs",
        methods="2-step ridge; 2-step RF; 2-step NN; tau_Conf; tau_Unc; tau_Avg; tau_Weight; tau_CorNet; tau_CorNet+",
        benchmark_preset="none",
        benchmark_expressible="no",
        benchmark_gap_or_overrides="No STAR/ACTG/NSW adapters; the benchmark's empirical adapters simulate outcomes rather than select on observed outcomes.",
        benchmark_method_mapping="none",
        extraction="transcribed; checked against a 200-dpi rendering",
        notes=(
            "1) The ground truth is a TARNet estimate, not a known CATE. "
            "2) Section 7.2.1 describes NSW as 465 randomized subjects with 297 treated and 425 controls; 297 + 425 = 722. "
            "3) tau_CorNet on STAR is 0.59 +/- 0.01 here but 0.59 +/- 0.10 in Table 4 row (i), the same unregularized configuration; the other matching cells agree."
        ),
    ),
    dict(
        experiment_id="hatt2022_realdata_other",
        paper_id="hatt2022_cornet",
        source=f"{CORNET_SRC}; Figure 11 (p.44), Table 4 (p.45), Table 5 (p.80), Figure 13 (p.81), Table 6 (p.82), Figure 14 (p.83)",
        result_form="tables and figures",
        code_reference=f"{CORNET_CODE} real_world_exp/",
        design="Real-data sensitivity to nConf and nUnc, ablations, modified Kallus et al. baselines and multi-task variants.",
        d="8 | 12 | 8",
        n_obs="varied",
        n_rct="varied",
        evaluation="as Table 3",
        replications="10",
        metric="sqrt empirical PEHE",
        uncertainty="SD",
        methods="CorNet variants and baselines",
        benchmark_preset="none",
        benchmark_expressible="no",
        benchmark_gap_or_overrides="Same gap as Table 3.",
        benchmark_method_mapping="none",
        extraction="not extracted (inventory only)",
        notes="1) Listed so the inventory is complete; extract if STAR/ACTG/NSW adapters are added.",
    ),
    dict(
        experiment_id="wu2022_table1",
        paper_id="wu2022_intrlearner",
        source=f"{INTR_SRC}; Table 1, printed p.11; Section 6.1.1 (Simulation I)",
        result_form="table",
        code_reference=f"{INTR_CODE} R/intRlearner.R:gen_data(KPS_data=TRUE), gen_y",
        design=(
            "RCT: X~U[-1,1], U~N(0,1), A~Bernoulli(0.5). OS: A~Bernoulli(0.5), (X,U)|A ~ N(0, [[1, A-0.5], [A-0.5, 1]]). "
            "Y = 1 + X + 0.5X^2 + U + A*tau(X) + 0.5*eps, tau = 1 + 2X + 0.75X^2, confounding function c(X) = X."
        ),
        d="1",
        n_obs="2000",
        n_rct="200 | 500 | 1000",
        evaluation="T = 1e5 test X~U[-1,1]",
        replications="100",
        metric="RMSE: per replicate sqrt(mean_test (tau_hat - tau)^2), averaged over replicates",
        uncertainty="SE in parentheses",
        methods="rct oa; os oa; int oa; rct rlearner; os rlearner; int rlearner; rct rf; os rf; KPS lasso; KPS linear; naive rlearner",
        benchmark_preset="paper:integrative_r_learner_kps_code",
        benchmark_expressible="yes",
        benchmark_gap_or_overrides="Set n_obs=2000, n_rct_train in {200, 500, 1000} and n_eval=100000.",
        benchmark_method_mapping="int rlearner -> integrative_r_quadratic (degree-2 basis, SCAD); other columns have no matching recipe",
        extraction="transcribed; checked against a 220-dpi rendering",
        notes=(
            "1) The released README labels Simulations I and II in reverse; the benchmark follows the paper. "
            "2) In this branch gen_data returns tau = 0 (true_tau_coef is built but not returned), so RMSE against the returned tau "
            "would be wrong; the true CATE is 1 + 2x + cf*x^2, and cf defaults to 0 in the signature while the paper uses 0.75. "
            "3) The driver that produced Tables 1 and 2 is not released. "
            "4) OS-only columns are printed once, in the n=500 row, and apply to every n. "
            "5) Per the README, the rct/os/naive R-learners use the L1 rlasso while int rlearner uses SCAD, so their regularizers differ. "
            "6) A replay of the released code over 100 seeds (replay_intr_tables) agrees with 9 of the 10 R-learner values, including "
            "all three int rlearner values; naive rlearner at n=1000 gives 0.490 against 0.50. The benchmark translation agrees with "
            "the released code at every n (int rlearner 0.234, 0.134 and 0.088)."
        ),
    ),
    dict(
        experiment_id="wu2022_table2",
        paper_id="wu2022_intrlearner",
        source=f"{INTR_SRC}; Table 2, printed p.11; Section 6.1.2 (Simulation II)",
        result_form="table",
        code_reference=f"{INTR_CODE} R/intRlearner.R:gen_data(KPS_data=FALSE)",
        design=(
            "Population 1e5, X~N(0,I_10), X10 unobserved. Y(a) = X1 + X10 + a*tau(W) + eps(a), eps~N(0,1), tau = 1 + sum_{j<=J} X_j. "
            "RCT selection logit = -7 - X1 + X2; RCT propensity logit = -1 - X1 + X2. OS: 500 units sampled from the population, "
            "propensity logit = -1 - X1 + X2 + b*X10."
        ),
        d="9 observed",
        n_obs="500",
        n_rct="random, expected 243.7 (SD 15.6); text says around 200",
        evaluation="All 1e5 population units (includes RCT and OS units)",
        replications="100",
        metric="RMSE averaged over replicates",
        uncertainty="SE in parentheses",
        methods="rct oa; os oa; int oa; rct rlearner; os rlearner; int rlearner; rct rf; os rf; KPS lasso; KPS linear; naive rlearner",
        benchmark_preset="paper:integrative_r_learner_multivariate_paper",
        benchmark_expressible="yes (approximate RCT size)",
        benchmark_gap_or_overrides="Preset is J=9, b=2.5; J=1 via eight zero modifiers, b=0 via confounding strength 0. The benchmark draws fixed-size RCT roles, so choose n_rct_train (e.g. 244).",
        benchmark_method_mapping="int rlearner -> integrative_r_quadratic; other columns have no matching recipe",
        extraction="transcribed; checked against a 220-dpi rendering",
        notes=(
            "1) The expected RCT size is 243.7 (SD 15.6) under the selection model; the text says around 200. "
            "2) README p0 suggestions (1 or p-2) give J=8 or J=1; J=9 needs p0=0. The code comment calls p0 the number of non-zero "
            "coefficients, but the code uses it as the number of zero coefficients. "
            "3) The README example (J=8, b=2.5, seed 0, one replicate) prints RMSE 0.866 int, 1.559 rct, 1.457 os, 1.303 naive; "
            "it is a different setting from Table 2. "
            "4) The J=1, b=2.5 os rlearner standard-error line prints '1.33(0.01)', duplicating the value printed above it. "
            "5) RCT-only columns are printed once per J and apply to both b. "
            "6) The test set includes the units used for fitting. "
            "7) In R 4.3.3 the README example reproduces its printed RMSEs to seven decimals. A replay of the released code over "
            "100 seeds (replay_intr_tables) agrees with 11 of the 14 R-learner values. The three that differ are in the J=9 rows and "
            "are higher in the replay: rct rlearner 1.76 against 1.64, and at b=0 os rlearner 1.07 against 0.97 and naive rlearner "
            "0.82 against 0.77. The replay's RCT size averages 242.8 (SD 14.0). The benchmark translation agrees with the released "
            "code in all four cells."
        ),
    ),
    dict(
        experiment_id="wu2022_star",
        paper_id="wu2022_intrlearner",
        source=f"{INTR_SRC}; Section 6.2 text and Figure 1, printed pp.11-13",
        result_form="text and figure",
        code_reference="not released",
        design=(
            "STAR: RCT = half of rural or inner-city students; OS = remaining controls plus lower-half treated outcomes, "
            "plus urban/suburban students with the same rule. Nuisances by XGBoost or random forest; 20 iterations."
        ),
        d="7 (gender, race, birth month, birthday, birth year, free lunch, teacher id)",
        n_obs="derived from 4,218 students",
        n_rct="about 1,405 (half of 2,811)",
        evaluation="Held-out sample excluding the RCT; ground truth = per-unit IPW pseudo-outcome",
        replications="not stated",
        metric="RMSE against the pseudo-outcome",
        uncertainty="none",
        methods="int rlearner; KPS lasso; RCT-only and OS-only estimators",
        benchmark_preset="none",
        benchmark_expressible="no",
        benchmark_gap_or_overrides="No STAR adapter.",
        benchmark_method_mapping="none",
        extraction="from text (approximate where the paper writes approximately)",
        notes=(
            "1) The STAR counts here (4,218; 2,413 control, 1,805 treated) differ from the CorNet paper's STAR preprocessing "
            "(4,139; 2,365 and 1,774); the inclusion rules differ, so this is not an error in either paper. "
            "2) The RCT size is computed from the stated rule, not printed."
        ),
    ),
]

# ---------------------------------------------------------------- digitized figures
SERIES = {"black": "tau_CorNet", "blue": "tau_Conf", "red": "tau_Unc"}
BENCH = {"tau_CorNet": "cornet_official twice: standardize=False and the default standardize=True", "tau_Conf": "", "tau_Unc": ""}
LIN = np.linspace(25, 500, 10)
T1 = np.concatenate((np.linspace(0.4472136, 1.5, 10), np.linspace(1.5, 2.44948974, 6)[1:]))
FIG6_X = np.sqrt(np.round(T1 ** 2 * 50) / 50)
FIG7_X = np.concatenate((np.linspace(3, 12.5, 10), np.linspace(12.5, 25, 6)[1:]))

PAGES = {
    29: [("hatt2022_fig3", "nUnc=25"), ("hatt2022_fig3", "nUnc=50"), ("hatt2022_fig3", "nUnc=100")],
    31: [("hatt2022_fig4", "sigma_u=1 (d_inf=1)"), ("hatt2022_fig4", "sigma_u=0.1 (d_inf>1)"),
         ("hatt2022_fig5", "C_B small"), ("hatt2022_fig5", "C_B large")],
    33: [("hatt2022_fig6", "d_inf=1, C_B small"), ("hatt2022_fig6", "d_inf>1, C_B small"), ("hatt2022_fig6", "d_inf=1, C_B large")],
    34: [("hatt2022_fig7", "d_inf=1, Delta=1"), ("hatt2022_fig7", "d_inf>1, Delta=1"), ("hatt2022_fig7", "d_inf=1, Delta=1.5")],
    35: [("hatt2022_fig8", "beta_phi=0 (shared)"), ("hatt2022_fig8", "beta_phi small"), ("hatt2022_fig8", "beta_phi large")],
    36: [("hatt2022_fig9", "Delta=4"), ("hatt2022_fig9", "Delta=1"), ("hatt2022_fig9", "Delta=0")],
    38: [("hatt2022_fig10", "X_Unc~U[-3,3]^d"), ("hatt2022_fig10", "X_Unc~U[-1,1]^d"), ("hatt2022_fig10", "X_Unc~U[-1/2,1/2]^d"),
         ("hatt2022_fig10", "X_Unc~N(0,1^2 I)"), ("hatt2022_fig10", "X_Unc~N(0,(1/3)^2 I)"), ("hatt2022_fig10", "X_Unc~N(0,(1/6)^2 I)")],
}
FIG3_RCT = {"nUnc=25": 25, "nUnc=50": 50, "nUnc=100": 100}


def nearest(grid, x):
    i = int(np.argmin(np.abs(grid - x)))
    assert abs(grid[i] - x) < 1e-3, (x, grid[i])
    return grid[i]


rows = []
for page, panels in PAGES.items():
    raw = list(csv.DictReader(open(ROOT / "raw" / f"hatt2022_page{page}_digitized.csv")))
    for r in raw:
        exp, label = panels[int(r["panel"])]
        method = SERIES[r["series"]]
        x = float(r["x"])
        if exp == "hatt2022_fig6":
            xg = nearest(FIG6_X, x); x_name = "sqrt(nConf/nUnc)"; n_obs = int(round(xg ** 2 * 50)); n_rct = 50; metric = "sqrt_pehe"
        elif exp == "hatt2022_fig7":
            xg = nearest(FIG7_X, x); x_name = "nUnc"; n_obs = 300; n_rct = int(xg); metric = "pehe"
        else:
            xg = nearest(LIN, x); x_name = "nConf"; n_obs = int(xg); n_rct = FIG3_RCT.get(label, 50); metric = "sqrt_pehe"
        reps = "unknown" if exp == "hatt2022_fig6" else "10"
        note = ""
        if method == "tau_Unc":
            note = "RCT-only fit; constant across x within the panel"
        elif method == "tau_Conf" and exp == "hatt2022_fig7":
            note = "OBS-only fit; constant across nUnc within the panel"
        rows.append(dict(
            experiment_id=exp, panel=label, method_paper=method, benchmark_method=BENCH[method],
            x_name=x_name, x_plotted=f"{xg:.6g}", n_obs=n_obs if method != "tau_Unc" else "",
            n_rct=n_rct if method != "tau_Conf" else "", condition=label, metric=metric,
            value=f"{float(r['mean']):.4f}", uncertainty=f"{float(r['sd']):.4f}",
            uncertainty_type="sd_over_replications_ddof0", n_replications=reps,
            value_source="vector_digitized", row_notes=note,
        ))

# ---------------------------------------------------------------- CorNet Table 3
T3 = {
    "2-step ridge": ("3.01", "0.01", "1.51", "0.01", "2.82", "0.02"),
    "2-step RF": ("3.14", "0.03", "1.58", "0.07", "3.10", "0.12"),
    "2-step NN": ("3.03", "0.02", "1.60", "0.02", "2.82", "0.02"),
    "tau_Conf": ("2.66", "0.01", "1.08", "0.04", "0.85", "0.04"),
    "tau_Unc": ("1.33", "0.22", "1.10", "0.11", "0.52", "0.15"),
    "tau_Avg": ("1.68", "0.35", "0.91", "0.29", "0.51", "0.20"),
    "tau_Weight": ("2.26", "0.09", "0.80", "0.10", "0.76", "0.07"),
    "tau_CorNet": ("0.59", "0.01", "0.42", "0.06", "0.14", "0.07"),
    "tau_CorNet+": ("0.38", "0.07", "0.27", "0.03", "0.21", "0.08"),
}
T3_SIZES = {"STAR": (643, 16), "ACTG": (552, 24), "NSW": (482, 16)}
for method, vals in T3.items():
    for k, ds in enumerate(("STAR", "ACTG", "NSW")):
        note = "Table 4 row (i) reports 0.59 +/- 0.10 for the same configuration" if (method, ds) == ("tau_CorNet", "STAR") else ""
        rows.append(dict(
            experiment_id="hatt2022_table3", panel=ds, method_paper=method, benchmark_method="",
            x_name="", x_plotted="", n_obs=T3_SIZES[ds][0], n_rct=T3_SIZES[ds][1], condition=ds,
            metric="sqrt_pehe_vs_estimated_truth", value=vals[2 * k], uncertainty=vals[2 * k + 1],
            uncertainty_type="sd_over_runs", n_replications="10", value_source="table_transcribed", row_notes=note,
        ))

# ---------------------------------------------------------------- Integrative R Table 1
COLS = ["rct oa", "os oa", "int oa", "rct rlearner", "os rlearner", "int rlearner", "rct rf", "os rf", "KPS lasso", "KPS linear", "naive rlearner"]
OS_ONLY = {"os oa", "os rlearner", "os rf"}
RCT_ONLY = {"rct oa", "rct rlearner", "rct rf"}
T1_VALUES = {
    200: {"rct oa": ("0.28", "0.01"), "int oa": ("0.29", "0.02"), "rct rlearner": ("0.26", "0.01"), "int rlearner": ("0.23", "0.01"),
          "rct rf": ("0.38", "0.01"), "KPS lasso": ("0.43", "0.01"), "KPS linear": ("0.32", "0.01"), "naive rlearner": ("0.56", "0.00")},
    500: {"rct oa": ("0.16", "0.01"), "os oa": ("0.57", "0.00"), "int oa": ("0.13", "0.01"), "rct rlearner": ("0.15", "0.01"),
          "os rlearner": ("0.58", "0.00"), "int rlearner": ("0.11", "0.01"), "rct rf": ("0.25", "0.01"), "os rf": ("0.62", "0.01"),
          "KPS lasso": ("0.29", "0.01"), "KPS linear": ("0.25", "0.01"), "naive rlearner": ("0.53", "0.00")},
    1000: {"rct oa": ("0.11", "0.00"), "int oa": ("0.08", "0.00"), "rct rlearner": ("0.11", "0.00"), "int rlearner": ("0.09", "0.01"),
           "rct rf": ("0.23", "0.00"), "KPS lasso": ("0.25", "0.01"), "KPS linear": ("0.22", "0.00"), "naive rlearner": ("0.50", "0.00")},
}
for n, cells in T1_VALUES.items():
    for method in COLS:
        if method not in cells:
            continue
        shared = method in OS_ONLY
        rows.append(dict(
            experiment_id="wu2022_table1", panel="all n" if shared else f"n={n}", method_paper=method,
            benchmark_method="integrative_r_quadratic" if method == "int rlearner" else "",
            x_name="n_rct", x_plotted="", n_obs=2000 if method not in RCT_ONLY else "", n_rct="" if shared else n,
            condition="all n" if shared else f"n={n}", metric="rmse", value=cells[method][0], uncertainty=cells[method][1],
            uncertainty_type="se_over_replicates", n_replications="100", value_source="table_transcribed",
            row_notes="printed once in the n=500 row; OS-only, applies to every n" if shared else "",
        ))

# ---------------------------------------------------------------- Integrative R Table 2
T2_RCT = {1: {"rct oa": ("1.12", "0.14"), "rct rlearner": ("0.88", "0.04"), "rct rf": ("1.19", "0.02")},
          9: {"rct oa": ("1.87", "0.13"), "rct rlearner": ("1.64", "0.03"), "rct rf": ("2.54", "0.01")}}
T2 = {
    (1, "0"): {"os oa": ("0.55", "0.02"), "int oa": ("0.86", "0.06"), "os rlearner": ("0.49", "0.02"), "int rlearner": ("0.28", "0.03"),
               "os rf": ("0.75", "0.01"), "KPS lasso": ("0.99", "0.03"), "KPS linear": ("1.45", "0.12"), "naive rlearner": ("0.41", "0.01")},
    (1, "2.5"): {"os oa": ("1.40", "0.01"), "int oa": ("1.35", "0.05"), "os rlearner": ("1.33", "0.01"), "int rlearner": ("0.48", "0.04"),
                 "os rf": ("1.31", "0.01"), "KPS lasso": ("0.83", "0.02"), "KPS linear": ("1.44", "0.11"), "naive rlearner": ("1.01", "0.01")},
    (9, "0"): {"os oa": ("1.08", "0.03"), "int oa": ("1.45", "0.08"), "os rlearner": ("0.97", "0.02"), "int rlearner": ("0.55", "0.03"),
               "os rf": ("2.33", "0.01"), "KPS lasso": ("2.22", "0.03"), "KPS linear": ("1.75", "0.13"), "naive rlearner": ("0.77", "0.01")},
    (9, "2.5"): {"os oa": ("1.55", "0.02"), "int oa": ("1.95", "0.08"), "os rlearner": ("1.54", "0.01"), "int rlearner": ("1.12", "0.05"),
                 "os rf": ("2.63", "0.01"), "KPS lasso": ("2.18", "0.03"), "KPS linear": ("1.70", "0.12"), "naive rlearner": ("1.31", "0.01")},
}
for J, cells in T2_RCT.items():
    for method, (v, se) in cells.items():
        rows.append(dict(
            experiment_id="wu2022_table2", panel=f"J={J}, all b", method_paper=method, benchmark_method="",
            x_name="", x_plotted="", n_obs="", n_rct="random (expected 243.7)", condition=f"J={J}, all b", metric="rmse",
            value=v, uncertainty=se, uncertainty_type="se_over_replicates", n_replications="100", value_source="table_transcribed",
            row_notes="printed once per J; RCT-only, applies to both b",
        ))
for (J, b), cells in T2.items():
    for method in COLS:
        if method not in cells:
            continue
        note = "SE line prints '1.33(0.01)', duplicating the value" if (J, b, method) == (1, "2.5", "os rlearner") else ""
        rows.append(dict(
            experiment_id="wu2022_table2", panel=f"J={J}, b={b}", method_paper=method,
            benchmark_method="integrative_r_quadratic" if method == "int rlearner" else "",
            x_name="", x_plotted="", n_obs=500, n_rct="" if method in OS_ONLY else "random (expected 243.7)",
            condition=f"J={J}, b={b}", metric="rmse", value=cells[method][0], uncertainty=cells[method][1],
            uncertainty_type="se_over_replicates", n_replications="100", value_source="table_transcribed", row_notes=note,
        ))

# ---------------------------------------------------------------- Integrative R STAR text values
for method, value, src, note in (
    ("int rlearner", "70", "text_approximate", "paper: RMSE approximately 70 after 20 iterations (Figure 1)"),
    ("KPS lasso", "71.01", "text", ""),
    ("RCT-only and OS-only estimators", "80", "text_approximate", "paper: around 80, not shown in Figure 1"),
):
    rows.append(dict(
        experiment_id="wu2022_star", panel="STAR", method_paper=method,
        benchmark_method="", x_name="", x_plotted="", n_obs="", n_rct="", condition="STAR", metric="rmse_vs_pseudo_outcome",
        value=value, uncertainty="", uncertainty_type="none", n_replications="not stated", value_source=src, row_notes=note,
    ))

# ---------------------------------------------------------------- checks and output
EXPECTED = {"hatt2022_fig3": 90, "hatt2022_fig4": 60, "hatt2022_fig5": 60, "hatt2022_fig6": 90, "hatt2022_fig7": 90,
            "hatt2022_fig8": 90, "hatt2022_fig9": 90, "hatt2022_fig10": 180, "hatt2022_table3": 27,
            "wu2022_table1": 27, "wu2022_table2": 38, "wu2022_star": 3, "hatt2022_realdata_other": 0}
counts = {e["experiment_id"]: 0 for e in EXPERIMENTS}
for r in rows:
    counts[r["experiment_id"]] += 1
    float(r["value"])
assert counts == EXPECTED, counts
for e in EXPERIMENTS:
    e["n_values"] = counts[e["experiment_id"]]

exp_fields = ["experiment_id", "paper_id", "source", "result_form", "code_reference", "design", "d", "n_obs", "n_rct",
              "evaluation", "replications", "metric", "uncertainty", "methods", "benchmark_preset", "benchmark_expressible",
              "benchmark_gap_or_overrides", "benchmark_method_mapping", "extraction", "n_values", "notes"]
val_fields = ["experiment_id", "panel", "method_paper", "benchmark_method", "x_name", "x_plotted", "n_obs", "n_rct",
              "condition", "metric", "value", "uncertainty", "uncertainty_type", "n_replications", "value_source", "row_notes"]
with open(ROOT / "experiments.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=exp_fields); w.writeheader(); w.writerows(EXPERIMENTS)
with open(ROOT / "reported_values.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=val_fields); w.writeheader(); w.writerows(rows)
print("experiments:", len(EXPERIMENTS), "values:", len(rows))
print(counts)
