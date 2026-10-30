# OBS access and competitiveness answer different questions

This development comparison covers six inspected mechanisms, two RCT budgets, eleven recipes and 64 seeds, fixed before comparative fitting. Its **8,448 requested fits** comprise **8,350 successes, 64 failures, 31 timeouts and three preserved missing attempts**. Separate incomplete-collection verification reconstructs trial targets and successful prediction scores; the original complete-union contract remains unmet. **All 72 joint decisions and the simultaneous precision statement are withheld.** Mean-risk uncertainty was never established.

## Improving a backbone does not establish competitiveness

In M2 at RCT128, GP Full has mean CATE-MSE **0.35128**, below its matched RCT-only GP's **0.53675**. The RCT-only causal forest has mean **0.18386**. At RCT512 the corresponding values are **0.16670**, **0.18775** and **0.084582**. Every one of these method-condition rows has 64 successful seeds. These are descriptive means for the two stated conditions, without a significance or global-ranking claim.

GP Full (`causal_icm_fixed`) versus GP RCT-only (`causal_icm_rct_only`) measures OBS access within an admitted common backbone, including separately fitted kernel/noise hyperparameters. It does not isolate a posterior update at fixed fitted hyperparameters. Fusion-versus-forest compares complete procedures with different models and native fitting rules. CorNet, PPI and Integrative R have no admitted matched source-excluded controls here, so their differences from forest cannot isolate OBS contribution.

## All twelve matched GP conditions

Both GP procedures have 64/64 successes in each condition. Full has lower descriptive mean MSE throughout the fixed set; negative differences favor Full. Strict wins compare finite 2,048-query MSE on seed blocks, not expected population risk. Complete GP pairs do not release the withheld 72-comparison family.

| Mechanism | RCT rows | GP Full MSE | GP RCT-only MSE | Mean Full−RCT | Strict wins / 64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 128 | 0.323223 | 0.556977 | −0.233754 | 59 |
| M0 | 512 | 0.168083 | 0.189171 | −0.0210884 | 53 |
| M2 | 128 | 0.351282 | 0.536750 | −0.185468 | 55 |
| M2 | 512 | 0.166702 | 0.187747 | −0.0210453 | 55 |
| M10 | 128 | 0.314972 | 0.429292 | −0.114321 | 49 |
| M10 | 512 | 0.140920 | 0.152575 | −0.0116548 | 47 |
| C08 trial target | 128 | 0.292280 | 0.449988 | −0.157708 | 62 |
| C08 trial target | 512 | 0.154698 | 0.176219 | −0.0215212 | 46 |
| Native KPS code | 128 | 0.0927577 | 0.142623 | −0.0498649 | 56 |
| Native KPS code | 512 | 0.0252814 | 0.0344864 | −0.00920502 | 54 |
| Native CorNet | 128 | 0.398701 | 0.938937 | −0.540236 | 64 |
| Native CorNet | 512 | 0.167714 | 0.171762 | −0.00404807 | 50 |

The [controlled GP experiment](../gp_source_pilot/SCIENTIFIC_RESULT.md) finds adverse OBS transfer at its largest discrepancy on another generator. Each study retains its own conditions, seeds and analysis. These construction-specific results establish neither universal OBS benefit nor unseen-mechanism generalization.

## Numerical coverage is part of the finding

| Recipe | Requested | Success | Failure | Timeout | Missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| RCT causal forest | 768 | 734 | 0 | 31 | 3 |
| RCT T ExtraTrees | 768 | 768 | 0 | 0 | 0 |
| RCT DR ExtraTrees | 768 | 768 | 0 | 0 | 0 |
| OBS T ExtraTrees | 768 | 768 | 0 | 0 | 0 |
| Naive pooled T ExtraTrees | 768 | 768 | 0 | 0 | 0 |
| CorNet | 768 | 768 | 0 | 0 | 0 |
| DataFusionPPI | 768 | 768 | 0 | 0 | 0 |
| Integrative R linear | 768 | 768 | 0 | 0 | 0 |
| Integrative R quadratic | 768 | 704 | 64 | 0 | 0 |
| Causal-ICM Full | 768 | 768 | 0 | 0 | 0 |
| Causal-ICM RCT-only | 768 | 768 | 0 | 0 | 0 |

All 64 failures occur in quadratic Integrative R on native CorNet at RCT512; that row has no successful risk estimate. Its unscored d25 profile reached the native 1,200-second deadline. The fixed basis, search and resource contract define the executable recipe being evaluated; failure does not make the method scientifically inapplicable.

All 31 outer timeouts and three unresolved attempts belong to forest. Its 120-second outer limit includes runtime inventory, fitting and prediction; its native subprocess has a separate 600-second limit. Some interruptions occurred during inventory. These counts therefore do not isolate statistical fitting time or intrinsic reliability. Means with fewer than 64 successful seeds are conditional on success.

A temporary-file permission error prevented terminal publication for the three [unresolved forest attempts](non_successful_fits.csv). Timing is consistent with the outer limit, but parent timeout and cleanup evidence is unavailable. Their status remains missing, without an invented terminal outcome or retry.

## Targets, information and uncertainty

All recipes are scored on trial-population CATE, with independent trial-law queries. **C08 trial target is different from canonical catalog C08**; their scalar errors cannot be substituted. Fusion receives 512 OBS outcomes and 128/512 RCT fitting outcomes; RCT references receive no OBS outcomes. No extra tuning labels are supplied. Equal RCT-label access does not imply equal model capacity, tuning or compute expenditure.

The packet contains all [132 method-condition rows](method_summaries.csv), [72 comparisons](paired_comparisons.csv), [per-seed scores/statuses](scalar_records.csv) and [98 non-successful or missing identities](non_successful_fits.csv). Incomplete paired means and probabilities are withheld. Complete pointwise strict-win outputs, including stored p-values, remain descriptive; they authorize no joint decision. Mean root-PEHE averages seedwise square roots, not the square root of mean MSE. Raw MSEs across different target scales do not form a global ranking.

## Verification and reproduction

The [verification receipt](evidence/verification_receipt.json) binds the independently reconstructed [summary](summary.json) to exactly 8,445 original terminal records and the three designated missing identities. The separate analysis authenticates trial truth and all successful saved-prediction scores; it does not claim a complete union or independent estimator implementation.

The [reproduction guide](README.md) supplies the fitting/source identities, public arithmetic command and retained execution evidence. Full observed/query/prediction arrays, environments, process logs and interrupted-attempt artifacts stay on research storage. Public scalar checks reconstruct aggregate arithmetic; they do not repeat native fitting or raw truth/prediction verification. Study-specific inference remains separate. Empirical-data evaluation, additional methods/matched controls, controlled selection and independently frozen mechanism classes are future work in the [comparison design](../../docs/V03_DESIGN.md).
