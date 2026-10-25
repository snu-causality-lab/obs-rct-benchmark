# Adaptive recruitment reference comparison

Terminal grid: 1152/1152; successful 1152, failed 0, missing 0.

[Budget curves](adaptive_budget_curves.svg) retain every mechanism, budget, policy and final estimator. [Policy contrasts](adaptive_policy_contrasts.svg) hold the final estimator fixed. The data are in [scores.csv](scores.csv), [policy_contrasts.csv](policy_contrasts.csv) and [summary.json](summary.json).

Each queried outcome, including the common pilot and tuning sample, is charged. The source pool and independent evaluation rows are shared within each mechanism/seed. The outcome policy is a residual-based heuristic and pooling is naive. Pointwise paired Monte Carlo intervals have no multiplicity adjustment; favorable, adverse and unresolved directions are all retained. Fresh seeds are sampling replications on development-inspected mechanisms, not an independent mechanism challenge.

Input manifests and records are authenticated by seed-grid/config/source/environment checks and file hashes in summary.json. This aggregation does not independently refit predictions; run the separate adaptive replay verifier for that check.
