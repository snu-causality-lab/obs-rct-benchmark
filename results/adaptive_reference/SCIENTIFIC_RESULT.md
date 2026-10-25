# What the adaptive recruitment comparison shows

All **1,152 replicate runs** completed and replayed: 36 combinations of mechanism, budget, policy and final estimator, each with 32 independent sample seeds. Budgets are 128/512 and policies are uniform, geometry and outcome-residual. Every policy purchases the same 32-row pilot, of which 16 rows enter final training and 16 enter final tuning; all later acquired rows enter training. Thus final fitting sizes are 112/496, plus 16 charged tuning labels. The tuning labels have already been seen by recruitment and are not held out from policy selection. See the [fixed protocol](../../docs/ADAPTIVE.md).

## Outcome-assisted recruitment has conditional benefits and costs

The [three mechanisms](../../docs/COVERAGE.md#structural-mechanisms) are M0 (clean reference), M3 (localized hidden confounding) and M9 (weak trial coverage of an effect region). The residual policy uses only acquired randomized outcomes and OBS-fitted predictions. Its difference from uniform recruitment depends on both the mechanism and the final estimator. For the RCT-only T-learner, the M3/budget128 difference is **+0.13387**, pointwise 95% interval **[+0.02900, +0.23874]**; in M9 it is **−0.26509 [−0.41462, −0.11555]** at budget128 and **−0.11617 [−0.18901, −0.04333]** at budget512. Positive means worse than uniform. These pointwise Monte Carlo summaries have no multiplicity adjustment and are not a simultaneous superiority claim.

Geometry has negative mean differences in all six mechanism/budget conditions for each final estimator, with several intervals including zero. The outcome-residual rule has negative means in two of six RCT-only conditions and five of six naive-pooled conditions. Improving a naive pooled learner does not establish that borrowing is preferable to an RCT-only learner: both the policy and the fitted estimator matter.

The implementation evaluates one adaptive choice after a fixed pilot, followed by fixed acquisition prefixes. It does not repeatedly learn an optimal acquisition policy, model enrollment/screening cost or establish unseen-mechanism generalization.

- [All budget curves](adaptive_budget_curves.svg), [same-estimator policy differences](adaptive_policy_contrasts.svg)
- [All 36 score rows](scores.csv), [all 24 paired policy contrasts](policy_contrasts.csv), [complete analysis](summary.json)
- [Selected query transcripts, manifests and replay counts](runs/)

## Reproduce and verify

Each shard's `replay.json` records the complete deterministic reconstruction of candidate pools, acquisition orders, exact costs, final predictions and scores. All 1,152 successful cells replayed; no terminal failures were retried or omitted. Aggregation separately authenticates the complete disjoint seed grid, source/environment identity and every stored cost. Replay is distinct from independent empirical replication.

Generate eight disjoint configurations with `python scripts/make_reference_configs.py --config configs/adaptive_reference.toml --output outputs/adaptive-configs --shards 8`. Execute each with `python -m obs_rct_benchmark.adaptive_runner --config outputs/adaptive-configs/shard-00.toml --output outputs/adaptive/shard-00`, then repeat with `--resume` and `--verify`. Aggregate all eight with `python scripts/analyze_adaptive_study.py --config configs/adaptive_reference.toml --runs outputs/adaptive/shard-* --output outputs/adaptive-analysis`. Add `--paper-assets` to regenerate vector PDFs. The public selection retains SVG figures; [delivery metadata](DELIVERY_MANIFEST.json) records the separately retained PDF hashes.
