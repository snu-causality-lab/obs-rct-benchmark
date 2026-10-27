# Inference permissions

Generic static and adaptive runs report descriptive CATE MSE and root-PEHE scores. Approximate Student-t uncertainty additionally requires complete pairs, at least two independent seed blocks and a separate permission for the metric in every contributing condition. Withholding uncertainty preserves complete paired means and record counts: expected, successful, failed and missing. If any requested pair is failed or missing, the entire comparison has no paired mean or uncertainty; successful-cell descriptive scores remain available.

Permissions are fixed before fitting from the configuration and resolved generator controls. They declare finite-variance assumptions for the requested procedures; they do not prove moment existence, adequate replication or confidence-interval coverage. In particular, Gaussian outcome noise alone does not establish finite moments for an arbitrary estimator.

## Resolution rules

The runner resolves each metric in this order:

1. Actual Student-t(3) outcome noise in a structural or addendum configuration blocks mean-MSE standard errors and intervals, even when `cate_mse` is explicitly declared eligible. This uses resolved controls, so changing a preset's noise is reflected in the policy.
2. An explicit scenario declaration governs each unblocked metric. Each declaration covers per-method score summaries and all requested method/reference comparisons, including the declared method options. It assumes finite variance for both the metric seed scores and their paired differences.
3. Canonical normal-noise structural/addendum presets and canonical paper presets have a built-in assumption of finite variance for MSE and root-PEHE seed scores and paired differences when only sample budgets or labels vary and all requested methods use unchanged options from the eight generic Ridge/ExtraTrees T- and RCT DR-learner recipes. This is an assumption, not a moment proof.
4. Other settings withhold uncertainty unless declared. These include native, external and custom methods, Gamma and empirical generators, scientific parameter overrides and changed method options.

For Student-t(3) configurations, root-PEHE defaults separately to withheld because finite expected fitted-method MSE has not been established. An explicit root-PEHE declaration can permit its uncertainty. A metric in an overall comparison is withheld if any contributing condition lacks permission for that metric. Conditions are averaged within a seed before uncertainty is computed; their count never increases the number of independent seed blocks.

The eight default recipes are `rct_t_ridge`, `obs_t_ridge`, `naive_pooled_t_ridge`, `rct_dr_ridge` and their `extra_trees` counterparts. Built-in permission requires absent or empty method options and only `n_obs`, `n_rct_train`, `n_rct_tune`, `n_rct_test`, `n_eval` or `n_target_reference` in `parameters`; supplying any other parameter key requires a declaration. The [preset catalog](COVERAGE.md) identifies the canonical designs. The [pairing protocol](PROTOCOL.md#scores-pairing-and-failures) defines difference orientation, condition weights and the uncertainty unit.

## Why MSE and root-PEHE differ

The [Student-t density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) with three degrees of freedom decays as `|x|^-4`. Its second moment exists but its fourth moment diverges, consistent with [NIST's moment table](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm). Squared-risk variance can therefore be infinite. More seeds or finite observed scores do not establish the assumptions needed for the ordinary mean standard error. The policy does not assume cancellation of heavy tails in an arbitrary paired difference.

For replicate MSE `M` and root-PEHE `R = sqrt(M)`, `E[R²] = E[M]`. Also, `(R_a - R_b)² <= 2(M_a + M_b)`, so finite expected MSE for both fitted procedures suffices for finite variance of their paired root difference. Missing fourth noise moments alone do not imply infinite root-PEHE variance. This is why its permission and withholding reason are separate.

## Configuration and Python API

Attach declarations to a scenario, outside its generator `parameters`. For example, after choosing procedures and justifying their moments prospectively:

```toml
[[scenarios]]
kind = "structural"
id = "M0"

[scenarios.inference_policy.cate_mse]
eligible = true
reason = "Finite variance of MSE scores and paired differences assumed for the declared procedures and options"

[scenarios.inference_policy.root_pehe]
eligible = true
reason = "Finite expected MSE assumed for both procedures in every declared pair"
```

Each declaration requires a boolean `eligible` and a nonempty one-line `reason`. Set `eligible = false` to explicitly withhold a metric. User configuration accepts no `source` field. A missing metric follows the resolution rules above, independently of the other metric. Omitted or empty `inference_policy` fields preserve the earlier configuration payload and hash; nonempty declarations become part of run identity. `ScenarioConfig(..., inference_policy={...})` accepts the same metric map in Python.

Both runners record the resolved `scenario_inference` map in `run_manifest.json` and `summary.json`, including the source of each decision. Adaptive runs expand it to each condition/budget key and use it for policy comparisons and raw metric summaries. Changing a declaration changes the configuration identity; changed source fingerprints also prevent resuming an old run under the new policy.

Direct callers of `summarize_records(..., scenario_inference=...)` supply a map from scenario keys to the same per-metric declarations, for example `{"structural:M0": {"cate_mse": {"eligible": True, "reason": "Finite score and paired-difference variance assumed"}}}`. A labeled static key is `structural:M0@rct64`; an adaptive key is `structural:M0/budget_128`. Omitting the map, a scenario or a metric withholds that metric's uncertainty; the low-level summarizer does not infer assumptions from a scenario name. The mathematical `mean_ci` helper remains policy-free.

Every paired metric exposes `eligible_for_inference` and `withheld_reason`. Eligibility means that permission, complete coverage and at least two seed blocks all hold. When several requirements fail, the reported reason gives incomplete coverage first, then withheld permission, then insufficient seed blocks. Withheld metrics have null `standard_error`, `ci95` and `ci_method`. For an incomplete comparison, `complete_seed_blocks` records observed complete pairs while the metric's `n_seed_blocks` is zero because no paired summary is reported. Comparison-level `eligible_for_comparison` and `withheld_reason` remain CATE-MSE compatibility aliases; inspect the metric fields when rendering root-PEHE. Eligible intervals are pointwise and unadjusted for multiplicity.

## Preserved studies

The separate [native/default panel](NATIVE_PANEL.md) withholds mean-MSE and mean-root-PEHE uncertainty for every recipe. Its prespecified exact-binomial/Holm analysis concerns a bounded strict finite-query win indicator, with ties as nonwins and one complete 72-comparison family. That endpoint does not grant the generic runner permission for mean-risk inference.

This contract applies to new generic runs under 0.3.0a3. Frozen study protocols and outputs retain their original producers and analyses, including the catalog's M11 exclusions. Historical smoke JSON and original tutorial outputs are execution diagnostics from their recorded source; their generic intervals do not establish current eligibility. The live [custom-method tutorial](../notebooks/README.md) handles withheld uncertainty. The integrated AI-generated report remains the frozen account of the scientific studies. See [verification scopes](VERIFICATION.md) for their separate evidence.
