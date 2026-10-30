# v0.3 comparison design

**Status:** reviewed comparison rules and an inclusion plan, with a completed [native/default trial-CATE collection](../results/native_panel/SCIENTIFIC_RESULT.md) under its [frozen protocol](NATIVE_PANEL.md). That collection has 8,350 successes, 64 failures, 31 timeouts and three preserved missing attempts; separate incomplete-collection verification passes while all 72 joint decisions remain withheld. The separately frozen [fixed-GP source-access pilot](../results/gp_source_pilot/SCIENTIFIC_RESULT.md) is complete. The [v0.2 catalog study](../results/catalog_study/SCIENTIFIC_RESULT.md) remains fixed. Remaining methods, targets and controlled-tuning lanes in this broader design are prospective.

The decision is to expand the scientific comparison before adding replications: separate the questions being answered, include missing estimator families, and compare their permitted information and selection rules explicitly. The primary-paper cutoff is **2026-10-26**. Later code or papers need a new provenance entry. The [method inventory](V03_METHODS.md) records native targets, source revisions and implementation readiness; the [scenario classification](V03_SCENARIOS.md) covers all 31 existing presets.

## Comparison rows

Every future comparison row must bind the fields below before evaluation results are inspected. An absent field blocks execution of that row.

| Field | Required declaration |
| --- | --- |
| Question and target | Native estimand, conditioning variables, target population and its evaluation distribution |
| Scientific assumptions | Identification route, support, effect transport and method-specific modeling restrictions; deliberate violations labeled separately |
| Operational status | Implemented and checked; candidate pending checks; prospectively unsupported input; or terminal failure after execution |
| Information | OBS/RCT roles, known assignment probabilities, target-reference profile, and any acquisition/assignment authority |
| Selection and resources | Default or tuned lane, fitting/tuning labels, candidate grid, observable selector, search/CPU/RAM/time limits and refit permission |
| Comparison | Fixed control, matched backbone where a component claim is made, metric and prespecified inference family |

Scientific and operational status are independent. A runnable algorithm may be evaluated under a deliberately false assumption. Missing code is a pending candidate, not a scientific N/A. A terminal error never becomes a retrospective eligibility exclusion.

## Questions and native outputs

| Question | Target and comparison rule |
| --- | --- |
| Trial-population CATE | Estimate the conditional effect in the randomized population, under its declared evaluation law. This is the first implementation priority. |
| External-population CATE | State conditional-effect transport and trial coverage, plus any target-law covariate access. An OBS prediction is not automatically an external-population causal effect. |
| Extrapolation or effect-shift stress | Retain the adverse setting, but state which additional restrictions identify the target for each method. Score numerical predictions without presenting every method's identifying assumptions as satisfied. |
| Average, subgroup or parametric effects | Use the native ATE, predefined subgroup vector, or specified effect model. Do not broadcast a scalar ATE into an unrestricted CATE prediction. A coefficient model's native prediction map and any misspecification must be explicit. |
| Partial identification | Evaluate the native effect bounds/sensitivity conclusions and uncertainty semantics. Identification intervals and confidence intervals have different meanings. |
| Recruitment with fixed randomization | Select subjects; treatment remains randomized by the evaluator under a known rule. Compare policy changes at fixed final estimators. |
| Joint subject-and-arm design | State both controls and their assignment assumptions. This needs a separate interface and comparison from fixed-randomization recruitment. |

Native-output rows use a trial-only control for the same estimand and model class. For a scalar ATE or an explicitly constant-effect CATE comparison, include a native RCT ATE estimator with the same constant prediction map; a flexible CATE learner alone is not a matched control.

These distinctions follow the native assumptions of the [Integrative R-learner](https://proceedings.mlr.press/v177/wu22a.html), the limits of experimental validation outside support in [Fawkes et al.](https://proceedings.mlr.press/v258/fawkes25b.html), and the different target of [partial-identification fusion](https://papers.nips.cc/paper_files/paper/2025/hash/dd4e79c2228c43954cc074abf2f85a07-Abstract-Conference.html).

For example, outside trial support an observed contrast can be written as `d(x) = tau(x) + b(x)`. Changing `tau` by `h` and `b` by `-h` there preserves that contrast. A simulator can supply one truth, but observed data do not distinguish those possibilities without additional restrictions. Conversely, a violation of trial-to-target effect transport does not imply that no observed source identifies the target: canonical C08 has an unconfounded OBS route. Identification statements therefore belong to the method/assumption/comparison row, not to a universal label on the DGP.

The scenario CSV describes **existing canonical evaluation targets**, while the [native/default panel](NATIVE_PANEL.md) explicitly binds trial-law queries and trial-conditional truth. Reusing canonical `eval_tau` is invalid when it denotes another effect. C08's canonical target is `tau_target(X)`; the implemented panel condition `addendum:C08@trial_target` instead scores `tau_trial(X)`, with an independently checked selected-participant moment. PPI's native trial-CATE output and an RCT-only control share the latter target; reporting the same output against the former is a transport-stress comparison. All other method assumptions still require their own labels, and each new condition must bind the required comparison-row fields.

## Method inclusion decisions

The inventory assigns planned core candidates, diagnostic controls, native-output comparisons and separate extensions. **Planned inclusion does not claim a reproduced adapter.** Implementation priority is trial-population CATE with the Integrative R-learner and Causal-ICM added to the existing CorNet and PPI references; CVCI is a priority in its native causal-parameter lane or an explicitly equivalent constant-effect condition. PPI DR and R are separate recipes. The original paper versions and code snapshots must be aligned before admission; in particular, PPI paper v2 and the currently pinned code are distinct provenance objects.

Experimental Grounding, Elastic Integrative, power likelihood, adaptive combination and the semiparametric confounding-function approach remain named candidates. Their native outputs/model restrictions determine their eligible comparisons; availability or development effort alone does not establish a scientific exclusion. Group-effect shrinkage, bounds, incomplete-source, IV/noncompliance, survival and long-term-outcome methods need their own target/input definitions rather than silent conversion to the current continuous-outcome CATE interface.

Each new adapter must reproduce a retained author example or an independently specified small numerical example, match the native prediction/parameter definition, obey the information boundary, retain a complete failure record and record source/environment identity. For deterministic float64 native examples, fix input arrays, seed and environment before the wrapper comparison; retain a direct native call and two fresh adapter calls, checking shapes, prediction/parameter definitions, and numeric agreement at absolute tolerance `1e-10` and relative tolerance `1e-8`. Any different dtype, tolerance or stochastic agreement rule needs a method-specific justification fixed before those outputs are inspected. A discrepancy blocks that adapter and is preserved; tolerance is not relaxed after seeing it.

The independent scientific example must check a closed-form target, native prediction/parameter map, loss term or relevant subroutine using calculations outside the adapter/native fitting path, at a declared tolerance (default `1e-12` for analytic float64 examples). It also tests that hidden evaluation/test outcomes cannot affect fitting and that tuning outcomes cannot affect a controlled candidate before selection. Passing means the defined identities and information checks hold; a fitted model is not required to recover a finite-sample noisy truth exactly or beat an RCT control. Stochastic algorithms need a prespecified repetition/error criterion before admission. Keep failed attempts and all evidence. Source inspection is not executable reproduction. Reproduction and bounded resource measurements occur on a server before a full campaign is configured.

## Fair controls and model selection

Keep two separate result blocks:

- **Native/default recipes:** preserve the disclosed method settings and internal fitting/splitting rules. Reproducing a method's implementation does not imply reproducing its published table or optimizing its recipe for this benchmark.
- **Controlled tuning:** methods receive the same declared fitting/tuning outcome budget and information profile. Publish method-appropriate search spaces, search counts and resource caps. Identical epochs, wall-clock consumption or statistical capacity are not implied by equal access. Internal folds reuse outcomes already charged; they are not free additional labels.

RCT-only references should include an R-learner, a causal forest and a Bayesian causal forest alongside the retained T/DR references. A claim about the benefit of OBS information also requires the candidate's matched-capacity RCT-only backbone. A claim about selective borrowing needs an unmodified-borrowing ablation. OBS-only and naive pooling remain visible diagnostics; their recipe count does not inflate the number of distinct fusion approaches. See [Nie and Wager](https://doi.org/10.1093/biomet/asaa076), [Wager and Athey](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839), and [Hahn et al.](https://arxiv.org/abs/1706.09523).

The external selector below governs only the **controlled static trial-population lane**. Native/default internal selection remains part of its disclosed recipe. Other native-output or acquisition lanes need their own declared observable selector. In the controlled static lane, select the candidate minimizing the following known-propensity score on **independent RCT tuning rows**:

```text
L(f) = mean_i { [(Y_i - m_hat(X_i)) - (T_i - e_i) f(X_i)]^2
                 / [e_i (1 - e_i)] }.
```

Here `e_i` is the known randomized assignment probability. The common nuisance `m_hat` is fitted using the allowed RCT fitting rows only, without tuning outcomes or OBS information, and is fixed across candidates and methods within a replicate. Under randomization, conditioning on the fitting data, the expected candidate-dependent term is trial-law CATE MSE. This is a candidate-score identity, not a guarantee of precise selection with few tuning outcomes or an unbiased estimate of the winning candidate's risk. Final evaluation remains independent of selection. Without the denominator, R-loss instead weights error by `e(X)(1-e(X))`. A factual outcome loss may select nuisances but is not the common direct CATE selector.

Candidate models are fitted without tuning labels; the external selector is their only permitted consumer. Candidate predictions must be unchanged when tuning outcomes are perturbed before selection. Treatment-stratified internal folds are used where the native procedure permits them; inadequate arm counts are retained as failures, not repaired by selective seed replacement. Use the smallest declared candidate ID for an exact score tie. A candidate that errors/nonfinite-predicts is unavailable for selection and its failure is recorded; if all candidates fail, the tuned recipe fails. No unreported fallback is allowed. **No final refit on tuning labels** is permitted in this initial design. The concrete common nuisance recipe, candidate grids, tuning-row counts and resource caps are pre-run configuration fields; the decision rule above is fixed, their numerical values remain pending adapter checks. Every method uses the same completed selector and frozen nuisance recipe within a comparison.

For an external target, trial validation remains trial-law validation unless valid transport/density weighting is explicitly supplied under the common target-reference profile. Do not imply that target covariates alone establish effect transport. Very small propensities, estimated density weights and heavy tails require declared numerical handling and precision diagnostics; neither clipping nor a new robust score is chosen after reading evaluation results. M11 remains descriptive unless a separate prospective analysis is justified.

## Information, mechanisms and acquisition

Keep the existing `eval_X` prediction-only boundary. Define separate access profiles with no target-reference sample and with an independent outcome-free target-reference sample. Supply the same reference rows to all methods in an enabled profile. Target density weights must use permitted observed data, not evaluator density truth or evaluation queries. Any supplied effect-modifier subset, model basis or subgroup definition is an information privilege shared with matched controls and declared before results; evaluator truth cannot choose those inputs. Existing paper generators do not currently supply that role; enabling it requires an explicit extension and verification.

All existing 31 presets are development-inspected reference cases. The [CSV](v03_scenarios.csv) separates trial-target effect equality, OBS-target effect equality, support, practical overlap, discrepancy origin and mean-inference status. Mathematical positive support is not evidence of adequate finite-sample coverage. Native dimensions and equations remain unchanged. New target-population sampling is a newly named condition and requires its own sampler/truth checks.

Any future unseen-mechanism claim needs a separately declared block of mechanisms/coefficients and source-grouped development/evaluation splits before adapter/selection development. New coefficients within inspected generator classes test parameter transfer; they do not establish generalization to unseen mechanism classes. New sample seeds alone do not create that block. Paired variants from one paper or mechanism must not be counted as independent design discoveries. The new block and the exact campaign grid are pending; the present classification makes no unseen-mechanism claim.

The current [adaptive reference](ADAPTIVE.md) orders subjects once after a common pilot; it does not implement repeated posterior updating. A new policy must state whether it repeats updates, its batch size/outcome delay, what it sees about candidates, and its assignment authority. [R-Design](https://arxiv.org/html/2603.03785v1) chooses both subjects and arms; a subject-only adaptation must have its own name and must not be reported as an unchanged reproduction.

For acquisition effects, cross policies with common final estimators and compare estimators on the same acquired data. Share pilot/pool and fixed-budget prefixes where the policies permit them. Charge all revealed outcomes, including pilot and tuning. Pilot outcomes already exposed to a policy are not untouched policy validation. A selected trial sample can have a different covariate law from the original trial; any weighting and native iid/inference assumptions must be specified. Deterministic arm choices cannot be relabeled as 1:1 randomization or passed to the current strictly-positive-propensity interface unchanged.

## Reporting and readiness

Report condition-specific paired effects, negative transfer relative to the declared RCT control, failure/missing/N/A counts, labeled numerical coverage, and compute costs. Native uncertainty methods additionally report empirical coverage and width with the target and interval semantics stated. Real randomized holdouts support observable randomized scores, subgroup checks and policy evaluation under their assumptions; they do not reveal individual counterfactual CATE truth. Empirical covariates with simulated outcomes remain semi-synthetic evidence.

Prespecify inference families by question, target, access and selection lane, and enumerate all their comparisons before execution. Cross-family claims need their own multiplicity rule. A missing required successful pair withholds the declared family's decisions; no post-hoc shrinking of a family or score replacement. Successful-only means remain conditional on success. An explicitly defined fallback system would be a separate recipe with its own costs and results. V0.2's 21 failures, 472 withheld Holm decisions, N/A cells and heavy-tail exclusions remain unchanged.

**Design completion:** source/native-target-bound candidates; every existing preset classified; comparison-row fields and selector/data/failure/inference rules stated; independent review and a readable companion delivered.

The [bounded panel](NATIVE_PANEL.md) translates the native/default trial-CATE lane into a concrete six-mechanism, two-budget, eleven-recipe, 64-seed protocol. It binds targets, information, outcome accounting, a separate strict-win analysis and measured resource readiness. [Native admission](../results/native_adapter_admission/README.md) covers Integrative R and fixed-rho Causal-ICM; [forest admission](../results/causal_forest_admission/README.md) supplies its new RCT reference. Only the panel's matched GP contrast attributes differences to OBS access. The former catalog grid remains retained evidence with its own budgets and inference.

**Beyond that panel, still pending:** remaining native adapters and matched controls; controlled-tuning nuisance/search recipes; external/native-output/acquisition comparisons; and an independently frozen unseen-mechanism block. Each extension needs its own numerical budgets, precision and server checks. The [GP pilot](GP_SOURCE_PILOT.md) keeps its separately fixed source controls and generator. Neither its results nor the catalog authorize choosing favorable later conditions.
