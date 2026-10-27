# RCT-only honest causal-forest admission

The optional `rct_causal_forest` recipe is admitted for **native/default RCT-only CATE prediction** in package 0.3.0a4. It calls official `grf` 2.6.1 at the [locked revision](https://github.com/grf-labs/grf/tree/5bee99b51471f76cb2d63acbc8a9b0ffec408ba0), with 2,000 causal trees, the native 500-tree outcome nuisance, known randomized propensity and one thread. [Recipe and reproduction](../../docs/CAUSAL_FOREST.md) · [complete native evidence](summary.json).

The fixed observed fixture has 128 trial rows, five covariates and eight prediction queries. No causal truth is used for fitting or selected on. One direct native call and two fresh-process adapter calls have maximum prediction difference **0** under prespecified absolute/relative tolerances 1e-10/1e-8. The second adapter receives a typed view whose excluded OBS/tuning/target-reference fields raise on access. Their equality verifies the intended RCT-only entry; fitting consumes 128 RCT outcomes and zero OBS/tuning outcomes. Appending two distant queries leaves the original eight predictions exactly unchanged.

Separately written scalar arithmetic reconstructs the native weighted residual regression **with a local intercept**. Its maximum prediction difference is **2.22e-15**, below the predeclared absolute 1e-12 tolerance. A hand oracle gives 8/3; an uncentered residual ratio gives 3.2 and raw weighted arm means give 43/12. This checks the prediction map conditional on upstream neighborhoods and nuisances, rather than independently verifying forest training, causal accuracy or confidence coverage.

Two bounded default-recipe profiles use five covariates and 2,048 queries:

| RCT fitting rows | Complete adapter wall time | Child CPU time | Largest child-process RSS |
| ---: | ---: | ---: | ---: |
|128|7.101s|2.601s|267,928KiB|
|512|8.305s|3.586s|295,756KiB|

Wall times include source/build authentication, serialization and R startup. RSS is the largest observed child-process mark, not simultaneous whole-job peak. These are single-fixture costs, not campaign forecasts. Six native procedure calls comprise the fidelity repeats, query check and two profiles; each includes its native nuisance fit.

The GSDS cohort passes **60 focused and 475 full-suite tests**, with CorNet and the R namespace probe configured. The source/build and installed `grf` bytes are bound; the authentication scope explicitly excludes a proof of compiler correspondence or whole-runtime/transitive-library authentication. Exact source, environment and raw logs remain retained on GSDS. The public [build recorder](../../scripts/record_causal_forest_build.py) generates and validates a manifest for a separately completed installation.

This is implementation admission, not a comparative performance result, source-paper table reproduction or proof that causal forest is a stronger baseline on this catalog. Native uncertainty is not requested; generic metric uncertainty still requires [explicit assumptions](../../docs/INFERENCE.md). Frozen studies and their protocols/results remain unchanged. A separately frozen comparison must establish the new candidate's effect on benchmark conclusions.
