# AI-generated technical report

[Read the PDF](https://github.com/snu-causality-lab/obs-rct-benchmark/releases/download/report-2026-11-17/obsrct_benchmark_icml.pdf) · [Report snapshot](https://github.com/snu-causality-lab/obs-rct-benchmark/releases/tag/report-2026-11-17) · [Document and evidence identities](report_provenance.json)

This **AI-generated research draft** connects the benchmark's assumptions, native implementation checks and every study in this repository in one manuscript: 9 main-text pages, 66 in total. It is not peer reviewed, and its human author list is unassigned. Earlier versions, including the [October 2026 report](https://github.com/snu-causality-lab/obs-rct-benchmark/releases/tag/report-2026-10-30-r2), stay available unchanged.

The report separates three questions: does a procedure improve an OBS starting prediction, does OBS improve a particular learner, and does that learner beat an RCT-only alternative an analyst could actually run?

## The v1 comparison program (Section 6, Appendix E)

Every implemented method is compared, condition by condition, with `rct_selected`: an RCT-only learner chosen from the trial's own data by out-of-fold R-loss. Each comparison is a paired t-test over 64 fresh seeds, Holm-corrected within one recipe; a failed condition is untested and voids nothing else. Each wave's plan, analyzer and checker were committed before its first fit.

| Study | Executed scope | Readout |
| --- | --- | --- |
| [v1 core study](../results/v1_core/SCIENTIFIC_RESULT.md) | 12 recipes × 124 conditions × 64 seeds; 94,962 of 94,976 fits | CorNet with raw inputs better in 88 of 120 conditions and worse in 7; DataFusionPPI better in 14 of 116 and worse in 43 |
| [Wave 2](../results/v1_wave2/SCIENTIFIC_RESULT.md) | Integrative R, causal forest, Causal-ICM and TSR in four environments; 182,336 of 182,528 fits | Linear Integrative R better in 62 and worse in 8; OBS helps Causal-ICM's GP in 109 of 120; TSR worse in 80 to 86; the RCT-only forest beats the reference in 38 of 60 |
| [Selected-ρ addendum](../results/v1_icm_selected/SCIENTIFIC_RESULT.md) | Causal-ICM with the paper's borrowing selection, core grid and stress dials; 60,160 of 60,160 fits | ρ ≥ 0.9 in 82% of core and 92% of stress fits. Against ρ = 0.5: 32 better and 17 worse of 120 core conditions (16 and 14 with the smaller trial), 35 and 2 of 64 stress conditions |
| [Stress study 1](../results/stress_cornet/SCIENTIFIC_RESULT.md) and [addendum](../results/stress_wave2/SCIENTIFIC_RESULT.md) | Eight dials from CorNet's stated assumptions × four levels × two trial sizes × 64 seeds; 143,360 fits | A hidden effect modifier undoes CorNet; linear Integrative R holds (46 better, 1 worse of 64), but a shifted target reverses it with the smaller trial |
| [TSR study](../results/tsr_study/SCIENTIFIC_RESULT.md) | Reimplemented TSR and controls × 124 conditions × 64 seeds | 4 wins and 49 losses against its own RCT-only GP |
| [Paper-reported values](../results/reported_values/README.md) | CorNet Figure 3 and Integrative R Tables 1–2, rerun with the authors' code | The adapters agree with the released code, CorNet's once input standardization is off; the released code does not reproduce CorNet's published values from 236 OBS rows upward |

The reference is realistic, not strong: in the median preset it has 1.3 times the error of its best candidate. Beating it is competitiveness at equal trial labels; only the matched Causal-ICM and TSR contrasts isolate what OBS adds.

## Earlier studies (Section 7, Appendices F–I)

The five October studies keep their numbers, assets and inference: the [full catalog](../results/catalog_study/SCIENTIFIC_RESULT.md), the [controlled GP experiment](../results/gp_source_pilot/SCIENTIFIC_RESULT.md), the [native/default panel](../results/native_panel/SCIENTIFIC_RESULT.md), the [matched-M2 comparison](../results/reference_study/SCIENTIFIC_RESULT.md) and [charged recruitment](../results/adaptive_reference/SCIENTIFIC_RESULT.md). The catalog withheld all 472 joint decisions after 21 failures; the v1 core study reruns its grid on fresh seeds.

## Limits

All mechanisms are synthetic designs inspected during development; fresh seeds replicate sampling within them, not new mechanisms or real data. Methods run at their authors' default settings, untuned. The stress dials target CorNet's assumptions only. The selected-ρ addendum reuses seeds whose fixed-ρ results were already read. One AI agent designed, ran and analyzed the v1 studies; separate AI reviewers checked the plans and part of the code, and no person has reproduced them. Each [result packet](../results/README.md) is the authoritative record of its study.
