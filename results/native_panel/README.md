# Native/default trial-CATE comparison

Read the [scientific findings](SCIENTIFIC_RESULT.md) and the [integrated AI-generated report](../../docs/REPORT.md). The retained collection and separate incomplete-collection verification finished on 2026-10-28. Of **8,448 requested fits**, **8,350 succeeded, 64 failed, 31 timed out and three remain missing**. The original complete-union contract remains unmet. All 72 joint Holm decisions and the simultaneous strict-win precision statement remain withheld; mean-risk inference was never established.

The [frozen protocol](../../docs/NATIVE_PANEL.md), [configuration](../../configs/native_panel.toml), [pre-fit source/runtime binding](../native_panel_preflight/freeze_v2.json) and [preflight](../native_panel_preflight/README.md) specify six inspected mechanisms, two RCT budgets (128/512), 64 seeds and eleven fixed recipes. Its prospective protocol is preserved byte-for-byte; this page supplies the final execution state. C08 is explicitly scored for trial-population CATE, which differs from its canonical catalog target.

## Read the compact evidence

| Artifact | Contents |
| --- | --- |
| [Verified summary](summary.json) | Original separate incomplete-collection analysis: coverage, 132 method-condition summaries, 72 fixed comparisons and withheld inference |
| [Scalar-record index](scalar_records.csv) | Six hash-bound scenario shards, containing all 8,448 requested identities, statuses and available per-seed scalar scores |
| [Method summaries](method_summaries.csv) | All eleven recipes in twelve conditions; success denominators, descriptive mean MSE and mean seedwise root-PEHE |
| [Paired comparisons](paired_comparisons.csv) | All 72 prespecified comparisons, complete/failed/missing pair counts and retained descriptive outputs |
| [Non-successful identities](non_successful_fits.csv) | All 64 failures, 31 timeouts and three unresolved attempts |
| [Verification receipt](evidence/verification_receipt.json) | Original verified-summary, manifest and terminal-inventory bindings |
| [Original incomplete-analysis source](evidence/original_incomplete_analysis.py.txt) and [recovery wrapper](evidence/original_recovery_wrapper.py.txt) | Exact preserved source bytes from the original server execution, including their recorded server paths and guards |
| [Delivery manifest](DELIVERY_MANIFEST.json) | Compact file hashes and the source identities used to export/check them |
| [Original launch](launch.json) | Pre-fit campaign source, requested resources and initial orchestration identities |

Method means using fewer than 64 successes are conditional on success. Incomplete paired means and probabilities are withheld. Complete pointwise strict-win outputs remain descriptive; their stored p-values do not release any joint decision. Mean root-PEHE averages seedwise square roots rather than taking the square root of mean MSE. The [findings](SCIENTIFIC_RESULT.md) distinguish OBS access within the matched GP from competitiveness of different complete procedures.

## Verify public arithmetic without fitting

From the repository root, in its documented Python environment:

```sh
python scripts/verify_native_panel_delivery.py \
  --config configs/native_panel.toml \
  --analysis results/native_panel \
  --expected-manifest-sha256 f2e9ec3c23cc686b48ff02a206088c92a9ec1e9ed2067923d81fc74d5749e879 \
  --output outputs/native-panel-arithmetic-check.json
```

The checker reconstructs compact summary arithmetic from published per-seed scores, checks identity/status coverage and hashes, and enforces the fixed comparison set and withheld inference. This is separate from the original server verification of full query truth and saved predictions, and it does not refit native algorithms. GitHub CI runs the public arithmetic check. The original source snapshots above identify the historical server procedure; their fixed server paths and retained-state guards are preserved, so they are not a portable replacement command for this compact check.

## What was preserved

The fitter remains `83c45c3acca5ffd316a48153647b078280d648cb`; campaign manifest SHA256 is `56003f39f113bbc84f673d0603f1fa2396030bdb64a809edde4d2473dd496c91`. Verification 731708 independently reconstructs trial targets and successful prediction scores from retained artifacts, requires exactly 8,445 terminal records and the three designated missing identities, and checks that the record inventory stays unchanged. Delivery 731709 authenticates the receipt and copies the verified summary byte-for-byte. Neither changes the original collection or makes its complete-union verifier pass.

All 64 failures occur in quadratic Integrative R on native CorNet at RCT512. All 31 outer timeouts and three missing attempts belong to forest. Its 120-second outer cap includes runtime inspection as well as fitting/prediction, so these are not isolated measurements of statistical fitting difficulty. The missing forest identities are C08 trial target/RCT512/seeds 50040 and 50042, and M0/RCT512/seed 50047. Their retained timing is consistent with interruption at the outer limit, but missing parent timeout/cleanup evidence prevents assigning a terminal status.

No failure, timeout or missing attempt is retried or replaced. Reproduction uses a new output directory and original source/runtime bindings. Resuming retained shards with unresolved attempts would risk rerunning them and is not the recovery procedure. Full arrays, predictions, environments, logs and interrupted-attempt artifacts remain on research storage; only compact reproducibility evidence is published here.

Each original fit job requests one CPU/thread, 8 GiB and six hours, with at most eight simultaneous fitting jobs. These are requested budgets, not measured whole-job peaks or a claim of equal compute across recipes. The quadratic Integrative R native deadline is 1,200 seconds. The broader [comparison design](../../docs/V03_DESIGN.md) retains additional methods, matched controls, controlled tuning and external/unseen-mechanism studies as separate future work.
