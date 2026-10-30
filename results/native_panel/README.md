# Native/default trial-CATE comparison

[Findings](SCIENTIFIC_RESULT.md) · [Integrated report](../../docs/REPORT.md) · [Frozen protocol](../../docs/NATIVE_PANEL.md) · [Configuration](../../configs/native_panel.toml)

The panel covers six inspected mechanisms, RCT budgets 128/512, 64 seeds and eleven fixed recipes. Its 8,448 requested fits comprise **8,350 successes, 64 failures, 31 timeouts and three missing attempts**. Separate incomplete-collection verification reconstructs trial targets and successful prediction scores; the original complete-union contract remains unmet. All 72 joint Holm decisions and simultaneous strict-win precision remain withheld. Mean-risk inference was never established.

## Read the evidence

| Artifact | Contents |
| --- | --- |
| [Verified summary](summary.json) | Coverage, all 132 method-condition summaries, 72 fixed comparisons and inference restrictions |
| [Scalar-record index](scalar_records.csv) | Six hash-bound shards containing all 8,448 requested identities, statuses and available per-seed scores |
| [Method summaries](method_summaries.csv) | Eleven recipes in twelve conditions, with success denominators, mean MSE and mean seedwise root-PEHE |
| [Paired comparisons](paired_comparisons.csv) | All 72 comparisons with complete/failed/missing pair counts and descriptive outputs |
| [Non-successful identities](non_successful_fits.csv) | All 64 failures, 31 timeouts and three unresolved attempts |
| [Delivery manifest](DELIVERY_MANIFEST.json) and [verification receipt](evidence/verification_receipt.json) | Compact file identities and original summary/campaign/terminal-inventory bindings |
| [Producer identity](evidence/producer_identity.json) and [execution record](evidence/execution_record.json) | Frozen fitting source, independent verification, export/checker identities and measured execution scope |
| [Original analysis source](evidence/original_incomplete_analysis.py.txt) and [recovery wrapper](evidence/original_recovery_wrapper.py.txt) | Exact historical server source bytes, retaining fixed paths and guards |

Method means with fewer than 64 successes are conditional on success. Incomplete paired means and probabilities are withheld; complete pointwise strict-win outputs are descriptive only. Mean root-PEHE averages seedwise square roots. The [findings](SCIENTIFIC_RESULT.md) explain why OBS access within the matched GP and competitiveness against forest answer different questions.

## Verify public arithmetic

From the repository root, with Python 3.11+ and SciPy installed:

```sh
python scripts/verify_native_panel_delivery.py \
  --config configs/native_panel.toml \
  --analysis results/native_panel \
  --expected-manifest-sha256 f2e9ec3c23cc686b48ff02a206088c92a9ec1e9ed2067923d81fc74d5749e879 \
  --output outputs/native-panel-arithmetic-check.json
```

This CI check validates pinned file identities, requested coverage, aggregate arithmetic and inference restrictions from public scalar records. Binding those scalars to original terminal JSONs is established by the authenticated server export; the public check does not replay omitted query truth/predictions or fit estimators. Historical analyzer snapshots identify the original server procedure, not a portable replacement for this command.

## Execution and reproduction

Fitting source is `83c45c3acca5ffd316a48153647b078280d648cb`, bound by campaign manifest `56003f39f113bbc84f673d0603f1fa2396030bdb64a809edde4d2473dd496c91`. The [pre-fit source/runtime binding](../native_panel_preflight/freeze_v2.json), [preflight](../native_panel_preflight/README.md) and [launch record](launch.json) fix targets, information, resources and analysis before comparative fitting. C08 is scored for trial-population CATE, which differs from its canonical catalog target.

Verification 731708 requires exactly 8,445 terminal records and the three designated missing identities, authenticates their artifacts and independently reconstructs trial truth/successful scores. Delivery 731709 copies its checked summary byte-for-byte. The [receipt](evidence/verification_receipt.json) and [execution record](evidence/execution_record.json) retain identities, timings and the distinction from compact arithmetic checks.

All 64 failures occur in quadratic Integrative R on native CorNet at RCT512. All 31 outer timeouts and three missing attempts belong to forest. Its 120-second outer cap includes runtime inspection, fitting and prediction, so these counts do not isolate statistical fitting difficulty. Missing forest identities are C08 trial target/RCT512/seeds 50040 and 50042, and M0/RCT512/seed 50047. Their timing is consistent with outer-limit interruption, but unavailable parent timeout/cleanup evidence prevents assigning a terminal status.

For fresh fitting, use the [runtime preparation and profile/init/run commands](../../docs/NATIVE_PANEL.md#reproduce-the-frozen-execution-plan) with fitting source `83c45c3` and a new output directory. Prepare all eleven recorded method runtimes and their separately obtained upstream sources/builds. Fresh fitting generates observations from the declared seeds; replaying the retained collection instead requires its full arrays and process artifacts on research storage. The interrupted original attempts are not retried or replaced; the original `--resume` on their shards would risk rerunning them.

Fit jobs request one CPU/thread, 8 GiB and six hours, with at most eight simultaneous fitting jobs. These are requested budgets, not whole-job memory measurements or equal compute across recipes. The quadratic Integrative R native deadline is 1,200 seconds. Other methods, matched controls, targets and controlled tuning have their own prerequisites in the [comparison design](../../docs/V03_DESIGN.md).
