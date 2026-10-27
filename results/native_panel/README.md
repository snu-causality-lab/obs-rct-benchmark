# Native/default trial-CATE comparison

**Running on GSDS, started 2026-10-27.** The complete frozen comparison requests **8,448 fits**: six mechanisms × two RCT budgets × 64 seeds × eleven fixed recipes. Initial shard array **725161** runs shards 0–7. Controller **725193** is registered after that complete array, with authenticated continuation through all 128 shards. This page records launch state, not a live numerical-results feed.

[Protocol](../../docs/NATIVE_PANEL.md) · [Configuration](../../configs/native_panel.toml) · [Source/runtime freeze](../native_panel_preflight/freeze_v2.json) · [Launch record](launch.json) · [Preflight evidence and known limitation](../native_panel_preflight/README.md).

Each fitting job requests one CPU/thread, 8 GiB and six hours. At most eight fitting jobs run concurrently. Each shard owns six cases and 66 terminal fits. After each eight-shard array terminates, an allocated controller authenticates its complete records before submitting the next eight. Valid estimator failures/timeouts are preserved and do not stop later waves. Missing records, unresolved locks, source/runtime drift or an ambiguous submission stop automatic advancement; there is no blind retry or duplicate submission.

After all 128 shards, separate allocated jobs perform the frozen full-collection verification and independent analysis. Analysis starts only after verification succeeds. Arrays, predictions, native artifacts and full logs remain on GSDS; verified compact summaries will be published here after completion. No comparative ranking or completed-study inference is reported at launch.

The scientific configuration, native recipes, targets, seeds, outcome budgets, deadlines and 72-comparison family remain frozen. In particular, high-dimensional quadratic Integrative R retains its native 1,200-second limit. Any required failed comparison withholds all 72 secondary decisions. Raw risks remain descriptive under the protocol. Orchestration runs outside the frozen scientific source, with script hashes in the launch record.

The existing integrated AI-generated report covers four completed studies. This running campaign will be incorporated into that same manuscript after completed verification and analysis.
