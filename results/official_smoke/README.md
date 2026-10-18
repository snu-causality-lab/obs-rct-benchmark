# Benchmark run

Scope: smoke/development-inspected; reduced CorNet optimization (5+5 epochs), no scientific ranking. This is a development example, not a method-ranking study.

Status: complete. Successful cells: 12/12; failed: 0; missing: 0.

`records.jsonl` stores one terminal record per scenario/seed/method. `run_manifest.json` stores the normalized configuration, source hashes, and environment. Requested official adapters also record external implementation hashes; custom method provenance strings are author-supplied identifiers, not automatic source checks. `summary.json` stores coverage, descriptive metrics, and paired comparisons.

| Method | Mean CATE MSE | Mean root-PEHE | Successful / expected |
| --- | ---: | ---: | ---: |
| `rct_t_ridge` | 0.575452 | 0.745863 | 4 / 4 |
| `datafusionppi` | 0.835143 | 0.856339 | 4 / 4 |
| `cornet_official` | 0.863802 | 0.818122 | 4 / 4 |

Table means describe successful cells only. Incomplete coverage prevents inferential comparisons and can make descriptive means misleading.

CATE MSE is the mean squared CATE error over evaluation rows. Root-PEHE is its square root for a replicate. Mean root-PEHE and root mean MSE are reported separately and generally differ.

Paired differences are method minus reference. For the overall comparison, scenario differences are averaged within each independent seed block. 95% Student t intervals use a conservative tabulated critical value; fewer than two seed blocks produces no interval. Two seeds are a smoke check, not sufficient evidence for a scientific comparison. Failed or missing cells withhold inferential comparisons; no failure is assigned a winning score.

Resume accepts only matching configuration and source hashes, rejects duplicates and partial final JSONL lines, and never retries terminal failed records. After a hard process kill, inspect the output and confirm no writer remains before manually removing a stale `.run.lock`.
