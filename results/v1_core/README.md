# v1 core study packet

[Scientific result](SCIENTIFIC_RESULT.md) · [plan](../../docs/V1_CORE_STUDY.md) · [configuration](../../configs/v1_core_study.toml)

The producer is the freeze commit `a10fe6d` (package version 0.3.0a6). It ran on GSDS CPU nodes with Python 3.12.3, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.5.2, torch 2.2.2+cpu and the pinned official CorNet and DataFusionPPI sources. The full environment freeze is in `evidence/`.

| File | Contents |
| --- | --- |
| `replicate_scores/` | All 94,976 cells, one file per preset: status, CATE MSE, root-PEHE, failure stage and type, and for `rct_selected` the chosen candidate and largest weight |
| `condition_scores.csv` | Per-condition mean, median, quartiles and success counts for the twelve recipes |
| `primary_contrasts.csv` | Every declared contrast against `rct_selected`: paired mean, interval, p-value, Holm-adjusted p-value, verdict and failure-as-loss win count |
| `descriptive_contrasts.csv` | Ratios of mean error to `rct_selected`, the catalog reference and the best of four, without inference |
| `selection.csv` | How often `rct_selected` chose each candidate, its largest weight and its regret against the best of four |
| `headline.md` | Verdict counts per recipe, transport conditions separate |
| `identity_check.json` | OBS-size identity of RCT-only records and the ridge-refit identity |
| `analysis.json` | Coverage, the 14 failures and family summaries |
| `checks/` | Independent arithmetic, selector recomputation and reproduction reports |
| `evidence/` | Job scripts, job record, environment freeze, shard manifest and checksums, and the split script |
| `DELIVERY_MANIFEST.json` | Freeze commit, code archive hash, job IDs and the SHA-256 of every exported file |

The analyzer wrote the per-seed scores as one 9.6 MB `replicate_scores.csv`. The repository keeps files under 1 MiB, so it was split by preset at publication. `python results/v1_core/evidence/split_replicate_scores.py --check` rejoins the files and confirms the result matches the exported SHA-256 byte for byte.

Full per-shard records, run manifests and logs remain on GSDS under `obsrct-v1-core-20261114-r01`.

## Job record

Job 819601 ran the analysis, the selector recomputation and the reproduction check. All three passed. Its arithmetic check failed only because the job script did not pass `--config` and `--runs`. Job 819966 reran that check with the correct arguments and found no mismatch. `evidence/jobs/analysis.sbatch` is the corrected script; the one that ran in 819601 lacked those two arguments on its arithmetic line. The rerun script is `evidence/jobs/arithmetic.sbatch`.

## Reproduce

Install `.[baselines,analysis,cornet,datafusionppi]` at the freeze commit and set `CORNET_SOURCE` to the official CorNet checkout, as described in the main README. Then:

```sh
python scripts/make_v1_core_configs.py --config configs/v1_core_study.toml --output shards
python -m obs_rct_benchmark.cli run --config shards/group-common-seed-70000.toml --output outputs/group-common-seed-70000
# ... repeat for all 128 shards listed in shards/study_manifest.json
python scripts/analyze_v1_core.py --config configs/v1_core_study.toml --runs outputs/group-* --output analysis
python scripts/verify_v1_core.py arithmetic --analysis analysis --config configs/v1_core_study.toml --runs outputs/group-* --output arithmetic.json
python scripts/verify_v1_core.py selector --config configs/v1_core_study.toml --runs outputs/group-* --output selector.json
python scripts/verify_v1_core.py reproduce --config configs/v1_core_study.toml --runs outputs/group-* --output reproduce.json
```

Each shard takes 10–15 minutes on one CPU. The analyzer refuses a missing shard, a cell without a terminal record, or shards whose settings, record identities, source or environment differ from the declared study. `python scripts/render_v1_core.py` redraws the README figure from this packet.
