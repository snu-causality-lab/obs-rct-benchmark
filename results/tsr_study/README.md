# TSR static study packet

[Scientific result](SCIENTIFIC_RESULT.md) · [protocol](../../docs/TSR_STUDY.md) · [configuration](../../configs/tsr_study.toml)

The producer is commit `5d09194` (package version string 0.3.0a4). It was run on GSDS RTX 3090 nodes with Python 3.12.3, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.5.2, torch 2.5.1+cu121 and tabpfn 2.0.9. The TabPFN v2 regressor checkpoint has SHA-256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`.

| File | Contents |
| --- | --- |
| `replicate_scores/` | All 39,680 cells, one file per preset. Each row has status, CATE MSE and root-PEHE, plus the Stage 1 device and GP convergence/iterations where they apply |
| `condition_scores.csv` | Per-condition mean MSE for the five recipes |
| `primary_contrasts.csv` | Families A and B: paired means, intervals, p-values, Holm-adjusted p-values and decisions |
| `descriptive_contrasts.csv` | TSR minus OBS-only, without inference |
| `budget_counts.md` | Descriptive counts against the matched GP and the catalog recipes |
| `identity_check.json` | Reproduction of the catalog's published per-seed scores |
| `analysis.json` | Coverage, failures and family summaries |
| `evidence/` | Producer identity, environment freeze, GPy check, job record, shard checksums and the job scripts used |
| `DELIVERY_MANIFEST.json` | SHA-256 of the exported files |

The export wrote the per-seed scores as one `replicate_scores.csv` of about 4 MB. The repository keeps files under 1 MiB, so it was split by preset at publication. `python results/tsr_study/evidence/split_replicate_scores.py --check` rejoins the files and confirms the result matches the export's SHA-256 in `DELIVERY_MANIFEST.json` byte for byte.

Full per-shard records, run manifests and the stopped r01 run remain on GSDS under `obsrct-tsr-study-20261113-r01`.

## Reproduce

Install `.[baselines,analysis,tsr]` at the producer commit and point `OBS_RCT_TABPFN_MODEL_PATH` to the pinned checkpoint. Then:

```sh
python scripts/make_reference_configs.py --config configs/tsr_study.toml --output shards --shards 64
python -m obs_rct_benchmark.cli run --config shards/shard-00.toml --output outputs/shard-00   # repeat for each shard, on a GPU
python scripts/analyze_tsr_study.py --config configs/tsr_study.toml --runs outputs/shard-* \
    --catalog results/catalog_study --output tsr-analysis
python scripts/verify_tsr_gp.py --config configs/tsr_study.toml --output tsr_gp_check.json   # needs GPy
```

The analysis refuses shards whose settings, seeds, record identities, source, environment or Stage 1 device differ from the declared study. Run at most four shards per 24 GB GPU: twelve per GPU caused out-of-memory failures in the first run.
