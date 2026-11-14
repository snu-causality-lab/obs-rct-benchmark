# Stress study 1 packet

[Scientific result](SCIENTIFIC_RESULT.md) · [plan](../../docs/V1_STRESS_CORNET.md) · [configuration](../../configs/v1_stress_cornet.toml)

The producer is the freeze commit `c4bd9a1` (package version 0.3.0a7). It ran on GSDS CPU nodes in the wave-1 environment: Python 3.12.3, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.5.2, torch 2.2.2+cpu and the pinned official CorNet and DataFusionPPI sources. The freeze is in `evidence/`.

| File | Contents |
| --- | --- |
| `replicate_scores/` | All 49,152 cells, one file per dial: status, CATE MSE, and for `rct_selected` the chosen candidate |
| `condition_scores.csv` | Per-condition mean MSE for the twelve recipes |
| `contrasts.csv` | Every declared contrast against `rct_selected`: paired mean, interval, p-value, Holm-adjusted p-value and verdict |
| `readout.csv` | CorNet's verdicts from level 0 to 3 for each dial and trial size, with the declared state |
| `dose_response.csv` | Every recipe's error relative to `rct_selected` by dial and level: ratio of means and geometric mean ratio with a pointwise 95% interval |
| `analysis.json` | Coverage and verdict counts, extrapolation conditions separate |
| `checks/` | Independent data regeneration, arithmetic and reproduction reports |
| `evidence/` | Job scripts, job record, environment freeze, shard manifest and checksums, and the split script |
| `DELIVERY_MANIFEST.json` | Freeze commit, code archive hash, job IDs and the SHA-256 of every exported file |

The analyzer wrote the per-seed scores as one 3.8 MB file. The repository keeps files under 1 MiB, so it was split by dial at publication. `python results/stress_cornet/evidence/split_replicate_scores.py --check` rejoins the files and confirms the export's SHA-256 byte for byte. Full records and logs remain on GSDS under `obsrct-v1-stress-20261114-r01`.

## Reproduce

Install `.[baselines,analysis,cornet,datafusionppi]` at the freeze commit and set `CORNET_SOURCE`. Then:

```sh
python scripts/make_stress_configs.py --config configs/v1_stress_cornet.toml --output shards
python -m obs_rct_benchmark.cli run --config shards/stress-seed-80000.toml --output outputs/stress-seed-80000
# ... repeat for all 64 seeds
python scripts/analyze_stress.py --config configs/v1_stress_cornet.toml --runs outputs/stress-seed-* --output analysis
python scripts/verify_stress.py targets --config configs/v1_stress_cornet.toml --output targets.json
python scripts/verify_stress.py arithmetic --config configs/v1_stress_cornet.toml --analysis analysis --runs outputs/stress-seed-* --output arithmetic.json
python scripts/verify_stress.py reproduce --config configs/v1_stress_cornet.toml --runs outputs/stress-seed-* --output reproduce.json
python scripts/render_stress.py
```

Each shard takes under ten minutes on one CPU.
