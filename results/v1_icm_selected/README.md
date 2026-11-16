# Selected-ρ addendum packet

[Scientific result](SCIENTIFIC_RESULT.md) · [plan](../../docs/V1_ICM_SELECTED.md) · configurations: [`core`](../../configs/v1_icm_selected_core.toml), [`stress`](../../configs/v1_icm_selected_stress.toml)

The addendum has two blocks, each run from its own freeze commit's snapshot on GSDS in wave 2's `gp` environment (Python 3.11, GPy 1.13.2, the authors' Causal-ICM code at `0f0c6ca`):

| Block | Freeze | Grid | Cells |
| --- | --- | --- | --- |
| `core` | `7758408` | Wave 2's 124 conditions, seeds 70000–70063 | 39,680 |
| `stress` | `5e48866` | The stress addendum's eight dials × four levels × two trial sizes, seeds 80000–80063 | 20,480 |

The package list is in `evidence/environments/gp-python.txt`.

| File | Contents |
| --- | --- |
| `<block>/replicate_scores/` | Every cell, one file per preset or dial: status, CATE MSE and its root, failure stage and type, the candidate `rct_selected` chose and the ρ `causal_icm_selected` chose |
| `<block>/rho_selection.csv` | Per condition, how often each ρ was chosen, the mean choice, and selection failures and weight-convergence counts |
| `<block>/condition_scores.csv` | Per-condition mean error and success counts for every recipe |
| `<block>/contrasts.csv` | Every declared contrast: paired mean, interval, p-value, Holm-adjusted p-value and verdict |
| `stress/readout.csv` | Each family's verdicts from level 0 to 3 for each dial and trial size, with the declared state |
| `stress/dose_response.csv` | Each recipe's error relative to `rct_selected` by dial and level: ratio of means and geometric mean ratio with a pointwise 95% interval |
| `<block>/headline.md` | Verdict counts per family, the identity checks and the distribution of chosen ρ |
| `<block>/identity_check.json` | Reproduction of the earlier wave's scores for the refitted recipes, selection at 0.5 against the fixed recipe, and the RCT-only identity |
| `<block>/analysis.json` | Coverage and family summaries |
| `<block>/checks/` | Independent arithmetic and the reproducibility check; the stress check ran in 16 parts, kept in `reproduce_parts/` |
| `core/void_r01/` | The first core analysis, void because it read no wave-2 scores; kept with its arithmetic |
| `evidence/` | Job scripts, job record, rerun lists, shard manifests, data check, environment package list, the split-check smoke part and the split script |
| `DELIVERY_MANIFEST.json` | Freeze commits, code archive hashes, job IDs, notes and the SHA-256 of each analysis file as exported, including the unsplit per-seed scores |

The analyzer writes each block's per-seed scores as one file. The repository keeps files under 1 MiB, so they were split at publication. `python results/v1_icm_selected/evidence/split_replicate_scores.py --check` rejoins each block's files and confirms the export's SHA-256 byte for byte. Full records, run manifests, logs and moved-aside attempts remain on GSDS under `obsrct-causal-icm-selected-20261115-r01`.

## Job record

- **Data check.** Job 834151.
- **Core block.**
  - Main run: shards written by job 834496, arrays 834497 and 834498; 14 job-limit reruns in 837388.
  - Analysis: job 838098 read no wave-2 scores and is void (`core/void_r01/`); job 838129 is the final analysis.
  - Reproducibility: job 838099.
- **Stress block.**
  - Main run: shards written by job 838184; the `obs_size` dial ran as array 838185 and the other seven as 838186.
  - 16 `obs_size` tasks, all on one node, ended at the time limit. 14 had not finished and were rerun once in 842196; all 14 completed. The other two had written every record and their status file before the scheduler ended them, so they stand, as the rerun script requires.
  - Analysis: job 845339.
  - Reproducibility: array 845340, one condition per task, merged by job 845341.
- Every entry, with its time and outcome, is in `evidence/JOBS.txt`; the plan's [revisions and errata](../../docs/V1_ICM_SELECTED.md#revisions-and-errata) give each job-script change.

## Reproduce

Install the `gp` extras at the block's freeze commit, as described in the main README. Write the shards with [`make_icm_selected_configs.py`](../../scripts/make_icm_selected_configs.py) and run each with `python -m obs_rct_benchmark.cli run --config <shard> --output <dir>`; `evidence/jobs/` holds the exact driver and job scripts used. Then, from the repository root with the package and its scripts importable:

```sh
export PYTHONPATH=src:scripts
python scripts/analyze_icm_selected.py --config configs/v1_icm_selected_<block>.toml \
  --runs <the block's shard directories> --wave2 results/<v1_wave2|stress_wave2> --output analysis
python scripts/verify_icm_selected.py arithmetic --config configs/v1_icm_selected_<block>.toml \
  --analysis analysis --runs <...> --wave2 results/<v1_wave2|stress_wave2> --output arithmetic.json
python scripts/verify_icm_selected.py reproduce --config configs/v1_icm_selected_<block>.toml \
  --runs <...> --output reproduce.json
```

The core block reads the wave-2 packet and the stress block reads the stress addendum's. The analyzer refuses a missing, undeclared or duplicate shard, a cell without a terminal record, a record whose identity does not match its shard, or shards whose source or environment differ.
