# CorNet Figure 3 replay: execution record

**Status: complete.** Run on 2026-11-08. The fixed question, arms and readouts are in [PROTOCOL.md](PROTOCOL.md). The [results](#results) and [interpretation](#interpretation) are below.

## Where it runs

All paths are on GSDS under `/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02`, with permissions `0700`.

| Path | Content |
| --- | --- |
| `code/obs-rct-benchmark-173af2a/` | `git archive` of the public benchmark at `173af2a`: `src/` and `pyproject.toml` only |
| `drivers/`, `stubs/`, `jobs/` | Copies of this folder's scripts, placeholder modules and Slurm files |
| `inputs/` | `PROTOCOL.md` and `reported_values.csv` as staged |
| `SHA256SUMS` | Hashes of the 31 staged files, verified on the server after transfer |
| `results/author/seed-NN/`, `results/benchmark/mech-NNNN/`, `results/analysis/` | Arm A, Arm B and aggregated outputs |
| `logs/` | One log per job or array task |

Two existing assets are reused read-only.
- The interpreter is `obsrct-public-upgrade-20261025-r01/environment-v1/bin/python`: Python 3.12.3 with torch 2.2.2+cpu, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.5.2 and matplotlib 3.10.8. It has no pandas or seaborn, hence the placeholders.
- The CorNet checkout is `obsrct-public-upgrade-20261025-r01/upstreams/CorNet-official-v1`, at `38b9c37` with no tracked changes. Jobs set `PYTHONDONTWRITEBYTECODE=1` so that they write nothing into it.

## First attempt

The first bundle in `obsrct-cornet-replay-20261108-r01` failed its preflight, job 770268. scikit-learn imports pandas when it is installed and reads its version. The placeholder directory was already importable, so scikit-learn found the placeholder and raised an error. The same preflight verified the CorNet source and completed one tiny benchmark fit per setting. Its dependents, 770269 to 770271, could never start and were cancelled. No replay output existed.

The fix loads every library the script imports before the placeholders become importable. Each Arm A task now records which modules came from the placeholders and whether scikit-learn saw pandas. The `-r01` folder is kept unchanged as the record of that attempt.

## Source identities

| File | SHA-256 |
| --- | --- |
| CorNet `simulation_studies/sim_n_conf.py` | `9d99f06ae92650114dc8ad314a93446703491b4d4502d4af8922eba993e581c3` |
| CorNet `simulation_studies/simdata.py` | `07576aabc15e490fa60595b1cdf3c3631eb60dd5344c14a5d3cbde06a9fa4541` |
| CorNet `model/models.py` | `31ff1086ca97fe729cff320338da3e3dd619bb879668e46de94c5bea4366826a` |
| Benchmark `src/obs_rct_benchmark/cornet.py` at `173af2a` | `3ccd76509d2965ea8451da9e279a7f3c8780873c5c24625f380d3c9b719b83bf` |
| Benchmark `src/obs_rct_benchmark/paper_dgp.py` at `173af2a` | `77bb899367e6ce8d12ee127b4ff6deb833215e90a707bb0f35c9aacf55f6a3f2` |
| `drivers/author_replay.py`, stored here as `author_replay.py.txt` | `8e959467551b2f19a1c019b521ff5987058acd7c7d9d793a727867eef1a73e84` |
| `drivers/benchmark_grid.py` | `05c54de6d31b8c91429f6c165b6d5bacfd9998bf31e70f459314530fdd5e6a42` |
| `drivers/preflight.py` | `03285156fc6cb931583f93797efec397e6abeea915fb3a79fd94dd37aafe399f` |
| `drivers/aggregate.py` | `9b8eba5a6e918caa7bd055553e02720548cd2a6680e6bd6fb27c2a07856ae26d` |
| `PROTOCOL.md` as staged | `b972a8ee1e14426f0a72ca3090d4241a4d7f049087fe7ec4a3dd254120be5e81` |

`author_replay.py` imports CorNet's own modules, which this repository's release check does not allow in Python files, so the released copy has a `.txt` suffix. Its bytes are those executed.

## Jobs

| Job | Slurm ID | Request | Runs after |
| --- | --- | --- | --- |
| Preflight | 770293 | 1 CPU, 4 GiB, 15 min | none |
| Arm A, released code, seeds 0 to 9 | 770294, array 0-9 | 1 CPU, 4 GiB, 1 h per task | preflight succeeds |
| Arm B, benchmark, mechanism seeds 2025 to 2034 | 770295, array 0-9 | 1 CPU, 4 GiB, 2 h per task | preflight succeeds |
| Aggregation | 770296 | 1 CPU, 4 GiB, 15 min | both arrays succeed |

All jobs run on partition `P2` without a GPU. Each fit is a small full-batch network on at most 512 rows, predicted on 10,000 rows. Earlier public driver tests in the same environment peaked at 219 MB of batch-step memory under a 4 GiB request. The per-fit time at 10,000 evaluation rows was not measured beforehand, so the time limits leave wide margins: Arm A runs 330 fits per task and Arm B runs 640.

## Results

All 22 allocations completed with exit code 0. Arm A tasks took 3 min 43 s to 3 min 59 s each. Arm B tasks took 7 min 20 s to 7 min 41 s, except tasks 1 and 2, which took 15 min 28 s and 18 min 12 s on the same node; the cause was not examined. Batch-step MaxRSS was 221 to 294 MB. All 6,400 Arm B fits succeeded.

The ten Arm A networks have mean squared OBS-RCT gaps of 3.87 to 4.14 on their last test draw, matching the design value of 4. The key readouts were recomputed from the raw outputs without the aggregation script, and they agree.

Compact outputs are in [`results/`](results/). They are the aggregated tables, `readouts.json`, and per-task records with per-replication MSEs. Each Arm A task saved its MSEs as `replay.npz`; this copy holds them as `replay_mse.csv`, and [`npz_conversion.json`](results/runs/author/npz_conversion.json) records both files' hashes and an exact round trip. The aggregation read the `.npz` files on GSDS. The job IDs are in `results/SUBMITTED.txt`. All values are mean sqrt PEHE.

**R1, reproduction by the released code.** The ten networks come from ten seeds of the authors' script.

| Series | Paper inside the 10-network range | Paper, OBS 500 and RCT 100 | Released code: mean, then range |
| --- | ---: | ---: | --- |
| tau_CorNet | 10 of 30 points | 0.646 | 1.275; 0.748 to 2.057 |
| tau_Conf | 30 of 30 | 2.093 | 2.103; 1.930 to 2.191 |
| tau_Unc | 30 of 30 | 1.009 | 1.100; 0.906 to 1.449 |

The ten CorNet points inside the range all have small OBS samples: 25 to 183 rows at RCT 25, and 25 to 130 rows at RCT 50 and 100. At every larger OBS size the paper's value is below the best of the ten draws.

**R2, translation and adapter.** Without standardization, the benchmark lies within the larger across-network SD of the released code at all 30 points. On average it is 0.119 higher. At OBS 500 and RCT 100, the released code gives 1.275 with SD 0.437, and the benchmark gives 1.317 with SD 0.541.

**R3, standardization.** These are paired differences, standardized minus unstandardized, averaged within each mechanism.

| Cell | Mean difference | SD across mechanisms | Mechanisms with a positive difference |
| --- | ---: | ---: | ---: |
| OBS 512, RCT 128 | +0.258 | 0.173 | 10 of 10 |
| OBS 512, RCT 512 | +0.260 | 0.177 | 10 of 10 |
| OBS 500, RCT 25 | +0.243 | 0.183 | 10 of 10 |
| OBS 500, RCT 50 | +0.241 | 0.187 | 10 of 10 |
| OBS 500, RCT 100 | +0.267 | 0.179 | 10 of 10 |

**Bridge to the native panel.**

| Cell | Native panel | Mechanism 2025, standardized | Mechanism 2025, unstandardized | 10 mechanisms, unstandardized: mean, then range |
| --- | ---: | ---: | ---: | --- |
| OBS 512, RCT 128 | 1.246 | 1.259 | 0.810 | 1.309; 0.804 to 2.007 |
| OBS 512, RCT 512 | 1.216 | 1.237 | 0.787 | 1.282; 0.787 to 1.978 |

## Interpretation

- **Standardization explains most of the native-panel gap.** The benchmark's default OBS-based standardization raises CorNet's error by about 0.25 in every mechanism. At mechanism 2025, the network used in all earlier benchmark studies, it moves the error from 0.81 to 1.26. The standardized run reproduces the native-panel value, so that earlier number is correct for the recipe as configured.
- **The DGP translation and adapter are faithful.** At matched settings and without standardization, the benchmark agrees with the authors' code within the spread across networks.
- **The released code reproduces Figure 3's baselines but not its CorNet curve.** Beyond OBS 183, the published CorNet values are better than all ten network draws. Two explanations remain: one favorable network, or a code difference that was never released. Under exchangeable draws, a favorable network would beat all ten with probability 1/11. The replay cannot tell the explanations apart. Figure 6 is already known not to come from the released script.
- **CorNet depends strongly on the network draw.** At OBS 500 and RCT 100, its error ranges from 0.75 to 2.06 across the ten draws of the authors' code, while the baselines vary far less. Mechanism 2025 is one of the easiest networks for CorNet: its unstandardized error is 0.81, against a 10-mechanism mean of 1.31. A single mechanism seed therefore misrepresents this DGP, and dashboard entries for it should span several mechanism seeds.
