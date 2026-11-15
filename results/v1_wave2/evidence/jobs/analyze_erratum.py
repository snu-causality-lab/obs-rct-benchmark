"""Erratum runner for the wave-2 and stress-addendum analyzers (2026-11-15).

The frozen analyzers compare each run's whole recorded `source` block with the analysis
process's, `version` included. `version` comes from importlib.metadata, which reads
whatever distribution is installed in the environment. Every job imports the code snapshot
from PYTHONPATH, and the four group environments hold different stale installs (0.2.0a1,
0.3.0a2, none, 0.3.0a4). So no single analysis environment passes all four groups, although
every run's package files are byte-identical to the analysis package's.

This runner first checks, for every run, that the recorded source equals the analysis
package's in every field except `version`: each file's SHA-256 and the aggregate. It stops on
any difference and prints each directory's version string. It then runs the frozen analyzer
unchanged, with the existing `check_source=False` switch on `collect`. Every other check in
`collect` runs as frozen: declared shards, one source and environment per group,
external-method provenance and coverage.

Usage: python analyze_erratum.py <analyzer.py> <the analyzer's own arguments>
"""
import functools
import json
import runpy
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    script, args = Path(sys.argv[1]), sys.argv[2:]
    from obs_rct_benchmark.runner import source_provenance

    def code(source: dict) -> dict:
        return {k: v for k, v in source.items() if k != "version"}

    ref = code(source_provenance())
    i = args.index("--runs") + 1
    runs = []
    while i < len(args) and not args[i].startswith("--"):
        runs.append(Path(args[i]))
        i += 1
    versions = Counter()
    for run in runs:
        source = json.loads((run / "run_manifest.json").read_text())["source"]
        if code(source) != ref:
            raise SystemExit(f"{run}: producer code differs from the analysis package")
        versions[(run.parent.name, source["version"])] += 1
    print(f"code identity: all {len(runs)} runs match the analysis package "
          f"(aggregate {ref['aggregate_sha256']})")
    for (directory, version), n in sorted(versions.items()):
        print(f"  {directory}: installed-metadata version {version!r} in {n} runs")
    sys.stdout.flush()

    import analyze_v1_wave2  # the stress analyzer imports collect from here

    analyze_v1_wave2.collect = functools.partial(analyze_v1_wave2.collect, check_source=False)
    sys.argv = [str(script), *args]
    if script.stem == "analyze_v1_wave2":
        analyze_v1_wave2.main()
    else:
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
