"""Record a separately completed grf source build on an allocated server.

This binds observed source/installed bytes to the supplied build command. It
does not compile, install, download packages or prove compiler correspondence.
The manifest contains private operational paths and must stay with the build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from obs_rct_benchmark import causal_forest as adapter


def runtime_files(package: Path) -> dict[str, str]:
    """Bind installed R code, metadata and native libraries, excluding help/headers."""
    files = {}
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if not path.is_file() or not (relative.parts[0] in {"R", "libs", "data"}
                or relative.as_posix() in {"DESCRIPTION", "NAMESPACE", "Meta/package.rds", "Meta/nsInfo.rds"}):
            continue
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(package):
            raise RuntimeError("runtime file leaves the selected dependency package")
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def record(source: Path, library: Path, rscript: Path, build_command: Path, output: Path):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("record the native installation in an allocated server job")
    if output.exists():
        raise FileExistsError("choose a new build-record output; retain prior records")
    source, library, rscript = (path.resolve(strict=True) for path in (source, library, rscript))
    if adapter._git(source, "rev-parse", "HEAD") != adapter.GRF_REVISION:
        raise RuntimeError("source checkout is not the declared grf revision")
    if adapter._git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("retain a pristine source checkout separate from build intermediates")
    command = build_command.read_text().strip()
    if not command:
        raise ValueError("record the actual completed build command")
    files, links = adapter._source_inventory(source)
    completed = subprocess.run([str(rscript), "--vanilla", "-e", adapter._R_PROBE, str(library)],
                               capture_output=True, text=True, check=False, timeout=120)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".probe.stdout").write_text(completed.stdout)
    output.with_suffix(".probe.stderr").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError("native runtime probe failed; process logs retained")
    version, package_version, dependencies = None, None, {}
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if fields[0] == "R" and len(fields) == 2:
            version = fields[1]
        elif fields[0] == "namespace" and len(fields) == 4 and fields[1] == "grf":
            package_version = fields[2]
        elif fields[0] == "package" and len(fields) == 4 and fields[1] != "grf":
            name, installed_version, raw_path = fields[1:]
            path = Path(raw_path).resolve(strict=True)
            dependencies[name] = {"version": installed_version, "library_root": str(path.parent),
                "description_sha256": adapter._sha256(path/"DESCRIPTION"),
                "files_sha256": runtime_files(path)}
    if not version or package_version != adapter.GRF_VERSION:
        raise RuntimeError("unexpected R or grf package version")
    manifest = {
        "schema_version": 1, "repo_revision": adapter.GRF_REVISION,
        "repo_tree": adapter._git(source, "rev-parse", "HEAD^{tree}"),
        "source_root": str(source), "library_root": str(library),
        "source_files_sha256": files, "source_symlink_targets": links,
        "installed_package": {"name": "grf", "version": package_version,
                              "files_sha256": adapter._file_inventory(library/"grf")},
        "rscript_sha256": adapter._sha256(rscript), "r_version": version,
        "dependencies": dependencies, "build_command": command,
        "binding_scope": "recorded completed build command and observed bytes; not a compiler proof",
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.environ.update(GRF_SOURCE=str(source), GRF_LIBRARY=str(library),
                      GRF_RSCRIPT=str(rscript), GRF_BUILD_MANIFEST=str(output.resolve()))
    verified = adapter.causal_forest_provenance()
    if not verified.get("verified"):
        raise RuntimeError("recorded installation failed independent pinned-source/runtime checks")
    return {"status": "recorded", "manifest_sha256": adapter._sha256(output),
            "source_files": len(files), "installed_grf_files": len(manifest["installed_package"]["files_sha256"]),
            "dependency_packages": len(dependencies), "version": package_version}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--build-command-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(record(args.source, args.library, args.rscript, args.build_command_file, args.output)))
