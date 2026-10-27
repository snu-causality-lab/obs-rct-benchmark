"""Check the distributable source tree without importing benchmark methods.

This catches common accidental exports, not every possible disclosure. Review
the exact Git diff and distribution file list before publishing a release.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {
    ".github", ".gitignore", "src", "tests", "docs", "configs", "examples",
    "scripts", "results", "notebooks", "README.md", "LICENSE", "CONTRIBUTING.md",
    "CITATION.cff", "THIRD_PARTY_NOTICES.md", "sources.lock.toml",
    "pyproject.toml", "MANIFEST.in",
}
OMIT = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
PATH_PATTERNS = [
    re.compile(r"/(?:Users|home)/[^/\s]+/"),
    re.compile(r"/mnt/(?:nfs|hdd\d*)/"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
]
PUBLIC_EXTERNAL_IMPORTS = {"numpy", "scipy", "sklearn", "torch", "dfppi", "GPy", "pytest", "nbclient", "nbformat", "jupyter_client", "matplotlib"}


def candidate_files():
    try:
        result = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"], stderr=subprocess.DEVNULL,
        )
        return [ROOT / name.decode() for name in result.split(b"\0") if name]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in ROOT.rglob("*") if p.is_file()
                and not (set(p.relative_to(ROOT).parts) & OMIT)
                and not any(part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)]


def check() -> list[str]:
    errors = []
    for path in sorted(set(candidate_files())):
        relative = path.relative_to(ROOT)
        if relative.parts[0] not in ALLOWED_ROOTS:
            errors.append(f"Unexpected release path: {relative}")
        if path.is_symlink():
            errors.append(f"Symlink is not a standalone source file: {relative}")
            continue
        if path.stat().st_size > 1024 * 1024:
            errors.append(f"Release file exceeds 1 MiB: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            errors.append(f"Non-text release file: {relative}")
            continue
        for pattern in PATH_PATTERNS:
            if pattern.search(content):
                errors.append(f"Personal path or credential pattern: {relative}")
                break
        if path.suffix == ".py":
            try:
                tree = ast.parse(content, filename=str(relative))
            except SyntaxError:
                errors.append(f"Invalid Python syntax: {relative}")
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [item.name.split(".")[0] for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name not in PUBLIC_EXTERNAL_IMPORTS | sys.stdlib_module_names | {"obs_rct_benchmark"}:
                        errors.append(f"Unlisted dependency {name!r}: {relative}")
                if path.name in {"methods.py", "cornet.py", "causal_icm.py", "integrative_r.py", "causal_forest.py"} and isinstance(node, ast.Attribute):
                    if node.attr in {"eval_tau", "rct_test", "truth"}:
                        errors.append(f"Evaluation-only attribute in a method: {relative}")
    return errors


if __name__ == "__main__":
    findings = check()
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        raise SystemExit(1)
    print("Release source checks passed; inspect the exact export before publication.")
