"""Execute each tutorial in a fresh kernel, keeping portable outputs only.

Install the notebooks extra. Source notebooks are never overwritten. A failed
cell fails the command; publication additionally requires source/diff review.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json
import re
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = re.compile(r"/(?:Users|home)/[^/\s]+/|/mnt/(?:nfs|hdd\d*)/")


def execute(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    source_dir = (ROOT / "notebooks").resolve()
    if output_dir == source_dir:
        raise ValueError("choose an output directory distinct from source notebooks")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scope": "default synthetic tutorial cells; optional external data disabled", "notebooks": {}}
    for source in sorted(source_dir.glob("[0-9][0-9]_*.ipynb")):
        started = time.monotonic()
        raw = source.read_bytes()
        notebook = nbformat.reads(raw.decode(), as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            cell.metadata = {}
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        # This explicit interpreter also works when a global python3 kernel
        # specification points at an unrelated environment. It is in-memory.
        with tempfile.TemporaryDirectory(prefix="obsrct-notebook-") as working:
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client = NotebookClient(
                notebook, km=manager, timeout=120, allow_errors=False,
                record_timing=False, store_widget_state=False,
                resources={"metadata": {"path": working}},
            )
            # An externally supplied manager is cleaned up explicitly below.
            try:
                client.execute(cleanup_kc=True)
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if [cell.execution_count for cell in code_cells] != list(range(1, len(code_cells) + 1)):
            raise ValueError(f"incomplete execution: {source.name}")
        images = 0
        for cell in notebook.cells:
            cell.metadata = {}
            if cell.cell_type == "code":
                for item in cell.outputs:
                    if item.output_type == "error":
                        raise ValueError(f"cell error: {source.name}")
                    if item.output_type in {"display_data", "execute_result"}:
                        item.metadata = {}
                        images += int("image/png" in item.get("data", {}))
        notebook.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        }
        rendered = nbformat.writes(notebook)
        if RUNTIME_PATH.search(rendered):
            raise ValueError(f"runtime path in notebook: {source.name}")
        if len(rendered.encode()) > 1024 * 1024:
            raise ValueError(f"notebook exceeds release size limit: {source.name}")
        if not code_cells or not images:
            raise ValueError(f"tutorial must execute code and produce a figure: {source.name}")
        (output_dir / source.name).write_text(rendered + "\n", encoding="utf-8")
        item = {"code_cells": len(code_cells), "png_figures": images,
                "source_cell_sha256": hashlib.sha256(json.dumps(
                    [(cell.cell_type, cell.source) for cell in notebook.cells],
                    ensure_ascii=False, separators=(",", ":"),
                ).encode()).hexdigest(), "elapsed_seconds": round(time.monotonic() - started, 3)}
        report["notebooks"][source.name] = item
        print(f'{source.name}: {len(code_cells)} code cells, {images} figures, complete', flush=True)
    if not report["notebooks"]:
        raise ValueError("no tutorial notebooks found")
    (output_dir / "execution.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args().output_dir)
