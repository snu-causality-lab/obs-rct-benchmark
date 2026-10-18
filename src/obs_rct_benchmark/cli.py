"""Console entry point for listing methods and executing benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys

from .methods import method_names
from .runner import load_config, run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="obs-rct-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-methods", help="list built-in and registered methods")
    run = commands.add_parser("run", help="run a TOML benchmark configuration")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list-methods":
        print("\n".join(method_names()))
        return 0
    try:
        summary = run_benchmark(load_config(args.config), args.output, resume=args.resume)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        # Exception strings can include local paths; expose only their type.
        print(f"benchmark setup/resume failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "coverage": summary["coverage"],
                      "config_hash": summary["config_hash"]}, allow_nan=False))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
