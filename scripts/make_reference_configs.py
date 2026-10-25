"""Split a declared seed grid into disjoint configurations without changing recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib


def write_config(payload: dict, destination: Path) -> None:
    """Write the benchmark's primitive-valued TOML configuration schema."""
    def value(item):
        return json.dumps(item, ensure_ascii=False, allow_nan=False)

    lines = [f"schema_version = {payload['schema_version']}", "", "[run]"]
    lines.extend(f"{key} = {value(item)}" for key, item in payload["run"].items())
    for name, options in payload.get("method_options", {}).items():
        lines.extend(["", f"[method_options.{name}]"])
        lines.extend(f"{key} = {value(item)}" for key, item in options.items())
    for scenario in payload["scenarios"]:
        lines.extend(["", "[[scenarios]]"])
        lines.extend(f"{key} = {value(item)}" for key, item in scenario.items()
                     if key != "parameters")
        lines.append("[scenarios.parameters]")
        lines.extend(f"{key} = {value(item)}"
                     for key, item in scenario.get("parameters", {}).items())
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reference_study.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    payload = tomllib.loads(args.config.read_text(encoding="utf-8"))
    seeds = payload["run"]["seeds"]
    if (isinstance(args.shards, bool) or not 1 <= args.shards <= len(seeds)
            or len(set(seeds)) != len(seeds)):
        parser.error("shards must be between one and the number of unique seeds")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    for shard in range(args.shards):
        part = {**payload, "run": {**payload["run"], "seeds": seeds[shard::args.shards]}}
        write_config(part, args.output / f"shard-{shard:02d}.toml")
    print(f"Wrote {args.shards} disjoint seed shards; complete union has {len(seeds)} seeds.")


if __name__ == "__main__":
    main()
