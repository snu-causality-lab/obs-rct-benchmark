"""Split or rejoin this packet's per-seed scores (publication step, run outside the compute job).

The analyzer wrote one `replicate_scores.csv`, larger than the repository's 1 MiB limit per
file, so the packet stores it as one file per group and preset under `replicate_scores/`.
Rejoining them must reproduce the exported file byte for byte; `--check` verifies this
against the SHA-256 recorded in `DELIVERY_MANIFEST.json`.

    python results/<packet>/evidence/split_replicate_scores.py --check
"""
import argparse, csv, hashlib, io, json
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]


def _key(row):
    return (row["group"], row["scenario"], int(row["seed"]), row["method"])  # the export's order


def _write(rows, fields) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def split(source: Path, folder: Path) -> None:
    with source.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    groups = {}
    for row in rows:
        name = f"{row['group']}__{row['scenario'].split('@')[0].replace(':', '__')}"
        groups.setdefault(name, []).append(row)
    folder.mkdir(exist_ok=False)
    for name, items in sorted(groups.items()):
        (folder / f"{name}.csv").write_bytes(_write(items, fields))
    print(f"{len(rows)} rows into {len(groups)} files")


def join(folder: Path) -> bytes:
    rows, fields = [], None
    for path in sorted(folder.glob("*.csv")):
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fields = fields or reader.fieldnames
            if reader.fieldnames != fields:
                raise SystemExit(f"header differs: {path.name}")
            rows.extend(reader)
    return _write(sorted(rows, key=_key), fields)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, help="exported replicate_scores.csv to split")
    ap.add_argument("--check", action="store_true", help="rejoin and compare with DELIVERY_MANIFEST.json")
    args = ap.parse_args()
    folder = PACKET / "replicate_scores"
    if args.split:
        split(args.split, folder)
    if args.check:
        expected = json.loads((PACKET / "DELIVERY_MANIFEST.json").read_text())["files_sha256_as_exported"]["replicate_scores.csv"]
        got = hashlib.sha256(join(folder)).hexdigest()
        print("rejoined replicate_scores.csv", "matches" if got == expected else "DIFFERS", got)
        raise SystemExit(0 if got == expected else 1)
