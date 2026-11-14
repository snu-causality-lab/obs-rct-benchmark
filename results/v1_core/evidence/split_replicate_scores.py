"""Split or rejoin the v1 core packet's per-seed scores (publication step, run outside the compute job).

The analyzer (`scripts/analyze_v1_core.py`) wrote one `replicate_scores.csv` of about 9.6 MB. The
repository keeps release files under 1 MiB, so the published packet stores it as one file per
preset under `replicate_scores/`. Rejoining them must reproduce the exported file byte for byte;
`--check` verifies this against the SHA-256 recorded in `DELIVERY_MANIFEST.json`.

    python results/v1_core/evidence/split_replicate_scores.py --check
"""
import argparse, csv, hashlib, io, json
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]


def _key(row):
    return (row["scenario"], int(row["seed"]), row["method"])  # the export's sort order


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
        groups.setdefault(row["scenario"].split("@")[0], []).append(row)
    folder.mkdir(exist_ok=False)
    for preset, items in sorted(groups.items()):
        (folder / (preset.replace(":", "__") + ".csv")).write_bytes(_write(items, fields))
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
        manifest = json.loads((PACKET / "DELIVERY_MANIFEST.json").read_text())
        expected = manifest["files_sha256_as_exported"]["replicate_scores.csv"]
        joined = join(folder)
        actual = hashlib.sha256(joined).hexdigest()
        if actual != expected:
            raise SystemExit(f"rejoined SHA-256 {actual} differs from the export's {expected}")
        n_rows = joined.count(b"\n") - 1
        print(f"rejoined {n_rows} rows; SHA-256 matches the export ({expected[:12]}…)")
