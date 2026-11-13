"""Export the compact TSR study packet: per-seed scores, analysis outputs and provenance."""
import csv, hashlib, json, shutil, sys
from pathlib import Path
run, analysis, out, commit, outputs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
if out.exists():
    raise SystemExit("export directory exists")
(out / "evidence").mkdir(parents=True)
rows = []
for shard in sorted((run / outputs).glob("shard-*")):
    for line in (shard / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        fi = r.get("fit_info") or {}
        m = r.get("metrics") or {}
        rows.append({"scenario": r["scenario"], "seed": r["seed"], "method": r["method"], "status": r["status"],
                     "cate_mse": m.get("cate_mse"), "root_pehe": m.get("root_pehe"),
                     "error_type": r.get("error_type"), "stage": r.get("stage"),
                     "stage1_device": fi.get("stage1_device"), "gp_converged": fi.get("gp_converged"),
                     "gp_iterations": fi.get("gp_iterations")})
rows.sort(key=lambda r: (r["scenario"], r["seed"], r["method"]))
with (out / "replicate_scores.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for f in sorted(analysis.iterdir()):
    shutil.copy2(f, out / f.name)
manifest0 = json.loads((run / outputs / "shard-00" / "run_manifest.json").read_text())
(out / "evidence" / "producer_identity.json").write_text(json.dumps({
    "code_commit": commit, "outputs": outputs, "source": manifest0["source"], "environment": manifest0["environment"],
    "external_methods": manifest0["external_methods"], "config_hash_shard_00": manifest0["config_hash"]}, indent=2) + "\n")
for name in ("JOBS.txt", f"gpy-check-{commit}.json", f"environment-v1-freeze-{commit}.txt", "shards/SHA256SUMS"):
    src = run / name
    if src.exists():
        shutil.copy2(src, out / "evidence" / Path(name).name)
files = sorted(p for p in out.rglob("*") if p.is_file())
(out / "DELIVERY_MANIFEST.json").write_text(json.dumps({str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2) + "\n")
print(len(rows), "rows;", len(files), "files")
