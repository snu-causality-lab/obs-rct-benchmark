"""Export compact retained native-panel records on an allocated server node.

Read-only on the scientific campaign. This does not fit, retry, regenerate data,
or replace the original incomplete-collection verifier. It authenticates its
receipt and exact terminal JSON inventory before exporting their scalar fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from verify_native_panel_delivery import (CAMPAIGN_SHA256, CONFIG_SHA256, DATA_FILES,
    EXPECTED_MISSING, SCALAR_FILES, SUMMARY_SHA256, flattened, read_json, require, run, sha256)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            values = {k:(json.dumps(v) if isinstance(v, list) else v) for k,v in row.items()}
            for k in fields:
                if k.startswith("status_counts__"):
                    values.setdefault(k, 0)
            writer.writerow(values)


def export(campaign, verification, summary_path, code, config_path, destination):
    started = time.monotonic()
    require(os.environ.get("SLURM_JOB_ID"), "run under an allocated Slurm job")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "1"
            and os.environ.get("SLURM_MEM_PER_NODE") == "4096", "request one CPU and 4 GiB")
    resolved_destination = destination.resolve()
    for protected in (campaign, verification, summary_path.parent, code, config_path.parent):
        protected = protected.resolve()
        require(not resolved_destination.is_relative_to(protected)
                and not protected.is_relative_to(resolved_destination), "destination overlaps scientific inputs")
    require(sha256(summary_path) == SUMMARY_SHA256
            == sha256(verification/"verified-summary.json"), "verified summary identity differs")
    require(sha256(config_path) == CONFIG_SHA256, "configuration identity differs")
    receipt = read_json(verification/"receipt.json")
    require(receipt["verified_summary_sha256"] == SUMMARY_SHA256
            and receipt["manifest_sha256"] == CAMPAIGN_SHA256, "receipt differs")
    summary = read_json(summary_path)
    manifest = read_json(campaign/"manifest.json")
    declared = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(hashlib.sha256(canonical).hexdigest() == declared == CAMPAIGN_SHA256,
            "scientific manifest differs")
    require(manifest["bindings"]["config_sha256"] == CONFIG_SHA256, "manifest configuration differs")
    for name, digest in manifest["bindings"]["source_files_sha256"].items():
        require(sha256(code/name) == digest, "frozen scientific source differs: " + name)
    records_root = campaign/"records"
    inventory = hashlib.sha256()
    records = {}
    for path in sorted(records_root.rglob("record.json")):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        relative = str(path.relative_to(records_root))
        inventory.update(relative.encode()+b"\0"+digest.encode()+b"\n")
        record = json.loads(payload)
        c = record["case"]
        identity = (c["scenario_key"], c["n_rct"], c["seed"], record["method"])
        require(identity not in records and record["manifest_sha256"] == CAMPAIGN_SHA256,
                "duplicate or differently bound terminal record")
        metrics = record.get("metrics") or {}
        records[identity] = dict(scenario_key=c["scenario_key"], n_rct=c["n_rct"], seed=c["seed"],
            method=record["method"], status=record["status"],
            cate_mse=metrics.get("cate_mse"), root_pehe=metrics.get("root_pehe"),
            record_path=relative, record_sha256=digest, method_seed=record.get("method_seed"),
            error_type=record.get("error_type", ""), launcher_seconds=record.get("launcher_seconds"),
            fit_predict_seconds=record.get("fit_predict_seconds"))
    require(len(records) == 8445 and inventory.hexdigest() == receipt["record_inventory_sha256"],
            "terminal record union changed since verification")
    expected = {(c["scenario_key"], c["n_rct"], c["seed"], method)
                for c in manifest["cases"] for method in manifest["methods"]}
    require(expected-set(records) == EXPECTED_MISSING and not set(records)-expected,
            "requested identity union differs")
    rows = []
    for identity in sorted(expected):
        if identity in records:
            rows.append(records[identity])
        else:
            s,b,n,m = identity
            rows.append(dict(scenario_key=s,n_rct=b,seed=n,method=m,status="missing",cate_mse=None,
                root_pehe=None,record_path="",record_sha256="",method_seed=None,error_type="",
                launcher_seconds=None,fit_predict_seconds=None))
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"evidence").mkdir()
    shutil.copyfile(summary_path,destination/"summary.json")
    shutil.copyfile(verification/"receipt.json",destination/"evidence/verification_receipt.json")
    (destination/"scalar_records").mkdir()
    # Explicit names keep each public text file below the repository's 1 MiB cap.
    scenarios = ["structural:M0", "structural:M2", "structural:M10", "addendum:C08@trial_target",
                 "paper:integrative_r_learner_kps_code", "paper:cornet_official_n_conf"]
    index = []
    for scenario, name in zip(scenarios, SCALAR_FILES):
        shard = [r for r in rows if r["scenario_key"] == scenario]
        write_csv(destination/name, shard)
        index.append(dict(scenario_key=scenario,path=name,rows=len(shard),sha256=sha256(destination/name)))
    write_csv(destination/"scalar_records.csv", index)
    write_csv(destination/"non_successful_fits.csv", [r for r in rows if r["status"] != "success"])
    write_csv(destination/"method_summaries.csv",[flattened(r) for r in summary["method_summaries"]])
    write_csv(destination/"paired_comparisons.csv",[flattened(r) for r in summary["paired_comparisons"]])
    producer = dict(schema_version=1, scientific_fitter_revision="83c45c3acca5ffd316a48153647b078280d648cb",
        campaign_manifest_sha256=CAMPAIGN_SHA256, original_manifest_file_sha256=sha256(campaign/"manifest.json"),
        config_sha256=CONFIG_SHA256, source_files_sha256=manifest["bindings"]["source_files_sha256"],
        runtime_inventory_sha256={m:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
                                  for m,v in manifest["bindings"]["runtime_inventory"].items()},
        original_summary_sha256=SUMMARY_SHA256, verification=summary["verification"],
        omitted="Raw observed/evaluation/prediction arrays, native artifacts, environments and process logs stay on research storage.")
    write_json(destination/"evidence/producer_identity.json",producer)
    write_json(destination/"DELIVERY_MANIFEST.json",dict(schema_version=1,
        scope="compact native-panel delivery from the preserved incomplete collection",
        files={name:sha256(destination/name) for name in DATA_FILES},
        requested_fits=8448,terminal_fits=8445,missing_fits=3,
        original_summary_sha256=SUMMARY_SHA256,terminal_inventory_sha256=inventory.hexdigest()))
    result=run(config_path,destination)
    result.update(slurm_job_id=os.environ["SLURM_JOB_ID"],export_seconds=time.monotonic()-started,
        exporter_sha256=sha256(Path(__file__)),checker_sha256=sha256(Path(__file__).with_name("verify_native_panel_delivery.py")),
        scientific_campaign_written=False)
    write_json(destination/"evidence/public_arithmetic_verification.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "input_sha256"},sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","verification","summary","code","config","destination"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    export(a.campaign,a.verification,a.summary,a.code,a.config,a.destination)


if __name__ == "__main__":
    main()
