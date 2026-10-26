import json
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["GSDS_WORK_ROOT"]) / "obsrct-v03-adapters-20261026-r01"
result = {"atol": 1e-10, "rtol": 1e-8, "prediction_checks": {}, "status": "running"}
for stage in ("icm-direct", "icm-adapter-1", "icm-adapter-2"):
    original = sorted((root / "native-v2" / stage).glob("*/record.json"))[-1]
    replay = sorted((root / "environment-replay-native-v1" / stage).glob("*/record.json"))[-1]
    a, b = [json.loads(p.read_text())["result"]["prediction"] for p in (original, replay)]
    np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-8)
    result["prediction_checks"][stage] = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
for basis in ("linear", "quadratic"):
    a = np.load(root / "intr-native-v3" / basis / "wrapper_1.npz")
    b = np.load(root / "environment-replay-intr-v1" / basis / "wrapper_1.npz")
    for field in ("prediction", "tau_beta", "center", "scale"):
        np.testing.assert_allclose(a[field], b[field], atol=1e-10, rtol=1e-8)
        result["prediction_checks"][f"integrative_r_{basis}/{field}"] = float(np.max(np.abs(a[field]-b[field])))
result["status"] = "passed"
(root / "environment-replay-comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
print(json.dumps(result, sort_keys=True))
