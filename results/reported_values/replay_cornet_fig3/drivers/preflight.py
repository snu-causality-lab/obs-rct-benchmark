"""Preflight: environment, source identities, Arm A imports and one tiny Arm B fit per setting.

Usage: python preflight.py --code SRC_DIR --cornet DIR --stubs DIR
Outputs are execution checks only; they are not replay results.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--code", required=True)
parser.add_argument("--cornet", required=True)
parser.add_argument("--stubs", required=True)
args = parser.parse_args()
root = Path(args.cornet).resolve()
checks = {}

script = root / "simulation_studies" / "sim_n_conf.py"
checks["sim_n_conf_sha256_ok"] = hashlib.sha256(script.read_bytes()).hexdigest() == \
    "9d99f06ae92650114dc8ad314a93446703491b4d4502d4af8922eba993e581c3"

# Arm A imports in a separate interpreter, exactly as the executed lines import them.
probe = (
    "import os, sys; os.environ['MPLBACKEND']='Agg'; "
    f"sys.path.insert(0, {str(root)!r}); "
    "import numpy, torch, sklearn.model_selection, sklearn.utils.fixes, matplotlib.pyplot, matplotlib.lines, "
    "scipy.sparse, simulation_studies.simdata, model.models; "
    "assert 'pandas' not in sys.modules and 'seaborn' not in sys.modules; "
    "assert sklearn.utils.fixes.pd is None; "
    f"sys.path.insert(0, {str(Path(args.stubs).resolve())!r}); "
    "import numpy as np; import pandas as pd; from sklearn.model_selection import train_test_split; "
    "import matplotlib.pyplot as plt; import matplotlib; from pathlib import Path; "
    "from simulation_studies.simdata import *; from model.models import *; "
    "from scipy.sparse import random; import seaborn as sns; from matplotlib.lines import Line2D; "
    "print('ARM_A_IMPORTS_OK', pd.__file__, sns.__file__)"
)
proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
checks["arm_a_imports_ok"] = proc.returncode == 0 and "ARM_A_IMPORTS_OK" in proc.stdout
checks["arm_a_import_output"] = (proc.stdout + proc.stderr)[-2000:]

os.environ["CORNET_SOURCE"] = str(root)
sys.path.insert(0, str(Path(args.code).resolve()))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from obs_rct_benchmark.cornet import OfficialCorNet, cornet_provenance  # noqa: E402
from obs_rct_benchmark.paper_dgp import generate_paper_problem, make_paper_config  # noqa: E402

prov = cornet_provenance()
checks["cornet_provenance_verified"] = bool(prov["verified"])
cfg = make_paper_config(name="preflight", paper_id="cornet_official_n_conf", mechanism_seed=2025,
                        n_obs=25, n_rct_train=25, n_rct_tune=4, n_rct_test=4, n_eval=1000)
problem = generate_paper_problem(cfg, seed=0)
fits = {}
for standardize in (False, True):
    pred = OfficialCorNet(standardize=standardize).fit_predict(problem.training_view(), problem.eval_X, seed=0)
    fits[str(standardize)] = bool(np.all(np.isfinite(pred))) and pred.shape == problem.eval_tau.shape
checks["arm_b_tiny_fits_ok"] = fits
checks["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}
checks["node"] = os.environ.get("SLURMD_NODENAME")
print(json.dumps(checks, indent=2))
ok = (checks["sim_n_conf_sha256_ok"] and checks["arm_a_imports_ok"]
      and checks["cornet_provenance_verified"] and all(fits.values()))
sys.exit(0 if ok else 1)
