"""Adapter to a separately obtained, pinned official CorNet checkout.

No upstream code is bundled. The fixed recipe is deliberately explicit: it
calls the released training and armwise correction routines without tuning on
simulation truth. Optional source standardization is a benchmark adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .data import TrainingView


CORNET_REVISION = "38b9c37d5a86bc87950945d6ddb1ceb5354a45ba"
CORNET_FILES = {
    "model/models.py": "31ff1086ca97fe729cff320338da3e3dd619bb879668e46de94c5bea4366826a",
    "model/helper.py": "f18e6e7eed78cd25896e9344a470f8c35be888a14eb3831a3bbf6acb307000ac",
    "model/functions.py": "5eda20f55c31c69968f6dc40a8ad2973482421c16b61ca0b1b47efcf49e5685e",
    "simulation_studies/simdata.py": "07576aabc15e490fa60595b1cdf3c3631eb60dd5344c14a5d3cbde06a9fa4541",
}


def cornet_provenance():
    """Sanitized runtime evidence, including files actually imported upstream."""
    result = {"expected_revision": CORNET_REVISION, "availability": "unavailable",
              "declared_git_revision": None, "package_file_sha256": {},
              "tracked_changes": None, "extra_import_files": [], "verified": False}
    raw = os.environ.get("CORNET_SOURCE")
    if not raw:
        return result
    root = Path(raw).expanduser().resolve()
    try:
        result["declared_git_revision"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        result["tracked_changes"] = bool(changes)
        result["package_file_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in CORNET_FILES
        }
        # The pinned tree uses namespace packages, without __init__.py files.
        # An added package initializer would execute even if Git ignores it.
        extra = [name for name in ("model/__init__.py", "simulation_studies/__init__.py")
                 if (root / name).exists()]
        result.update(availability="available", extra_import_files=extra)
        result["verified"] = (
            result["declared_git_revision"] == CORNET_REVISION
            and not result["tracked_changes"] and not extra
            and result["package_file_sha256"] == CORNET_FILES
        )
    except (OSError, subprocess.CalledProcessError):
        return result
    return result


def _official_class():
    raw = os.environ.get("CORNET_SOURCE")
    if not raw:
        raise RuntimeError("cornet_official requires CORNET_SOURCE; see README")
    root = Path(raw).expanduser().resolve()
    if not (root / "model" / "models.py").is_file():
        raise ValueError("CORNET_SOURCE must contain model/models.py")
    if not cornet_provenance()["verified"]:
        raise ValueError("CorNet source files do not match the locked public implementation")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot verify the external CorNet Git checkout") from exc
    if revision != CORNET_REVISION or changes:
        raise ValueError("CorNet checkout must be clean and match the locked revision")
    # Upstream uses generic top-level imports. Refuse cached modules belonging
    # to another checkout rather than silently running a different model.
    for name in ("model", "model.models", "model.helper", "model.functions",
                 "simulation_studies", "simulation_studies.simdata"):
        cached = sys.modules.get(name)
        if cached is None:
            continue
        filename = getattr(cached, "__file__", None)
        locations = getattr(cached, "__path__", ())
        if filename is not None and not Path(filename).resolve().is_relative_to(root):
            raise RuntimeError("An unrelated upstream module is already imported")
        if any(not Path(p).resolve().is_relative_to(root) for p in locations):
            raise RuntimeError("An unrelated upstream namespace is already imported")
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        # Upstream has a models -> helper -> simdata -> models cycle. Its
        # simulation scripts enter through simdata, which resolves that cycle.
        importlib.import_module("simulation_studies.simdata")
        module = importlib.import_module("model.models")
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("CorNet import did not resolve to the locked checkout")
        return module.CorNet
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous


@dataclass
class OfficialCorNet:
    """Fixed public CorNet recipe, CPU only, without early stopping.

    OBS fits the shared representation; RCT training fits the released
    arm-specific residual corrections. The tuning sample is not consumed.
    Standardization uses OBS covariates/outcomes only and is frozen for RCT
    and query prediction. This is an adapted recipe, not an exact reproduction
    of any paper table or a tuned implementation.
    """

    standardize: bool = True
    source_epochs: int = 500
    correction_epochs: int = 200
    width: int = 25
    learning_rate: float = 0.1
    outcome_penalty: float = 0.1
    correction_penalty: float = 0.0

    def __post_init__(self):
        if not isinstance(self.standardize, bool):
            raise ValueError("standardize must be boolean")
        for key in ("source_epochs", "correction_epochs", "width"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("learning_rate", "outcome_penalty", "correction_penalty"):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")

    def fit_predict(self, training: TrainingView, eval_X, *, seed: int):
        if not isinstance(training, TrainingView):
            raise TypeError("methods require a TrainingView")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install the cornet optional dependencies") from exc
        cls = _official_class()
        obs, rct = training.obs_train, training.rct_train
        for sample in (obs, rct):
            if set(np.unique(sample.T)) != {0.0, 1.0}:
                raise ValueError("CorNet requires both treatment arms in each fitting source")
        x_mean = np.zeros(obs.d)
        x_scale = np.ones(obs.d)
        y_mean, y_scale = 0.0, 1.0
        if self.standardize:
            x_mean = np.mean(obs.X, axis=0)
            x_scale = np.maximum(np.std(obs.X, axis=0), 1e-8)
            y_mean = float(np.mean(obs.Y))
            y_scale = max(float(np.std(obs.Y)), 1e-8)

        def tensor(values):
            return torch.tensor(np.asarray(values), dtype=torch.float32, device="cpu")

        ox = tensor((obs.X - x_mean) / x_scale)
        oy = tensor((obs.Y - y_mean) / y_scale).reshape(-1, 1)
        oa = tensor(obs.T).reshape(-1, 1)
        rx = tensor((rct.X - x_mean) / x_scale)
        ry = tensor((rct.Y - y_mean) / y_scale).reshape(-1, 1)
        ra = tensor(rct.T).reshape(-1, 1)
        query = tensor((np.asarray(eval_X) - x_mean) / x_scale)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                model = cls(obs.d, 3, self.width, 0, self.width)
                model._train(
                    ox, oy, oa, [], [], [], self.outcome_penalty, obs.n,
                    self.learning_rate, self.source_epochs, early_stop=False,
                )
                model._retrain_delta(
                    rx, ry, ra, self.correction_penalty, self.learning_rate,
                    self.correction_epochs, bias_norm=1, bias_learning=True,
                )
                with torch.no_grad():
                    prediction = model.predict_delta(query).detach().cpu().numpy().reshape(-1)
        finally:
            torch.set_num_threads(previous_threads)
        prediction = np.asarray(prediction, dtype=np.float64) * y_scale
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("Official CorNet produced nonfinite predictions")
        return prediction
