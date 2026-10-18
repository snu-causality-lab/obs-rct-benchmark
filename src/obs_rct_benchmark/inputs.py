"""Resolve separately obtained empirical data without putting paths in results."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .datasets import EmpiricalDGPConfig, load_acic2016, load_ihdp_csv, load_ihdp_npz


_SOURCES = {
    "ihdp": (("ihdp_train", "OBSRCT_IHDP_TRAIN", True),
             ("ihdp_test", "OBSRCT_IHDP_TEST", False)),
    "acic2016": (("acic_covariates", "OBSRCT_ACIC_X", True),
                 ("acic_response", "OBSRCT_ACIC_RESPONSE", True)),
}


def empirical_input_provenance(dataset_id: str) -> dict[str, Any]:
    """Stream hashes of requested source files; emit fixed role labels only.

    Missing inputs are reportable rather than an excuse to omit failed cells.
    The data's distribution terms are documented in docs/DATASETS.md.
    """
    if dataset_id not in _SOURCES:
        raise ValueError("empirical dataset must be ihdp or acic2016")
    files, missing = {}, []
    for role, variable, required in _SOURCES[dataset_id]:
        raw = os.environ.get(variable)
        if not raw:
            if required:
                missing.append(role)
            continue
        digest = hashlib.sha256()
        try:
            with Path(raw).expanduser().open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            missing.append(role)
            continue
        files[role] = {"sha256": digest.hexdigest()}
    return {"dataset_id": dataset_id, "availability": "available" if not missing else "unavailable",
            "files": files, "missing_roles": missing}


def _path(variable: str, *, required: bool = True) -> Path | None:
    raw = os.environ.get(variable)
    if not raw:
        if required:
            raise ValueError(f"set {variable} to a separately obtained local file")
        return None
    return Path(raw).expanduser()


def load_empirical_from_environment(config: EmpiricalDGPConfig):
    if config.dataset_id == "ihdp":
        train = _path("OBSRCT_IHDP_TRAIN")
        test = _path("OBSRCT_IHDP_TEST", required=False)
        if train.suffix.lower() == ".csv":
            if test is not None or config.replication != 0:
                raise ValueError("a single IHDP CSV has no separate test file or replication axis")
            return load_ihdp_csv(train)
        return load_ihdp_npz(train, test, replication=config.replication)
    if config.dataset_id == "acic2016":
        return load_acic2016(_path("OBSRCT_ACIC_X"), _path("OBSRCT_ACIC_RESPONSE"))
    raise ValueError("empirical dataset must be ihdp or acic2016")
