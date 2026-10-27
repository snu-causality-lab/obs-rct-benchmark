"""Optional native recipe registration and provenance-sensitive runner checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from obs_rct_benchmark import causal_forest, methods, runner
from obs_rct_benchmark.adaptive_runner import AdaptiveConfig, _provenance
from obs_rct_benchmark.runner import BenchmarkConfig, ScenarioConfig


def test_registration_is_lightweight_and_does_not_infer_moment_permission(monkeypatch):
    def forbidden():
        raise AssertionError("construction must not inspect R")
    monkeypatch.setattr(causal_forest, "causal_forest_provenance", forbidden)
    estimator = methods.make_method("rct_causal_forest")
    assert isinstance(estimator, causal_forest.RCTCausalForest)
    assert (estimator.num_trees, estimator.num_threads) == (2000, 1)
    policy = runner.resolve_inference_policy(ScenarioConfig("structural", "M0"),
                                             ("rct_t_ridge", "rct_causal_forest"))
    assert not policy["cate_mse"]["eligible"]
    assert not policy["root_pehe"]["eligible"]


def test_external_snapshot_is_requested_only_and_binds_adaptive_runtime(monkeypatch):
    calls = []
    snapshot = {"version": "2.6.1", "build_manifest_sha256": "original"}
    def probe():
        calls.append("grf")
        return dict(snapshot)
    monkeypatch.setattr(causal_forest, "causal_forest_provenance", probe)
    assert runner.external_method_provenance(("rct_t_ridge",)) == {}
    assert calls == []
    assert runner.external_method_provenance(("rct_causal_forest",)) == {"rct_causal_forest": snapshot}
    config = AdaptiveConfig((ScenarioConfig("structural", "M0"),), estimators=("rct_causal_forest",))
    original, first = _provenance(config)
    assert original["external_methods"]["rct_causal_forest"] == snapshot
    snapshot["build_manifest_sha256"] = "changed"
    assert _provenance(config)[1] != first


@pytest.mark.parametrize("field", ["build_manifest_sha256", "rscript_sha256", "installed_package_files_sha256"])
def test_changed_grf_build_runtime_or_native_bytes_reject_static_resume(tmp_path, monkeypatch, field):
    snapshot = {"version": "2.6.1", "build_manifest_sha256": "build",
                "rscript_sha256": "runtime", "installed_package_files_sha256": {"libs/grf.so": "native"}}
    monkeypatch.setattr(causal_forest, "causal_forest_provenance", lambda: dict(snapshot))
    monkeypatch.setattr(runner, "_generate", lambda *args: SimpleNamespace(
        metadata={}, training_view=lambda: object(), eval_X=np.zeros((4, 2)), eval_tau=np.zeros(4)))
    monkeypatch.setattr(runner, "make_method", lambda *args, **kwargs: SimpleNamespace(
        fit_predict=lambda training, query, seed: np.zeros(len(query))))
    config = BenchmarkConfig((ScenarioConfig("structural", "M0"),),
                             methods=("rct_t_ridge", "rct_causal_forest"), seeds=(3,))
    output = tmp_path/"grf"
    runner.run_benchmark(config, output)
    record_bytes = (output/"records.jsonl").read_bytes()
    manifest = json.loads((output/"run_manifest.json").read_text())
    assert manifest["external_methods"]["rct_causal_forest"] == snapshot
    snapshot[field] = {"libs/grf.so": "changed"} if isinstance(snapshot[field], dict) else "changed"
    with pytest.raises(ValueError, match="config or source provenance"):
        runner.run_benchmark(config, output, resume=True)
    assert (output/"records.jsonl").read_bytes() == record_bytes
