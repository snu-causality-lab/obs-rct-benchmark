"""Independent hand regression for the native prediction-map verifier."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "grf_admission", Path(__file__).parents[1]/"scripts/admit_causal_forest.py")
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)
record_spec = importlib.util.spec_from_file_location(
    "grf_build_recorder", Path(__file__).parents[1]/"scripts/record_causal_forest_build.py")
recorder = importlib.util.module_from_spec(record_spec)
record_spec.loader.exec_module(recorder)


def test_weighted_local_intercept_is_distinct_from_two_wrong_prediction_maps():
    weights = [.1, .2, .3, .4]
    prediction, variance = admission.weighted_intercept_slopes(
        [weights], [0, 1, 0, 1], [0, 3, 1, 5], [0, 1, 1, 2], [.5]*4)
    # Weighted residual means are y=1.6,w=.1, yw=.8,w²=.25.
    # Covariance .64 / variance .24 = 8/3.
    assert prediction[0] == pytest.approx(8/3, abs=1e-12)
    assert variance[0] == pytest.approx(.24, abs=1e-12)
    assert abs(prediction[0]-3.2) > .5  # wrong uncentered residual ratio
    assert abs(prediction[0]-43/12) > .5  # wrong raw weighted arm means


@pytest.mark.parametrize("weights,treatment", [
    ([[.2, .2, .2, .2]], [0, 1, 0, 1]),
    ([[1.1, -.1, 0, 0]], [0, 1, 0, 1]),
    ([[.25]*4], [0]*4),
    ([[np.nan, 0, 0, 1]], [0, 1, 0, 1]),
])
def test_prediction_map_rejects_invalid_neighborhoods(weights, treatment):
    with pytest.raises(ValueError):
        admission.weighted_intercept_slopes(weights, treatment, [0, 3, 1, 5],
                                             [0, 1, 1, 2], [.5]*4)


def test_build_recorder_binds_runtime_code_and_data_without_claiming_headers(tmp_path):
    for name in ("DESCRIPTION", "NAMESPACE", "R/code.rdb", "R/code.rdx", "libs/native.so",
                 "data/input.rda", "Meta/nsInfo.rds", "include/header.h", "help/index.rdb"):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    files = recorder.runtime_files(tmp_path)
    assert set(files) == {"DESCRIPTION", "NAMESPACE", "R/code.rdb", "R/code.rdx",
                           "libs/native.so", "data/input.rda", "Meta/nsInfo.rds"}
    previous = files["R/code.rdb"]
    (tmp_path/"R/code.rdb").write_text("changed runtime code")
    assert recorder.runtime_files(tmp_path)["R/code.rdb"] != previous


def test_build_recorder_refuses_runtime_links_outside_selected_package(tmp_path):
    (tmp_path/"R").mkdir()
    outside = tmp_path.parent/(tmp_path.name+"-external-code")
    outside.write_text("external runtime")
    (tmp_path/"R/code.rdb").symlink_to(outside)
    with pytest.raises(RuntimeError, match="runtime file leaves"):
        recorder.runtime_files(tmp_path)
