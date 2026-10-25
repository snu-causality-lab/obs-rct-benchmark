"""Catalog identity, applicability and disjoint-shard contract regressions."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from obs_rct_benchmark.runner import BenchmarkConfig, ScenarioConfig, build_dgp_config, load_config


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("catalog_configs", ROOT / "scripts/make_catalog_configs.py")
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


def _small_config():
    common = ScenarioConfig("structural", "M0", {
        "n_obs": 8, "n_rct_train": 8, "n_rct_tune": 4, "n_rct_test": 4, "n_eval": 8,
    }, label="small")
    variable = ScenarioConfig("paper", "integrative_r_learner_multivariate_paper", {
        "n_obs": 8, "n_rct_train": 8, "n_rct_tune": 4, "n_rct_test": 4, "n_eval": 8,
    }, label="small")
    return BenchmarkConfig((common, variable), seeds=(17, 19),
        methods=catalog.CATALOG_METHODS, reference_method="rct_dr_extra_trees",
        purpose="reference", method_options={"datafusionppi": {"learner": "DR"}})


def test_complete_declared_grid_preserves_native_dimensions_and_mechanisms():
    config = load_config(ROOT / "configs/catalog_study.toml")
    catalog.validate_catalog_study(config)
    assert len(config.scenarios) == 124 and len(config.seeds) == 64
    assert len(config.methods) == 10 and len(catalog.catalog_items()) == 31
    assert len(catalog.expected_cells(config)) == 79104
    assert len(catalog.expected_not_applicable(config)) == 256
    assert len(catalog.primary_contrasts(config)) == 472
    first_budget = [s for s in config.scenarios if s.label == "o512_r128"]
    paper_dimensions = {
        "cornet_official_n_conf": 25, "experimental_grounding_kps": 1,
        "integrative_r_learner_kps_normal_target_extension": 1,
        "integrative_r_learner_kps_code": 1, "integrative_r_learner_multivariate_paper": 9,
        "elastic_integrative_gen_data_overlap": 2, "power_likelihood_paper_text": 2,
        "power_likelihood_supplement_code": 2, "cvci_official_linear_bias": 5,
    }
    for scenario in first_budget:
        expected_fields = catalog.PAPER_ROLE_FIELDS if scenario.kind == "paper" else catalog.ROLE_FIELDS
        assert set(scenario.parameters) == expected_fields
        actual = build_dgp_config(scenario)
        expected = paper_dimensions[scenario.id] if scenario.kind == "paper" else (
            6 if scenario.id in {"C09", "C10"} else 5)
        assert actual.d == expected
        if scenario.id == "M11":
            assert actual.noise_distribution == "student_t"
            assert catalog.mean_inference_reason(scenario) is not None
        elif scenario.id == "C03":
            assert actual.noise_distribution == "normal"
            assert catalog.mean_inference_reason(scenario) is None


def test_nominal_and_applicable_cells_have_explicit_independent_counts():
    config = _small_config()
    valid = catalog.expected_cells(config)
    excluded = catalog.expected_not_applicable(config)
    # Two conditions × two seeds × ten methods is40 nominal cells; one
    # method is unsupported in one condition at both seeds, leaving38.
    assert len(valid) == 38 and len(excluded) == 2
    nominal = {(scenario.key, seed, method) for scenario in config.scenarios
               for seed in config.seeds for method in config.methods}
    assert valid.isdisjoint(excluded) and valid | set(excluded) == nominal
    assert all(key[2] == "datafusionppi" for key in excluded)
    assert len(catalog.primary_contrasts(config)) == 6


def test_shards_are_disjoint_cell_rectangles_and_remove_unused_options(tmp_path):
    config = _small_config()
    manifest = catalog._write_shards_unchecked(config, tmp_path / "shards")
    assert len(manifest["shards"]) == 4
    assert manifest["counts"]["applicable_fit_cells"] == 38
    seen = set()
    for entry in manifest["shards"]:
        part = load_config(tmp_path / "shards" / entry["path"])
        assert part.config_hash == entry["config_hash"]
        expected = catalog.shard_config(config, entry["seed"], entry["group"])
        assert part.payload() == expected.payload()
        cells = {(s.key, seed, method) for s in part.scenarios
                 for seed in part.seeds for method in part.methods}
        assert seen.isdisjoint(cells)
        seen.update(cells)
        if entry["group"] == "variable_propensity":
            assert "datafusionppi" not in part.methods
            assert "datafusionppi" not in part.method_options
    assert seen == catalog.expected_cells(config)
    saved = json.loads((tmp_path / "shards/study_manifest.json").read_text())
    assert saved == manifest
    with pytest.raises(ValueError, match="empty"):
        catalog._write_shards_unchecked(config, tmp_path / "shards")


def test_public_shard_writer_rejects_noncanonical_design_before_writing(tmp_path):
    output = tmp_path / "shards"
    with pytest.raises(ValueError, match="fixed 64-seed"):
        catalog.write_shards(_small_config(), output)
    assert not output.exists()


def test_frozen_catalog_rejects_registry_expansion(monkeypatch):
    config = load_config(ROOT / "configs/catalog_study.toml")
    monkeypatch.setattr(catalog, "PAPER_DGPS", (*catalog.PAPER_DGPS, "future_preset"))
    with pytest.raises(ValueError, match="frozen 31 preset identities"):
        catalog.validate_catalog_study(config)
    assert len(catalog.catalog_items()) == 31


def test_heavy_tail_cells_are_fitted_but_not_mean_t_primaries():
    config = _small_config()
    heavy = ScenarioConfig("structural", "M11", dict(config.scenarios[0].parameters), label="small")
    config = replace(config, scenarios=(heavy,))
    assert len(catalog.expected_cells(config)) == 20
    assert len(catalog.expected_not_applicable(config)) == 0
    assert len(catalog.primary_contrasts(config)) == 0
    assert catalog.mean_inference_reason(heavy) is not None


def test_design_validator_rejects_posthoc_seed_or_parameter_changes():
    config = load_config(ROOT / "configs/catalog_study.toml")
    with pytest.raises(ValueError, match="fixed 64-seed"):
        catalog.validate_catalog_study(replace(config, seeds=(9,)))
    first = config.scenarios[0]
    changed = replace(first, parameters={**first.parameters, "d": 7})
    with pytest.raises(ValueError, match="only the declared"):
        catalog.validate_catalog_study(replace(config, scenarios=(changed, *config.scenarios[1:])))
    paper = next(scenario for scenario in config.scenarios if scenario.kind == "paper")
    # ScenarioConfig constructs the preset immediately, before protocol validation.
    with pytest.raises(TypeError, match="n_target_reference"):
        replace(paper, parameters={**paper.parameters, "n_target_reference": 0})
    options = {name: dict(value) for name, value in config.method_options.items()}
    options["cornet_official"]["source_epochs"] = 5
    with pytest.raises(ValueError, match="recipes differ"):
        catalog.validate_catalog_study(replace(config, method_options=options))


def test_shard_selection_rejects_undeclared_seed_group_and_empty_rectangle():
    config = _small_config()
    for seed, group in ((True, "common"), (18, "common"), (17, "other")):
        with pytest.raises(ValueError):
            catalog.shard_config(config, seed, group)
    common_only = replace(config, scenarios=config.scenarios[:1])
    with pytest.raises(ValueError, match="no declared scenarios"):
        catalog.shard_config(common_only, 17, "variable_propensity")
