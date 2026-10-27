"""OBS/RCT CATE simulation assets with separate training and evaluation roles."""

from .data import (
    SimulationProblem,
    SimulationTruth,
    SourceSample,
    TargetCovariateSample,
    TrainingView,
)
from .paper_dgp import (
    PAPER_DGPS,
    PaperDGPConfig,
    generate_paper_problem,
    make_paper_config,
    paper_truth_at,
)
from .structural_dgp import (
    STRUCTURAL_ADDENDUM_CELLS,
    STRUCTURAL_MECHANISMS,
    StructuralDGPConfig,
    generate_structural_problem,
    make_structural_addendum_config,
    make_structural_config,
    structural_truth_at,
)
from .sensitivity import (
    GammaDGPConfig, construct_msm_propensities, gamma_truth_at,
    generate_gamma_problem, msm_truth_at, sample_msm_source,
)
from .datasets import (
    EmpiricalDGPConfig, EmpiricalPopulation, generate_empirical_problem,
    load_acic2016, load_ihdp_csv, load_ihdp_npz,
)

__version__ = "0.3.0a3"

__all__ = [
    "SimulationProblem",
    "SimulationTruth",
    "SourceSample",
    "TargetCovariateSample",
    "TrainingView",
    "PAPER_DGPS",
    "PaperDGPConfig",
    "make_paper_config",
    "generate_paper_problem",
    "paper_truth_at",
    "STRUCTURAL_MECHANISMS",
    "STRUCTURAL_ADDENDUM_CELLS",
    "StructuralDGPConfig",
    "make_structural_config",
    "make_structural_addendum_config",
    "generate_structural_problem",
    "structural_truth_at",
    "GammaDGPConfig",
    "construct_msm_propensities",
    "gamma_truth_at",
    "generate_gamma_problem",
    "msm_truth_at",
    "sample_msm_source",
    "EmpiricalDGPConfig",
    "EmpiricalPopulation",
    "generate_empirical_problem",
    "load_acic2016",
    "load_ihdp_csv",
    "load_ihdp_npz",
]
