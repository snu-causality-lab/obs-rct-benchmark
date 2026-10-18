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

__version__ = "0.1.0a0"

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
]
