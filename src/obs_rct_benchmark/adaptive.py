"""Outcome-budgeted, two-stage recruitment from an evaluator-owned RCT pool.

Policies see covariates and already revealed trial observations, never the
unqueried treatments/outcomes or evaluation rows/truth. These simple reference
policies make no claim to optimal acquisition or causal trust estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .data import SimulationProblem, SourceSample, TrainingView
from .evaluation import cate_metrics
from .methods import TLearner, make_method
from .runner import method_seed

IntArray = NDArray[np.int64]
POLICY_NAMES = ("uniform", "geometry", "outcome_residual")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _indices(values: Any, *, size: int, n: int | None = None) -> IntArray:
    raw = np.asarray(values)
    if raw.ndim == 1 and raw.size == 0:
        raw = np.empty(0, dtype=np.int64)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("query indices must be a one-dimensional integer array")
    if ((raw < 0) | (raw >= size)).any():
        raise ValueError("query index lies outside the candidate pool")
    if len(np.unique(raw)) != len(raw) or (n is not None and len(raw) != n):
        raise ValueError("queries must be distinct and match the requested count")
    result = np.array(raw, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _sample_rows(sample: SourceSample, indices: IntArray, role: str) -> SourceSample:
    return SourceSample(sample.X[indices], sample.T[indices], sample.Y[indices],
                        sample.source, role,
                        None if sample.known_propensity is None else sample.known_propensity[indices])


@dataclass(frozen=True)
class RecruitmentView:
    """The complete policy information boundary; no unrevealed trial labels."""

    obs_train: SourceSample
    pool_X: NDArray[np.float64]
    revealed_indices: IntArray
    revealed: SourceSample | None
    remaining_budget: int

    def __post_init__(self) -> None:
        X = np.array(self.pool_X, dtype=np.float64, copy=True)
        if X.ndim != 2 or X.shape[0] == 0 or not np.isfinite(X).all():
            raise ValueError("pool covariates must be a nonempty finite matrix")
        ids = _indices(self.revealed_indices, size=len(X))
        if self.obs_train.source != "obs" or self.obs_train.role != "obs_train":
            raise ValueError("policy OBS sample must have the obs_train role")
        if self.obs_train.d != X.shape[1]:
            raise ValueError("pool and OBS covariate dimensions differ")
        if isinstance(self.remaining_budget, bool) or not isinstance(self.remaining_budget, int) or self.remaining_budget < 0:
            raise ValueError("remaining budget must be a nonnegative integer")
        if len(ids) == 0:
            if self.revealed is not None:
                raise ValueError("empty query history must not contain trial observations")
        elif (self.revealed is None or self.revealed.source != "rct"
              or self.revealed.role != "rct_train" or self.revealed.n != len(ids)
              or not np.array_equal(self.revealed.X, X[ids])):
            raise ValueError("revealed observations must match the ordered queried rows")
        X.setflags(write=False)
        object.__setattr__(self, "pool_X", X)
        object.__setattr__(self, "revealed_indices", ids)

    @property
    def available_indices(self) -> IntArray:
        result = np.flatnonzero(~np.isin(np.arange(len(self.pool_X)), self.revealed_indices))
        result.setflags(write=False)
        return result


class RecruitmentPolicy(Protocol):
    def select(self, view: RecruitmentView, n: int, *, seed: int) -> IntArray: ...


class RecruitmentError(RuntimeError):
    """A terminal policy failure retaining every outcome already purchased."""

    def __init__(self, error_type: str, queried: IntArray):
        super().__init__(error_type)
        self.error_type = error_type
        self.query_indices = queried


class RecruitmentSession:
    """Evaluator-owned query state. Only ``view()`` is handed to a policy."""

    def __init__(self, obs_train: SourceSample, pool: SourceSample, budget: int):
        _positive_integer(budget, "budget")
        if pool.source != "rct" or budget > pool.n:
            raise ValueError("budget must fit an RCT candidate pool")
        self._obs = obs_train
        self._pool = pool
        self._budget = budget
        self._queried: list[int] = []

    @property
    def cost(self) -> int:
        return len(self._queried)

    def view(self) -> RecruitmentView:
        ids = np.asarray(self._queried, dtype=np.int64)
        observed = _sample_rows(self._pool, ids, "rct_train") if len(ids) else None
        return RecruitmentView(self._obs, self._pool.X, ids, observed, self._budget-self.cost)

    def reveal(self, indices: Any) -> SourceSample:
        ids = _indices(indices, size=self._pool.n)
        if len(ids) == 0 or self.cost + len(ids) > self._budget:
            raise ValueError("query must be nonempty and stay within the outcome budget")
        if set(ids.tolist()) & set(self._queried):
            raise ValueError("a trial outcome cannot be queried twice")
        self._queried.extend(ids.tolist())
        return _sample_rows(self._pool, ids, "rct_train")


@dataclass(frozen=True)
class UniformPolicy:
    def select(self, view: RecruitmentView, n: int, *, seed: int) -> IntArray:
        return np.random.default_rng(seed).permutation(view.available_indices)[:n]


def _pilot_distances(view: RecruitmentView) -> tuple[IntArray, NDArray[np.float64]]:
    if view.revealed is None:
        raise ValueError("geometry and outcome policies require a revealed pilot")
    ids = view.available_indices
    scale = np.maximum(view.pool_X.std(axis=0), 1e-8)
    # A bounded candidate-by-pilot matrix, with no candidate outcomes.
    difference = (view.pool_X[ids, None, :] - view.revealed.X[None, :, :]) / scale
    return ids, np.mean(difference*difference, axis=2)


def _rank(ids: IntArray, scores: NDArray[np.float64], n: int, seed: int) -> IntArray:
    if not np.isfinite(scores).all():
        raise ValueError("recruitment priorities must be finite")
    tie_order = np.random.default_rng(seed).permutation(len(ids))
    order = tie_order[np.argsort(-scores[tie_order], kind="stable")]
    return np.asarray(ids[order[:n]], dtype=np.int64)


@dataclass(frozen=True)
class GeometryPolicy:
    def select(self, view: RecruitmentView, n: int, *, seed: int) -> IntArray:
        ids, distances = _pilot_distances(view)
        return _rank(ids, distances.min(axis=1), n, seed)


@dataclass(frozen=True)
class OutcomeResidualPolicy:
    """Local squared randomized-pseudo-outcome/OBS disagreement heuristic."""

    def select(self, view: RecruitmentView, n: int, *, seed: int) -> IntArray:
        ids, distances = _pilot_distances(view)
        pilot = view.revealed
        assert pilot is not None
        # TLearner requires a TrainingView; both trial roles below contain only
        # revealed pilot observations. Its source='obs' recipe ignores them.
        visible = TrainingView(view.obs_train, pilot,
                               _sample_rows(pilot, np.arange(pilot.n), "rct_tune"))
        obs_tau = TLearner(source="obs", learner="ridge").fit_predict(
            visible, pilot.X, seed=seed)
        p = pilot.known_propensity
        assert p is not None
        pseudo = pilot.T*pilot.Y/p - (1-pilot.T)*pilot.Y/(1-p)
        residual2 = (pseudo-obs_tau)**2
        k = min(8, pilot.n)
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
        weights = 1/(0.05+np.take_along_axis(distances, nearest, axis=1))
        local = (weights*residual2[nearest]).sum(axis=1)/weights.sum(axis=1)
        geometry = distances.min(axis=1)
        geometry = geometry/(1+geometry)
        priority = local + 0.05*max(float(np.mean(residual2)), 1e-12)*geometry
        return _rank(ids, priority, n, seed)


def make_policy(name: str) -> RecruitmentPolicy:
    try:
        return {"uniform": UniformPolicy, "geometry": GeometryPolicy,
                "outcome_residual": OutcomeResidualPolicy}[name]()
    except KeyError as exc:
        raise ValueError("unknown recruitment policy") from exc


def recruit(problem: SimulationProblem, policy: RecruitmentPolicy, *, budget: int,
            pilot_size: int, seed: int) -> IntArray:
    """Buy a common uniform pilot, then an ordered outcome-adaptive batch."""
    _positive_integer(pilot_size, "pilot_size")
    if pilot_size >= budget:
        raise ValueError("pilot must be smaller than the final budget")
    session = RecruitmentSession(problem.obs_train, problem.rct_train, budget)
    # Policy RNGs differ; the pilot uses the independent data replicate seed.
    pilot_seed = method_seed(problem.name, int(problem.seed), "adaptive_common_pilot")
    pilot = np.random.default_rng(pilot_seed).permutation(problem.rct_train.n)[:pilot_size]
    session.reveal(pilot)
    n = budget-pilot_size
    try:
        selected = _indices(policy.select(session.view(), n, seed=seed),
                            size=problem.rct_train.n, n=n)
        session.reveal(selected)
    except Exception as exc:
        raise RecruitmentError(type(exc).__name__, session.view().revealed_indices) from exc
    return session.view().revealed_indices


def acquired_training(problem: SimulationProblem, indices: Any, *, tune_size: int) -> TrainingView:
    ids = _indices(indices, size=problem.rct_train.n)
    _positive_integer(tune_size, "tune_size")
    if tune_size >= len(ids):
        raise ValueError("charged tuning sample must leave trial training rows")
    return TrainingView(problem.obs_train,
                        _sample_rows(problem.rct_train, ids[tune_size:], "rct_train"),
                        _sample_rows(problem.rct_train, ids[:tune_size], "rct_tune"),
                        problem.target_reference)


def run_adaptive_problem(problem: SimulationProblem, *, budgets: Sequence[int],
                         pilot_size: int, tune_size: int,
                         policies: Sequence[str], estimators: Sequence[str],
                         seed: int, method_options: Mapping[str, Mapping[str, Any]] | None = None,
                         scenario: str | None = None,
                         requested_cells: set[tuple[str, int, str]] | None = None) -> list[dict[str, Any]]:
    """Evaluate fixed policy prefixes; return every success or terminal failure."""
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be nonempty and unique")
    for budget in budgets:
        _positive_integer(budget, "budget")
    _positive_integer(pilot_size, "pilot_size")
    _positive_integer(tune_size, "tune_size")
    if tune_size >= pilot_size or pilot_size >= min(budgets) or max(budgets) > problem.rct_train.n:
        raise ValueError("require tune_size < pilot_size < every budget <= pool size")
    if not policies or len(set(policies)) != len(policies) or any(x not in POLICY_NAMES for x in policies):
        raise ValueError("policies must be distinct registered policy names")
    if not estimators or len(set(estimators)) != len(estimators):
        raise ValueError("estimators must be nonempty and unique")
    options = method_options or {}
    scenario = problem.name if scenario is None else scenario
    rows = []
    for policy in policies:
        if requested_cells is not None and not any(cell[0] == policy for cell in requested_cells):
            continue
        policy_seed = method_seed(scenario, seed, "recruitment_"+policy)
        try:
            order = recruit(problem, make_policy(policy), budget=max(budgets),
                            pilot_size=pilot_size, seed=policy_seed)
            acquisition_error = None
        except Exception as exc:
            order = None
            acquisition_error = exc.error_type if isinstance(exc, RecruitmentError) else type(exc).__name__
            partial = exc.query_indices if isinstance(exc, RecruitmentError) else np.empty(0, dtype=np.int64)
        for budget in budgets:
            for estimator in estimators:
                if requested_cells is not None and (policy, budget, estimator) not in requested_cells:
                    continue
                fit_seed = method_seed(scenario+f"/budget_{budget}", seed, "fit_"+estimator)
                row = {"scenario": scenario+f"/budget_{budget}", "seed": seed,
                       "method": policy+"__"+estimator, "policy": policy,
                       "estimator": estimator, "budget": budget,
                       "policy_seed": policy_seed, "fit_seed": fit_seed,
                       "planned_pilot_size": pilot_size, "planned_tune_size": tune_size}
                if order is None:
                    spent = len(partial)
                    row.update(status="failed", stage="recruitment", error_type=acquisition_error,
                               metrics=None, query_indices=partial.tolist(), query_cost=spent,
                               pilot_cost=min(spent, pilot_size), tune_cost=min(spent, tune_size),
                               train_cost=max(0, spent-tune_size))
                else:
                    indices = order[:budget]
                    row.update(query_indices=indices.tolist(), query_cost=len(indices),
                               pilot_cost=pilot_size, tune_cost=tune_size,
                               train_cost=budget-tune_size)
                    try:
                        training = acquired_training(problem, indices, tune_size=tune_size)
                        model = make_method(estimator, **dict(options.get(estimator, {})))
                        prediction = model.fit_predict(training, problem.eval_X, seed=fit_seed)
                        row.update(status="ok", metrics=cate_metrics(prediction, problem.eval_tau))
                    except Exception as exc:
                        row.update(status="failed", stage="method_or_metric", error_type=type(exc).__name__, metrics=None)
                rows.append(row)
    return rows


def replay_adaptive_records(problem: SimulationProblem, records: Sequence[Mapping[str, Any]], *,
                             pilot_size: int, tune_size: int,
                             method_options: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """Reconstruct policy decisions, budgets, fits and scores from saved queries.

    This is deterministic replay, not an independent empirical replication.
    Independent hand calculations and information-boundary tests accompany it.
    """
    options = method_options or {}
    for row in records:
        if row.get("stage") == "recruitment" and row["status"] != "failed":
            raise ValueError("recruitment-stage records must be failed cells")
        scenario = row["scenario"].rsplit("/budget_", 1)[0]
        if (row["seed"] != problem.seed
                or row["policy_seed"] != method_seed(scenario, row["seed"], "recruitment_"+row["policy"])
                or row["fit_seed"] != method_seed(row["scenario"], row["seed"], "fit_"+row["estimator"])):
            raise ValueError("recorded replicate or method seed does not match its identity")
        budget = row["budget"]
        ids = _indices(row["query_indices"], size=problem.rct_train.n)
        spent = len(ids)
        if (spent > budget or row["query_cost"] != spent
                or row["pilot_cost"] != min(spent, pilot_size)
                or row["tune_cost"] != min(spent, tune_size)
                or row["train_cost"]+row["tune_cost"] != spent):
            raise ValueError("recorded acquisition costs do not match the budget")
        if row.get("stage") == "simulation":
            if spent != 0 or row["status"] != "failed":
                raise ValueError("simulation failures cannot have queried outcomes")
            continue
        if row.get("stage") != "recruitment" and spent != budget:
            raise ValueError("completed acquisition must spend its declared budget")
        if row["status"] != "ok":
            continue
        training = acquired_training(problem, ids, tune_size=tune_size)
        model = make_method(row["estimator"], **dict(options.get(row["estimator"], {})))
        prediction = model.fit_predict(training, problem.eval_X, seed=row["fit_seed"])
        metrics = cate_metrics(prediction, problem.eval_tau)
        if metrics != row["metrics"]:
            raise ValueError("replayed predictions do not reproduce recorded metrics")
    # Compare all stored prefixes to the longest recorded successful order for
    # each policy. A caller regenerates problems from the bound seed/config.
    for policy in {row["policy"] for row in records if row.get("stage") != "simulation"}:
        group = [row for row in records if row["policy"] == policy and row.get("stage") != "simulation"]
        maximum = max(row["budget"] for row in group)
        try:
            expected = recruit(problem, make_policy(policy), budget=maximum,
                               pilot_size=pilot_size, seed=group[0]["policy_seed"])
            recruitment_failed = False
        except RecruitmentError as exc:
            expected = exc.query_indices
            recruitment_failed = True
        for row in group:
            if ((row.get("stage") == "recruitment") != recruitment_failed
                    or row["query_indices"] != expected[:row["budget"]].tolist()):
                raise ValueError("replayed recruitment does not reproduce the recorded prefix")
