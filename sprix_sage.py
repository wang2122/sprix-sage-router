"""Sprix SAGE: progress-aware routing for open agent-to-agent networks.

The module intentionally has no third-party runtime dependencies.  It combines
hard authorization constraints with contextual online learning, task-DAG role
assignment, and bounded beam search over candidate teams.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import combinations
import math
import random
from typing import Iterable, Mapping


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class Mode(str, Enum):
    SELF = "self"
    COLLABORATE = "collaborate"
    HANDOFF = "handoff"


@dataclass(frozen=True)
class Requirement:
    name: str
    weight: float = 1.0
    minimum: float = 0.55
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("requirement name must not be empty")
        if self.weight <= 0:
            raise ValueError("requirement weight must be positive")
        if not 0 <= self.minimum <= 1:
            raise ValueError("requirement minimum must be in [0, 1]")
        if self.name in self.depends_on:
            raise ValueError("a requirement cannot depend on itself")


@dataclass(frozen=True)
class Task:
    task_id: str
    requirements: tuple[Requirement, ...]
    value: float = 1.0
    budget: float = math.inf
    deadline_ms: float = math.inf
    required_permissions: frozenset[str] = frozenset()
    risk_tolerance: float = 0.5
    progress: float = 0.0
    handoff_friction: float = 0.25
    coordination_overhead: float = 0.06
    context_transferability: float = 0.70
    replan_friction: float = 0.03

    def __post_init__(self) -> None:
        if not self.requirements:
            raise ValueError("task must have at least one requirement")
        if self.value <= 0 or self.budget <= 0 or self.deadline_ms <= 0:
            raise ValueError("value, budget, and deadline must be positive")
        for value, label in (
            (self.risk_tolerance, "risk_tolerance"),
            (self.progress, "progress"),
            (self.context_transferability, "context_transferability"),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be in [0, 1]")
        if self.handoff_friction < 0 or self.coordination_overhead < 0 or self.replan_friction < 0:
            raise ValueError("routing friction and overhead values must be non-negative")
        self._validate_dag()

    def _validate_dag(self) -> None:
        names = [item.name for item in self.requirements]
        if len(names) != len(set(names)):
            raise ValueError("requirement names must be unique")
        name_set = set(names)
        for item in self.requirements:
            unknown = set(item.depends_on) - name_set
            if unknown:
                raise ValueError(f"unknown requirement dependencies: {sorted(unknown)}")

        graph = {item.name: item.depends_on for item in self.requirements}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError("requirement dependencies must form a DAG")
            if name in visited:
                return
            visiting.add(name)
            for dependency in graph[name]:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in names:
            visit(name)


@dataclass(frozen=True)
class ExecutionState:
    """Live state used when routing or replanning an in-flight task."""

    active_agents: tuple[str, ...] = ()
    active_mode: Mode = Mode.SELF
    completed_requirements: frozenset[str] = frozenset()
    progress: float | None = None
    transferable_context: float | None = None
    failed_agents: frozenset[str] = frozenset()
    failure_count: int = 0

    def __post_init__(self) -> None:
        if self.progress is not None and not 0 <= self.progress <= 1:
            raise ValueError("state progress must be in [0, 1]")
        if self.transferable_context is not None and not 0 <= self.transferable_context <= 1:
            raise ValueError("transferable_context must be in [0, 1]")
        if self.failure_count < 0:
            raise ValueError("failure_count must be non-negative")


@dataclass(frozen=True)
class Agent:
    agent_id: str
    skills: Mapping[str, float]
    cost: float
    latency_ms: float
    permissions: frozenset[str] = frozenset()
    availability: float = 1.0
    load: float = 0.0

    def __post_init__(self) -> None:
        if self.cost < 0 or self.latency_ms < 0:
            raise ValueError("cost and latency must be non-negative")
        if not 0 <= self.availability <= 1 or not 0 <= self.load <= 1:
            raise ValueError("availability and load must be in [0, 1]")
        if any(not 0 <= score <= 1 for score in self.skills.values()):
            raise ValueError("skill scores must be in [0, 1]")


@dataclass(frozen=True)
class Bid:
    agent_id: str
    task_id: str
    quoted_cost: float
    promised_latency_ms: float
    confidence: float = 0.7

    def __post_init__(self) -> None:
        if self.quoted_cost < 0 or self.promised_latency_ms < 0:
            raise ValueError("bid cost and latency must be non-negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("bid confidence must be in [0, 1]")


@dataclass
class BetaBelief:
    alpha: float = 2.0
    beta: float = 2.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        total = self.alpha + self.beta
        return math.sqrt((self.alpha * self.beta) / (total * total * (total + 1)))

    def draw(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def update(self, score: float | bool, weight: float = 1.0) -> None:
        value = float(score)
        if not 0 <= value <= 1:
            raise ValueError("belief update score must be in [0, 1]")
        if weight <= 0:
            raise ValueError("belief update weight must be positive")
        self.alpha += weight * value
        self.beta += weight * (1.0 - value)


@dataclass
class OnlineSuccessModel:
    """Small online logistic model updated from execution outcomes."""

    learning_rate: float = 0.08
    l2: float = 0.001
    updates: int = 0
    bias: float = -1.15
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "coverage": 2.35,
            "bottleneck": 1.35,
            "trust": 0.80,
            "synergy": 0.35,
            "redundancy": -0.45,
            "coordination_loss": -0.55,
            "handoff_loss": -0.70,
            "switch_loss": -0.55,
            "load": -0.35,
        }
    )

    def predict(self, features: Mapping[str, float]) -> float:
        logit = self.bias + sum(self.weights.get(name, 0.0) * value for name, value in features.items())
        return _clip(_sigmoid(logit), 0.01, 0.99)

    def update(self, features: Mapping[str, float], outcome: float) -> None:
        prediction = self.predict(features)
        error = _clip(outcome) - prediction
        rate = self.learning_rate / math.sqrt(1.0 + self.updates / 50.0)
        self.bias += rate * error
        for name in self.weights:
            value = features.get(name, 0.0)
            self.weights[name] += rate * (error * value - self.l2 * self.weights[name])
        self.updates += 1


@dataclass(frozen=True)
class RouterWeights:
    cost: float = 0.18
    latency: float = 0.10
    risk: float = 0.12
    handoff: float = 0.22
    coordination: float = 0.08
    uncertainty: float = 0.05
    exploration: float = 0.08


@dataclass(frozen=True)
class RouteDecision:
    mode: Mode
    agents: tuple[str, ...]
    utility: float
    success_probability: float
    coverage: float
    cost: float
    latency_ms: float
    risk: float
    explanation: str
    assignments: Mapping[str, str] = field(default_factory=dict)
    topology: tuple[tuple[str, str], ...] = ()
    switch_recommended: bool = False
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    model_features: Mapping[str, float] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ExecutionOutcome:
    """Observed evidence used for contextual and bid-calibration updates."""

    success: float | bool
    agent_scores: Mapping[str, float] = field(default_factory=dict)
    requirement_scores: Mapping[str, float] = field(default_factory=dict)
    actual_cost: float | None = None
    actual_latency_ms: float | None = None

    def __post_init__(self) -> None:
        values = [float(self.success), *self.agent_scores.values(), *self.requirement_scores.values()]
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("outcome scores must be in [0, 1]")
        if self.actual_cost is not None and self.actual_cost < 0:
            raise ValueError("actual_cost must be non-negative")
        if self.actual_latency_ms is not None and self.actual_latency_ms < 0:
            raise ValueError("actual_latency_ms must be non-negative")


class SAGERouter:
    """Contextual, progress-aware agent router with bounded team search."""

    def __init__(
        self,
        agents: Iterable[Agent],
        incumbent_id: str,
        *,
        weights: RouterWeights | None = None,
        max_collaborators: int = 2,
        beam_width: int = 8,
        exploration: bool = False,
        seed: int = 7,
    ) -> None:
        self.agents = {agent.agent_id: agent for agent in agents}
        if incumbent_id not in self.agents:
            raise ValueError("incumbent_id must identify a registered agent")
        if max_collaborators < 0 or beam_width <= 0:
            raise ValueError("max_collaborators must be non-negative and beam_width positive")
        self.incumbent_id = incumbent_id
        self.weights = weights or RouterWeights()
        self.max_collaborators = max_collaborators
        self.beam_width = beam_width
        self.exploration = exploration
        self.rng = random.Random(seed)
        self.reliability = {agent_id: BetaBelief() for agent_id in self.agents}
        self.skill_reliability: dict[tuple[str, str], BetaBelief] = {}
        self.synergy: dict[tuple[str, str], BetaBelief] = {}
        self.cost_fidelity = {agent_id: BetaBelief() for agent_id in self.agents}
        self.latency_fidelity = {agent_id: BetaBelief() for agent_id in self.agents}
        self.success_model = OnlineSuccessModel()
        self._draw_cache: dict[tuple[str, ...], float] = {}

    def route(
        self,
        task: Task,
        bids: Iterable[Bid] | None = None,
        state: ExecutionState | None = None,
    ) -> RouteDecision:
        state = state or ExecutionState()
        self._validate_state(task, state)
        self._draw_cache = {}
        bid_map = self._prepare_bids(task, bids)
        eligible = [
            agent_id
            for agent_id, agent in self.agents.items()
            if agent_id not in state.failed_agents and self._eligible(agent, task, bid_map[agent_id])
        ]
        if not eligible:
            raise RuntimeError("no eligible agent satisfies permissions, budget, and deadline")

        decisions: list[RouteDecision] = []
        if self.incumbent_id in eligible:
            self_decision = self._evaluate(Mode.SELF, (self.incumbent_id,), task, bid_map, state)
            if self._team_feasible(self_decision, task):
                decisions.append(self_decision)
            decisions.extend(self._beam_collaboration_decisions(task, eligible, bid_map, state))

        for agent_id in eligible:
            if agent_id == self.incumbent_id:
                continue
            decision = self._evaluate(Mode.HANDOFF, (agent_id,), task, bid_map, state)
            if self._team_feasible(decision, task):
                decisions.append(decision)

        if not decisions:
            raise RuntimeError("no feasible route satisfies team-level budget and deadline constraints")

        best = max(decisions, key=lambda decision: decision.utility)
        active = tuple(state.active_agents)
        switched = bool(active) and (best.mode != state.active_mode or set(best.agents) != set(active))
        return replace(best, switch_recommended=switched)

    def record_outcome(
        self,
        decision: RouteDecision,
        outcome: ExecutionOutcome | float | bool,
    ) -> None:
        evidence = outcome if isinstance(outcome, ExecutionOutcome) else ExecutionOutcome(outcome)
        unknown_agents = set(evidence.agent_scores) - set(decision.agents)
        unknown_requirements = set(evidence.requirement_scores) - set(decision.assignments)
        if unknown_agents:
            raise ValueError(f"outcome contains unselected agents: {sorted(unknown_agents)}")
        if unknown_requirements:
            raise ValueError(f"outcome contains unknown requirements: {sorted(unknown_requirements)}")
        overall = _clip(float(evidence.success))
        self.success_model.update(decision.model_features, overall)

        assigned_by_agent: dict[str, list[str]] = {agent_id: [] for agent_id in decision.agents}
        for requirement, agent_id in decision.assignments.items():
            assigned_by_agent.setdefault(agent_id, []).append(requirement)

        attributed_scores: dict[str, float] = {}
        for agent_id in decision.agents:
            explicit = evidence.agent_scores.get(agent_id)
            assigned_scores = [
                evidence.requirement_scores[name]
                for name in assigned_by_agent.get(agent_id, [])
                if name in evidence.requirement_scores
            ]
            if explicit is not None:
                credit, weight = explicit, 1.0
            elif assigned_scores:
                credit, weight = sum(assigned_scores) / len(assigned_scores), 0.85
            else:
                credit, weight = overall, 0.35
            attributed_scores[agent_id] = _clip(credit)
            self.reliability[agent_id].update(credit, weight)

            for requirement in assigned_by_agent.get(agent_id, []):
                skill_score = evidence.requirement_scores.get(requirement, credit)
                skill_weight = 1.0 if requirement in evidence.requirement_scores else weight
                self._skill_belief(agent_id, requirement).update(skill_score, skill_weight)

        for left, right in combinations(sorted(decision.agents), 2):
            if evidence.agent_scores or evidence.requirement_scores:
                individual_mean = (attributed_scores[left] + attributed_scores[right]) / 2.0
                pair_credit = _clip(0.5 + overall - individual_mean)
                pair_weight = 0.70
            else:
                # A team-only outcome is weak evidence: preserve backward compatibility
                # while avoiding the old full-credit update for every pair.
                pair_credit = overall
                pair_weight = 0.25
            self._synergy_belief(left, right).update(pair_credit, pair_weight)

        quoted_cost = decision.cost
        if evidence.actual_cost is not None and quoted_cost > 0:
            fidelity = _clip(1.0 - max(0.0, evidence.actual_cost / quoted_cost - 1.0))
            for agent_id in decision.agents:
                self.cost_fidelity[agent_id].update(fidelity)
        if evidence.actual_latency_ms is not None and decision.latency_ms > 0:
            fidelity = _clip(1.0 - max(0.0, evidence.actual_latency_ms / decision.latency_ms - 1.0))
            for agent_id in decision.agents:
                self.latency_fidelity[agent_id].update(fidelity)

    def _validate_state(self, task: Task, state: ExecutionState) -> None:
        requirement_names = {item.name for item in task.requirements}
        unknown_requirements = set(state.completed_requirements) - requirement_names
        unknown_agents = (set(state.active_agents) | set(state.failed_agents)) - set(self.agents)
        if unknown_requirements:
            raise ValueError(f"unknown completed requirements: {sorted(unknown_requirements)}")
        if unknown_agents:
            raise ValueError(f"unknown agents in execution state: {sorted(unknown_agents)}")
        if state.completed_requirements == requirement_names:
            raise RuntimeError("task is already complete")

    def _prepare_bids(self, task: Task, bids: Iterable[Bid] | None) -> dict[str, Bid]:
        supplied = {bid.agent_id: bid for bid in bids or () if bid.task_id == task.task_id}
        return {
            agent_id: supplied.get(
                agent_id,
                Bid(agent_id, task.task_id, agent.cost, agent.latency_ms, confidence=0.7),
            )
            for agent_id, agent in self.agents.items()
        }

    def _risk_adjusted_cost(self, agent_id: str, bid: Bid) -> float:
        return bid.quoted_cost * (1.0 + 0.20 * (1.0 - self.cost_fidelity[agent_id].mean))

    def _risk_adjusted_latency(self, agent_id: str, bid: Bid) -> float:
        agent = self.agents[agent_id]
        quote = bid.promised_latency_ms * (1.0 + 0.20 * (1.0 - self.latency_fidelity[agent_id].mean))
        return quote * (1.0 + 0.50 * agent.load) / max(agent.availability, 0.10)

    def _eligible(self, agent: Agent, task: Task, bid: Bid) -> bool:
        return (
            agent.availability > 0
            and task.required_permissions.issubset(agent.permissions)
            and self._risk_adjusted_cost(agent.agent_id, bid) <= task.budget
            and self._risk_adjusted_latency(agent.agent_id, bid) <= task.deadline_ms
        )

    def _belief_value(self, key: tuple[str, ...], belief: BetaBelief) -> float:
        if not self.exploration:
            return belief.mean
        if key not in self._draw_cache:
            self._draw_cache[key] = belief.draw(self.rng)
        return self._draw_cache[key]

    def _skill_belief(self, agent_id: str, requirement: str) -> BetaBelief:
        return self.skill_reliability.setdefault((agent_id, requirement), BetaBelief())

    def _synergy_belief(self, left: str, right: str) -> BetaBelief:
        pair = tuple(sorted((left, right)))
        return self.synergy.setdefault(pair, BetaBelief())

    def _contextual_trust(self, agent_id: str, requirement: str) -> tuple[float, float]:
        global_belief = self.reliability[agent_id]
        skill_belief = self._skill_belief(agent_id, requirement)
        global_value = self._belief_value(("global", agent_id), global_belief)
        skill_value = self._belief_value(("skill", agent_id, requirement), skill_belief)
        trust = 0.35 * global_value + 0.65 * skill_value
        uncertainty = 0.35 * global_belief.uncertainty + 0.65 * skill_belief.uncertainty
        return trust, uncertainty

    def _effective_skill(self, agent_id: str, requirement: str, bid: Bid) -> float:
        agent = self.agents[agent_id]
        declared = agent.skills.get(requirement, 0.0)
        trust, _ = self._contextual_trust(agent_id, requirement)
        calibrated_bid = trust * bid.confidence + (1.0 - trust) * 0.5
        return declared * (0.65 + 0.35 * trust) * (0.70 + 0.30 * calibrated_bid)

    def _remaining_requirements(self, task: Task, state: ExecutionState) -> tuple[Requirement, ...]:
        return tuple(item for item in task.requirements if item.name not in state.completed_requirements)

    def _coverage_and_assignment(
        self,
        team: tuple[str, ...],
        task: Task,
        bids: Mapping[str, Bid],
        state: ExecutionState,
    ) -> tuple[float, float, dict[str, str], float, float]:
        requirements = self._remaining_requirements(task, state)
        total_weight = sum(item.weight for item in requirements)
        weighted = 0.0
        bottlenecks: list[float] = []
        assignments: dict[str, str] = {}
        trust_total = uncertainty_total = 0.0
        for requirement in requirements:
            skills = {
                agent_id: self._effective_skill(agent_id, requirement.name, bids[agent_id])
                for agent_id in team
            }
            miss = math.prod(1.0 - score for score in skills.values())
            coverage = 1.0 - miss
            best_agent = max(skills, key=skills.get)
            best = skills[best_agent]
            assignments[requirement.name] = best_agent
            weighted += requirement.weight * coverage
            bottlenecks.append(min(1.0, best / max(requirement.minimum, 1e-9)))
            trust, uncertainty = self._contextual_trust(best_agent, requirement.name)
            trust_total += requirement.weight * trust
            uncertainty_total += requirement.weight * uncertainty
        return (
            weighted / total_weight,
            min(bottlenecks),
            assignments,
            trust_total / total_weight,
            uncertainty_total / total_weight,
        )

    def _cosine(self, left: str, right: str, requirements: tuple[Requirement, ...]) -> float:
        a = [self.agents[left].skills.get(item.name, 0.0) for item in requirements]
        b = [self.agents[right].skills.get(item.name, 0.0) for item in requirements]
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        return dot / norm if norm else 0.0

    def _team_terms(
        self,
        team: tuple[str, ...],
        requirements: tuple[Requirement, ...],
    ) -> tuple[float, float]:
        pairs = list(combinations(sorted(team), 2))
        if not pairs:
            return 0.0, 0.0
        synergy = sum(
            2 * self._belief_value(("pair", *pair), self._synergy_belief(*pair)) - 1 for pair in pairs
        )
        redundancy = sum(self._cosine(left, right, requirements) for left, right in pairs)
        return synergy / len(pairs), redundancy / len(pairs)

    def _topological_requirements(self, task: Task, state: ExecutionState) -> list[Requirement]:
        remaining = {item.name: item for item in self._remaining_requirements(task, state)}
        completed = set(state.completed_requirements)
        ordered: list[Requirement] = []
        emitted = set(completed)
        while remaining:
            ready = sorted(
                (item for item in remaining.values() if set(item.depends_on).issubset(emitted)),
                key=lambda item: item.name,
            )
            if not ready:
                raise RuntimeError("no executable requirement remains in task DAG")
            for item in ready:
                ordered.append(item)
                emitted.add(item.name)
                remaining.pop(item.name)
        return ordered

    def _schedule(
        self,
        assignments: Mapping[str, str],
        task: Task,
        bids: Mapping[str, Bid],
        state: ExecutionState,
    ) -> tuple[float, tuple[tuple[str, str], ...]]:
        requirements = self._remaining_requirements(task, state)
        total_weight = sum(item.weight for item in requirements)
        finish: dict[str, float] = {name: 0.0 for name in state.completed_requirements}
        agent_ready = {agent_id: 0.0 for agent_id in set(assignments.values())}
        topology: set[tuple[str, str]] = set()

        for item in self._topological_requirements(task, state):
            agent_id = assignments[item.name]
            dependency_ready = max((finish[name] for name in item.depends_on), default=0.0)
            start = max(dependency_ready, agent_ready[agent_id])
            duration = self._risk_adjusted_latency(agent_id, bids[agent_id]) * item.weight / total_weight
            finish[item.name] = start + duration
            agent_ready[agent_id] = finish[item.name]
            for dependency in item.depends_on:
                dependency_agent = assignments.get(dependency)
                if dependency_agent and dependency_agent != agent_id:
                    topology.add((dependency_agent, agent_id))

        used_agents = set(assignments.values())
        coordinator = self.incumbent_id if self.incumbent_id in used_agents else min(used_agents)
        connected = {node for edge in topology for node in edge}
        for agent_id in sorted(used_agents - {coordinator} - connected):
            topology.add((coordinator, agent_id))

        latency = max(finish.values(), default=0.0)
        cross_agent_edges = len(topology)
        latency *= 1.0 + task.coordination_overhead * cross_agent_edges
        return latency, tuple(sorted(topology))

    def _switch_loss(self, mode: Mode, team: tuple[str, ...], task: Task, state: ExecutionState) -> float:
        if not state.active_agents:
            if mode is Mode.HANDOFF:
                progress = task.progress if state.progress is None else state.progress
                transferability = (
                    task.context_transferability
                    if state.transferable_context is None
                    else state.transferable_context
                )
                return progress * task.handoff_friction * (1.0 - 0.65 * transferability)
            return 0.0
        if mode == state.active_mode and set(team) == set(state.active_agents):
            return 0.0
        union = set(team) | set(state.active_agents)
        retained = len(set(team) & set(state.active_agents)) / max(1, len(union))
        progress = task.progress if state.progress is None else state.progress
        transferability = task.context_transferability if state.transferable_context is None else state.transferable_context
        context_loss = progress * (1.0 - transferability) * (1.0 - retained)
        recovery_discount = 1.0 / (1.0 + state.failure_count)
        return (task.replan_friction + task.handoff_friction * context_loss) * recovery_discount

    def _evaluate(
        self,
        mode: Mode,
        team: tuple[str, ...],
        task: Task,
        bids: Mapping[str, Bid],
        state: ExecutionState,
    ) -> RouteDecision:
        requirements = self._remaining_requirements(task, state)
        coverage, bottleneck, assignments, trust, uncertainty = self._coverage_and_assignment(
            team, task, bids, state
        )
        synergy, redundancy = self._team_terms(team, requirements)
        latency, topology = self._schedule(assignments, task, bids, state)
        cost = sum(self._risk_adjusted_cost(agent_id, bids[agent_id]) for agent_id in team)
        coordination_loss = task.coordination_overhead * len(topology) if mode is Mode.COLLABORATE else 0.0
        switch_loss = self._switch_loss(mode, team, task, state)
        handoff_loss = switch_loss if mode is Mode.HANDOFF else 0.0
        mean_load = sum(self.agents[agent_id].load for agent_id in team) / len(team)
        features = {
            "coverage": coverage,
            "bottleneck": bottleneck,
            "trust": trust,
            "synergy": synergy,
            "redundancy": redundancy,
            "coordination_loss": coordination_loss,
            "handoff_loss": handoff_loss,
            "switch_loss": switch_loss,
            "load": mean_load,
        }
        probability = self.success_model.predict(features)
        risk = 1.0 - trust
        normalized_cost = cost / task.budget if math.isfinite(task.budget) else cost / task.value
        normalized_latency = latency / task.deadline_ms if math.isfinite(task.deadline_ms) else latency / 10_000
        exploration_bonus = self.weights.exploration * uncertainty if self.exploration else 0.0
        utility = (
            task.value * probability
            - self.weights.cost * normalized_cost
            - self.weights.latency * normalized_latency
            - self.weights.risk * risk * (1.0 - task.risk_tolerance / 2.0)
            - self.weights.handoff * handoff_loss
            - self.weights.coordination * coordination_loss
            - self.weights.uncertainty * uncertainty
            + exploration_bonus
        )
        explanation = self._explain(mode, team, probability, coverage, switch_loss, assignments)
        return RouteDecision(
            mode=mode,
            agents=team,
            utility=utility,
            success_probability=probability,
            coverage=coverage,
            cost=cost,
            latency_ms=latency,
            risk=risk,
            explanation=explanation,
            assignments=assignments,
            topology=topology,
            diagnostics={
                "bottleneck": bottleneck,
                "synergy": synergy,
                "redundancy": redundancy,
                "handoff_loss": handoff_loss,
                "coordination_loss": coordination_loss,
                "switch_loss": switch_loss,
                "uncertainty": uncertainty,
                "exploration_bonus": exploration_bonus,
                "model_updates": float(self.success_model.updates),
            },
            model_features=features,
        )

    @staticmethod
    def _team_feasible(decision: RouteDecision, task: Task) -> bool:
        return decision.cost <= task.budget and decision.latency_ms <= task.deadline_ms

    def _beam_collaboration_decisions(
        self,
        task: Task,
        eligible: list[str],
        bids: Mapping[str, Bid],
        state: ExecutionState,
    ) -> list[RouteDecision]:
        pool = sorted(agent_id for agent_id in eligible if agent_id != self.incumbent_id)
        frontier: list[tuple[str, ...]] = [(self.incumbent_id,)]
        decisions: dict[frozenset[str], RouteDecision] = {}
        for _ in range(self.max_collaborators):
            expanded: dict[frozenset[str], RouteDecision] = {}
            for team in frontier:
                for agent_id in pool:
                    if agent_id in team:
                        continue
                    candidate = team + (agent_id,)
                    quoted_cost = sum(self._risk_adjusted_cost(item, bids[item]) for item in candidate)
                    if quoted_cost > task.budget:
                        continue
                    key = frozenset(candidate)
                    if key in expanded:
                        continue
                    decision = self._evaluate(Mode.COLLABORATE, candidate, task, bids, state)
                    expanded[key] = decision
                    if self._team_feasible(decision, task):
                        decisions[key] = decision
            ranked = sorted(expanded.values(), key=lambda item: item.utility, reverse=True)
            frontier = [item.agents for item in ranked[: self.beam_width]]
            if not frontier:
                break
        return list(decisions.values())

    @staticmethod
    def _explain(
        mode: Mode,
        team: tuple[str, ...],
        probability: float,
        coverage: float,
        switch_loss: float,
        assignments: Mapping[str, str],
    ) -> str:
        names = ", ".join(team)
        role_summary = ", ".join(f"{requirement}->{agent}" for requirement, agent in assignments.items())
        if mode is Mode.SELF:
            reason = "the incumbent has the highest constrained expected utility"
        elif mode is Mode.HANDOFF:
            reason = f"specialist gain exceeds context-transfer and switching loss ({switch_loss:.3f})"
        else:
            reason = "the selected team improves task-DAG coverage after coordination cost"
        return (
            f"{mode.value.upper()} via [{names}]: {reason}; estimated success={probability:.3f}, "
            f"coverage={coverage:.3f}; assignments: {role_summary}."
        )
