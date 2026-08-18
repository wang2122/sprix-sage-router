"""Sprix SAGE: interpretable SELF/COLLABORATE/HANDOFF routing for A2A agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
import math
import random
from typing import Iterable, Mapping


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class Mode(str, Enum):
    SELF = "self"
    COLLABORATE = "collaborate"
    HANDOFF = "handoff"


@dataclass(frozen=True)
class Requirement:
    name: str
    weight: float = 1.0
    minimum: float = 0.55

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("requirement weight must be positive")
        if not 0 <= self.minimum <= 1:
            raise ValueError("requirement minimum must be in [0, 1]")


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

    def __post_init__(self) -> None:
        if not self.requirements:
            raise ValueError("task must have at least one requirement")
        if self.value <= 0 or self.budget <= 0 or self.deadline_ms <= 0:
            raise ValueError("value, budget, and deadline must be positive")
        if not 0 <= self.progress <= 1:
            raise ValueError("progress must be in [0, 1]")


@dataclass(frozen=True)
class Agent:
    agent_id: str
    skills: Mapping[str, float]
    cost: float
    latency_ms: float
    permissions: frozenset[str] = frozenset()
    availability: float = 1.0

    def __post_init__(self) -> None:
        if self.cost < 0 or self.latency_ms < 0:
            raise ValueError("cost and latency must be non-negative")
        if not 0 <= self.availability <= 1:
            raise ValueError("availability must be in [0, 1]")
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

    def update(self, success: bool, weight: float = 1.0) -> None:
        if success:
            self.alpha += weight
        else:
            self.beta += weight


@dataclass(frozen=True)
class RouterWeights:
    cost: float = 0.18
    latency: float = 0.10
    risk: float = 0.12
    handoff: float = 0.22
    coordination: float = 0.08
    uncertainty: float = 0.05
    synergy: float = 0.12
    redundancy: float = 0.07


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
    diagnostics: Mapping[str, float] = field(default_factory=dict)


class SAGERouter:
    """State-aware, trust-calibrated A2A routing with online learning."""

    def __init__(
        self,
        agents: Iterable[Agent],
        incumbent_id: str,
        *,
        weights: RouterWeights | None = None,
        max_collaborators: int = 2,
        exploration: bool = False,
        seed: int = 7,
    ) -> None:
        self.agents = {agent.agent_id: agent for agent in agents}
        if incumbent_id not in self.agents:
            raise ValueError("incumbent_id must identify a registered agent")
        self.incumbent_id = incumbent_id
        self.weights = weights or RouterWeights()
        self.max_collaborators = max_collaborators
        self.exploration = exploration
        self.rng = random.Random(seed)
        self.reliability = {agent_id: BetaBelief() for agent_id in self.agents}
        self.synergy: dict[tuple[str, str], BetaBelief] = {}

    def route(self, task: Task, bids: Iterable[Bid] | None = None) -> RouteDecision:
        bid_map = self._prepare_bids(task, bids)
        eligible = [
            agent_id
            for agent_id in self.agents
            if self._eligible(self.agents[agent_id], task, bid_map[agent_id])
        ]
        if not eligible:
            raise RuntimeError("no eligible agent satisfies permissions, budget, and deadline")

        decisions: list[RouteDecision] = []
        if self.incumbent_id in eligible:
            decisions.append(self._evaluate(Mode.SELF, (self.incumbent_id,), task, bid_map))
            team = self._build_team(task, eligible, bid_map)
            if len(team) > 1:
                decisions.append(self._evaluate(Mode.COLLABORATE, team, task, bid_map))

        decisions.extend(
            self._evaluate(Mode.HANDOFF, (agent_id,), task, bid_map)
            for agent_id in eligible
            if agent_id != self.incumbent_id
        )
        if not decisions:
            raise RuntimeError("no feasible route could be constructed")
        return max(decisions, key=lambda decision: decision.utility)

    def record_outcome(self, decision: RouteDecision, success: bool) -> None:
        for agent_id in decision.agents:
            self.reliability[agent_id].update(success)
        for left, right in combinations(sorted(decision.agents), 2):
            self.synergy.setdefault((left, right), BetaBelief()).update(success)

    def _prepare_bids(self, task: Task, bids: Iterable[Bid] | None) -> dict[str, Bid]:
        supplied = {bid.agent_id: bid for bid in bids or () if bid.task_id == task.task_id}
        return {
            agent_id: supplied.get(
                agent_id,
                Bid(agent_id, task.task_id, agent.cost, agent.latency_ms, confidence=0.7),
            )
            for agent_id, agent in self.agents.items()
        }

    def _eligible(self, agent: Agent, task: Task, bid: Bid) -> bool:
        return (
            agent.availability > 0
            and task.required_permissions.issubset(agent.permissions)
            and bid.quoted_cost <= task.budget
            and bid.promised_latency_ms <= task.deadline_ms
        )

    def _belief_value(self, belief: BetaBelief) -> float:
        return belief.draw(self.rng) if self.exploration else belief.mean

    def _effective_skill(self, agent_id: str, requirement: str, bid: Bid) -> float:
        agent = self.agents[agent_id]
        declared = agent.skills.get(requirement, 0.0)
        trust = self._belief_value(self.reliability[agent_id])
        calibrated_bid = trust * bid.confidence + (1 - trust) * 0.5
        return declared * (0.5 + 0.5 * trust) * (0.5 + 0.5 * calibrated_bid)

    def _coverage(self, team: tuple[str, ...], task: Task, bids: Mapping[str, Bid]) -> tuple[float, float]:
        total_weight = sum(requirement.weight for requirement in task.requirements)
        weighted = 0.0
        bottlenecks: list[float] = []
        for requirement in task.requirements:
            miss = 1.0
            best = 0.0
            for agent_id in team:
                skill = self._effective_skill(agent_id, requirement.name, bids[agent_id])
                miss *= 1.0 - skill
                best = max(best, skill)
            coverage = 1.0 - miss
            weighted += requirement.weight * coverage
            threshold = max(requirement.minimum, 1e-9)
            bottlenecks.append(min(1.0, best / threshold))
        return weighted / total_weight, min(bottlenecks)

    def _cosine(self, left: str, right: str, requirements: tuple[Requirement, ...]) -> float:
        a = [self.agents[left].skills.get(item.name, 0.0) for item in requirements]
        b = [self.agents[right].skills.get(item.name, 0.0) for item in requirements]
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        return dot / norm if norm else 0.0

    def _team_terms(self, team: tuple[str, ...], task: Task) -> tuple[float, float]:
        pairs = list(combinations(sorted(team), 2))
        if not pairs:
            return 0.0, 0.0
        synergy = sum(2 * self._belief_value(self.synergy.setdefault(pair, BetaBelief())) - 1 for pair in pairs)
        redundancy = sum(self._cosine(left, right, task.requirements) for left, right in pairs)
        return synergy / len(pairs), redundancy / len(pairs)

    def _evaluate(
        self,
        mode: Mode,
        team: tuple[str, ...],
        task: Task,
        bids: Mapping[str, Bid],
    ) -> RouteDecision:
        coverage, bottleneck = self._coverage(team, task, bids)
        synergy, redundancy = self._team_terms(team, task)
        probability = _clip(
            coverage * (0.72 + 0.28 * bottleneck)
            + self.weights.synergy * synergy
            - self.weights.redundancy * redundancy
        )
        cost = sum(bids[agent_id].quoted_cost for agent_id in team)
        latency = max(bids[agent_id].promised_latency_ms for agent_id in team)
        if mode is Mode.COLLABORATE:
            latency *= 1 + task.coordination_overhead * (len(team) - 1)
        mean_trust = sum(self.reliability[agent_id].mean for agent_id in team) / len(team)
        mean_uncertainty = sum(self.reliability[agent_id].uncertainty for agent_id in team) / len(team)
        risk = 1.0 - mean_trust
        normalized_cost = cost / task.budget if math.isfinite(task.budget) else cost / task.value
        normalized_latency = latency / task.deadline_ms if math.isfinite(task.deadline_ms) else latency / 10_000
        handoff_loss = task.progress * task.handoff_friction if mode is Mode.HANDOFF else 0.0
        coordination_loss = task.coordination_overhead * (len(team) - 1) if mode is Mode.COLLABORATE else 0.0
        utility = (
            task.value * probability
            - self.weights.cost * normalized_cost
            - self.weights.latency * normalized_latency
            - self.weights.risk * risk * (1.0 - task.risk_tolerance / 2)
            - self.weights.handoff * handoff_loss
            - self.weights.coordination * coordination_loss
            - self.weights.uncertainty * mean_uncertainty
        )
        explanation = self._explain(mode, team, probability, coverage, handoff_loss, coordination_loss)
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
            diagnostics={
                "bottleneck": bottleneck,
                "synergy": synergy,
                "redundancy": redundancy,
                "handoff_loss": handoff_loss,
                "coordination_loss": coordination_loss,
                "uncertainty": mean_uncertainty,
            },
        )

    def _build_team(
        self,
        task: Task,
        eligible: list[str],
        bids: Mapping[str, Bid],
    ) -> tuple[str, ...]:
        team = (self.incumbent_id,)
        current = self._evaluate(Mode.COLLABORATE, team, task, bids).utility
        pool = [agent_id for agent_id in eligible if agent_id != self.incumbent_id]
        for _ in range(self.max_collaborators):
            affordable = [
                agent_id
                for agent_id in pool
                if sum(bids[item].quoted_cost for item in team) + bids[agent_id].quoted_cost <= task.budget
            ]
            if not affordable:
                break
            scored = [
                (self._evaluate(Mode.COLLABORATE, team + (agent_id,), task, bids).utility, agent_id)
                for agent_id in affordable
            ]
            best_utility, best_agent = max(scored)
            if best_utility <= current:
                break
            team += (best_agent,)
            pool.remove(best_agent)
            current = best_utility
        return team

    @staticmethod
    def _explain(
        mode: Mode,
        team: tuple[str, ...],
        probability: float,
        coverage: float,
        handoff_loss: float,
        coordination_loss: float,
    ) -> str:
        names = ", ".join(team)
        if mode is Mode.SELF:
            reason = "the incumbent's expected utility exceeds delegation alternatives"
        elif mode is Mode.HANDOFF:
            reason = f"specialist gain exceeds the progress-dependent handoff loss ({handoff_loss:.3f})"
        else:
            reason = f"complementary coverage exceeds coordination overhead ({coordination_loss:.3f})"
        return (
            f"{mode.value.upper()} via [{names}]: {reason}; "
            f"estimated success={probability:.3f}, requirement coverage={coverage:.3f}."
        )
