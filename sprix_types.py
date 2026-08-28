"""Validated public data types for the SAGE router."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


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
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("requirement weight must be a positive finite number")
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
        if not math.isfinite(self.value) or self.value <= 0:
            raise ValueError("value must be a positive finite number")
        if math.isnan(self.budget) or self.budget <= 0:
            raise ValueError("budget must be a positive number or infinity")
        if math.isnan(self.deadline_ms) or self.deadline_ms <= 0:
            raise ValueError("deadline must be a positive number or infinity")
        for value, label in (
            (self.risk_tolerance, "risk_tolerance"),
            (self.progress, "progress"),
            (self.context_transferability, "context_transferability"),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be in [0, 1]")
        if (
            self.handoff_friction < 0
            or self.coordination_overhead < 0
            or self.replan_friction < 0
        ):
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
    active_assignments: Mapping[str, str] = field(default_factory=dict)
    inflight_requirement: str | None = None
    inflight_progress: float = 0.0
    inflight_quality: float | None = None
    artifact_transferability: Mapping[str, float] = field(default_factory=dict)
    failed_agents: frozenset[str] = frozenset()
    failure_count: int = 0

    def __post_init__(self) -> None:
        if self.progress is not None and not 0 <= self.progress <= 1:
            raise ValueError("state progress must be in [0, 1]")
        if self.transferable_context is not None and not 0 <= self.transferable_context <= 1:
            raise ValueError("transferable_context must be in [0, 1]")
        if not 0 <= self.inflight_progress <= 1:
            raise ValueError("inflight_progress must be in [0, 1]")
        if self.inflight_quality is not None and not 0 <= self.inflight_quality <= 1:
            raise ValueError("inflight_quality must be in [0, 1]")
        if self.inflight_requirement is None and self.inflight_progress:
            raise ValueError("inflight_progress requires inflight_requirement")
        if self.inflight_requirement is None and self.inflight_quality is not None:
            raise ValueError("inflight_quality requires inflight_requirement")
        if any(not 0 <= value <= 1 for value in self.artifact_transferability.values()):
            raise ValueError("artifact transferability values must be in [0, 1]")
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
        if not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("cost must be a non-negative finite number")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency must be a non-negative finite number")
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
        if not self.quoted_cost >= 0 or not self.promised_latency_ms >= 0:
            raise ValueError("bid cost and latency must be non-negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("bid confidence must be in [0, 1]")


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
    feasible: bool = True
    constraint_violations: tuple[str, ...] = ()
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    model_features: Mapping[str, float] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serializable representation for audit logs."""

        return {
            "mode": self.mode.value,
            "agents": list(self.agents),
            "utility": self.utility,
            "success_probability": self.success_probability,
            "coverage": self.coverage,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "risk": self.risk,
            "explanation": self.explanation,
            "assignments": dict(self.assignments),
            "topology": [list(edge) for edge in self.topology],
            "switch_recommended": self.switch_recommended,
            "feasible": self.feasible,
            "constraint_violations": list(self.constraint_violations),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class RoutingTrace:
    """Inspectable result containing the winner and evaluated alternatives."""

    selected: RouteDecision
    alternatives: tuple[RouteDecision, ...]
    eligible_agents: tuple[str, ...]
    excluded_agents: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    prefiltered_agents: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": self.selected.to_dict(),
            "alternatives": [decision.to_dict() for decision in self.alternatives],
            "eligible_agents": list(self.eligible_agents),
            "excluded_agents": {
                agent_id: list(reasons) for agent_id, reasons in self.excluded_agents.items()
            },
            "prefiltered_agents": list(self.prefiltered_agents),
        }


@dataclass(frozen=True)
class ExecutionOutcome:
    """Observed evidence used for contextual and bid-calibration updates."""

    success: float | bool
    agent_scores: Mapping[str, float] = field(default_factory=dict)
    requirement_scores: Mapping[str, float] = field(default_factory=dict)
    pair_scores: Mapping[tuple[str, str], float] = field(default_factory=dict)
    actual_cost: float | None = None
    actual_latency_ms: float | None = None

    def __post_init__(self) -> None:
        values = [
            float(self.success),
            *self.agent_scores.values(),
            *self.requirement_scores.values(),
            *self.pair_scores.values(),
        ]
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("outcome scores must be in [0, 1]")
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in self.pair_scores):
            raise ValueError("pair_scores keys must identify two distinct agents")
        if self.actual_cost is not None and self.actual_cost < 0:
            raise ValueError("actual_cost must be non-negative")
        if self.actual_latency_ms is not None and self.actual_latency_ms < 0:
            raise ValueError("actual_latency_ms must be non-negative")
