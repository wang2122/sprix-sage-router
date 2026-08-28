"""Checkpoint-aware held-out evaluator for mid-execution rerouting.

The evaluator scores concrete completed artifacts and remaining work.  It does
not call SAGE scoring helpers and never converts a scalar progress value into a
handoff penalty.  Switching waste instead follows from the portability of the
in-flight artifact and whether its producer is retained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

from benchmark_evaluator import HIDDEN_AGENTS, PAIR_COMPATIBILITY
from sprix_types import Agent, Mode, Requirement, Task


@dataclass(frozen=True)
class Artifact:
    """A completed requirement output available at a routing checkpoint."""

    producer: str
    quality: float
    portability: float


@dataclass(frozen=True)
class ExecutionCheckpoint:
    """Observable execution event plus evaluator-only artifact properties."""

    active_agents: tuple[str, ...]
    active_mode: Mode
    completed_artifacts: Mapping[str, Artifact]
    inflight_requirement: str
    inflight_agent: str
    inflight_fraction: float
    inflight_quality: float
    inflight_portability: float
    elapsed_cost: float
    elapsed_latency_ms: float
    failed_agents: frozenset[str] = frozenset()
    failure_count: int = 0

    def __post_init__(self) -> None:
        for value, label in (
            (self.inflight_fraction, "inflight_fraction"),
            (self.inflight_quality, "inflight_quality"),
            (self.inflight_portability, "inflight_portability"),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{label} must be in [0, 1]")
        if self.elapsed_cost < 0 or self.elapsed_latency_ms < 0:
            raise ValueError("elapsed resources must be non-negative")
        if self.failure_count < 0:
            raise ValueError("failure_count must be non-negative")


@dataclass(frozen=True)
class DynamicResult:
    quality: float
    utility: float
    added_cost: float
    recovery_latency_ms: float
    total_cost: float
    total_latency_ms: float
    wasted_work: float
    switched: bool
    deadline_miss: bool
    requirement_scores: Mapping[str, float]


def checkpoint_progress(task: Task, checkpoint: ExecutionCheckpoint) -> float:
    """Derive progress from completed and in-flight work for router input only."""

    total_weight = sum(item.weight for item in task.requirements)
    completed_weight = sum(
        item.weight
        for item in task.requirements
        if item.name in checkpoint.completed_artifacts
    )
    inflight = next(
        item for item in task.requirements if item.name == checkpoint.inflight_requirement
    )
    return min(
        1.0,
        (completed_weight + inflight.weight * checkpoint.inflight_fraction)
        / total_weight,
    )


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def latent_requirement_score(
    agent_id: str,
    requirement: Requirement,
    difficulty: float,
) -> float:
    if agent_id not in HIDDEN_AGENTS:
        raise ValueError(f"missing held-out agent profile: {agent_id}")
    latent_skill = HIDDEN_AGENTS[agent_id].skills.get(requirement.name, 0.0)
    threshold = 0.35 + 0.40 * difficulty + 0.15 * requirement.minimum**2
    return _sigmoid(8.2 * (latent_skill**1.30 - threshold))


def _weighted_geometric_mean(
    requirements: tuple[Requirement, ...],
    scores: Mapping[str, float],
) -> float:
    total_weight = sum(item.weight for item in requirements)
    return math.exp(
        sum(
            item.weight * math.log(max(scores[item.name], 1e-9))
            for item in requirements
        )
        / total_weight
    )


def _validate_plan(
    task: Task,
    agents: tuple[str, ...],
    assignments: Mapping[str, str],
    checkpoint: ExecutionCheckpoint,
) -> tuple[Requirement, ...]:
    remaining = tuple(
        item
        for item in task.requirements
        if item.name not in checkpoint.completed_artifacts
    )
    expected = {item.name for item in remaining}
    if set(assignments) != expected:
        raise ValueError("assignments must cover every remaining requirement exactly once")
    if set(assignments.values()) - set(agents):
        raise ValueError("assignments reference an agent outside the selected team")
    if not agents:
        raise ValueError("a recovery plan must select at least one agent")
    return remaining


def evaluate_recovery(
    task: Task,
    difficulty: float,
    mode: Mode,
    agents: tuple[str, ...],
    assignments: Mapping[str, str],
    advertised: Mapping[str, Agent],
    checkpoint: ExecutionCheckpoint,
) -> DynamicResult:
    """Score a recovery plan from artifact reuse and remaining execution work."""

    remaining = _validate_plan(task, agents, assignments, checkpoint)
    total_weight = sum(item.weight for item in task.requirements)
    switched = mode != checkpoint.active_mode or set(agents) != set(checkpoint.active_agents)

    requirement_scores = {
        name: artifact.quality
        for name, artifact in checkpoint.completed_artifacts.items()
    }
    remaining_fraction: dict[str, float] = {}
    wasted_work = 0.0
    for requirement in remaining:
        agent_id = assignments[requirement.name]
        failed = agent_id in checkpoint.failed_agents
        new_score = (
            0.01
            if failed
            else latent_requirement_score(agent_id, requirement, difficulty)
        )
        reused_fraction = 0.0
        if requirement.name == checkpoint.inflight_requirement:
            if agent_id == checkpoint.inflight_agent and not failed:
                reused_fraction = checkpoint.inflight_fraction
            else:
                reused_fraction = (
                    checkpoint.inflight_fraction * checkpoint.inflight_portability
                )
                wasted_work += (
                    requirement.weight
                    / total_weight
                    * checkpoint.inflight_fraction
                    * (1.0 - checkpoint.inflight_portability)
                )
        remaining_fraction[requirement.name] = 1.0 - reused_fraction
        score = reused_fraction * checkpoint.inflight_quality + (1.0 - reused_fraction) * new_score

        dependency_factors: list[float] = []
        for dependency in requirement.depends_on:
            artifact = checkpoint.completed_artifacts.get(dependency)
            if artifact is not None and artifact.producer != agent_id:
                dependency_factors.append(0.90 + 0.10 * artifact.portability)
        if dependency_factors:
            score *= math.prod(dependency_factors) ** (1.0 / len(dependency_factors))
        requirement_scores[requirement.name] = max(0.0, min(1.0, score))

    geometric_quality = _weighted_geometric_mean(task.requirements, requirement_scores)
    bottleneck = min(requirement_scores.values())
    selected_pairs = list(combinations(sorted(set(agents)), 2))
    compatibility = (
        math.prod(
            PAIR_COMPATIBILITY.get(frozenset(pair), 0.93)
            for pair in selected_pairs
        )
        ** (1.0 / len(selected_pairs))
        if selected_pairs
        else 1.0
    )
    span_penalty = 1.0 / (1.0 + 0.045 * max(0, len(agents) - 1) ** 1.6)
    quality = max(
        0.0,
        min(
            1.0,
            (0.70 * geometric_quality + 0.30 * bottleneck)
            * compatibility
            * span_penalty,
        ),
    )

    workload = {agent_id: 0.0 for agent_id in agents}
    for requirement in remaining:
        workload[assignments[requirement.name]] += (
            requirement.weight
            / total_weight
            * remaining_fraction[requirement.name]
        )
    active = set(checkpoint.active_agents)
    added_cost = 0.0
    for agent_id in agents:
        hidden = HIDDEN_AGENTS[agent_id]
        startup_fraction = 0.04 if agent_id in active else 0.20
        failure_multiplier = 2.5 if agent_id in checkpoint.failed_agents else 1.0
        added_cost += (
            advertised[agent_id].cost
            * (
                startup_fraction * hidden.startup_cost_multiplier
                + 0.80 * hidden.work_cost_multiplier * workload[agent_id] ** 0.84
            )
            * failure_multiplier
        )

    finish: dict[str, float] = {
        name: 0.0 for name in checkpoint.completed_artifacts
    }
    agent_ready = {agent_id: 0.0 for agent_id in agents}
    pending = {item.name: item for item in remaining}
    communication_edges: set[tuple[str, str]] = set()
    while pending:
        ready = sorted(
            (
                item
                for item in pending.values()
                if set(item.depends_on).issubset(finish)
            ),
            key=lambda item: item.name,
        )
        if not ready:
            raise RuntimeError("remaining requirements must form an executable DAG")
        for requirement in ready:
            agent_id = assignments[requirement.name]
            dependency_ready = max(
                (finish[name] for name in requirement.depends_on),
                default=0.0,
            )
            transfer_delay = 0.0
            for dependency in requirement.depends_on:
                artifact = checkpoint.completed_artifacts.get(dependency)
                if artifact is not None and artifact.producer != agent_id:
                    communication_edges.add((artifact.producer, agent_id))
                    transfer_delay = max(
                        transfer_delay,
                        85.0 * (1.0 - artifact.portability),
                    )
                dependency_agent = assignments.get(dependency)
                if dependency_agent is not None and dependency_agent != agent_id:
                    communication_edges.add((dependency_agent, agent_id))
            start = max(dependency_ready + transfer_delay, agent_ready[agent_id])
            hidden = HIDDEN_AGENTS[agent_id]
            failure_multiplier = 3.0 if agent_id in checkpoint.failed_agents else 1.0
            duration = (
                45.0
                + advertised[agent_id].latency_ms
                * hidden.latency_multiplier
                * (
                    requirement.weight
                    / total_weight
                    * remaining_fraction[requirement.name]
                )
                ** 0.74
            ) * failure_multiplier
            finish[requirement.name] = start + duration
            agent_ready[agent_id] = finish[requirement.name]
            pending.pop(requirement.name)

    recovery_latency = max(finish.values(), default=0.0) * (
        1.0 + 0.025 * len(communication_edges) ** 1.4
    )
    total_cost = checkpoint.elapsed_cost + added_cost
    total_latency = checkpoint.elapsed_latency_ms + recovery_latency
    cost_ratio = total_cost / task.budget
    latency_ratio = total_latency / task.deadline_ms
    deadline_penalty = max(0.0, latency_ratio - 1.0)
    utility = (
        quality
        - 0.22 * cost_ratio
        - 0.10 * latency_ratio
        - 0.40 * deadline_penalty
    )
    return DynamicResult(
        quality=quality,
        utility=utility,
        added_cost=added_cost,
        recovery_latency_ms=recovery_latency,
        total_cost=total_cost,
        total_latency_ms=total_latency,
        wasted_work=wasted_work,
        switched=switched,
        deadline_miss=total_latency > task.deadline_ms,
        requirement_scores=requirement_scores,
    )
