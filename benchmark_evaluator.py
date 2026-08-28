"""Held-out structural evaluator for the synthetic SAGE benchmark.

The evaluator intentionally does not import or reuse SAGE scoring helpers. Its
latent skills are specified independently from Agent Card values; artifact
quality uses a geometric bottleneck; team compatibility is multiplicative; and
realized resources use workload-sensitive formulas distinct from the router.
It remains a synthetic test environment, not external real-world evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

from sprix_sage import Agent, Mode, Requirement, Task


@dataclass(frozen=True)
class HiddenAgent:
    skills: Mapping[str, float]
    startup_cost_multiplier: float
    work_cost_multiplier: float
    latency_multiplier: float


@dataclass(frozen=True)
class ExternalResult:
    quality: float
    utility: float
    cost: float
    latency_ms: float
    requirement_scores: Mapping[str, float]
    agent_scores: Mapping[str, float]


# These latent values are independently specified, not advertised values plus
# a fixed perturbation. Several within-skill rankings deliberately reverse.
HIDDEN_AGENTS = {
    "generalist": HiddenAgent(
        {"code": 0.58, "research": 0.82, "vision": 0.43, "security": 0.74, "writing": 0.69},
        0.92,
        1.08,
        1.04,
    ),
    "coder": HiddenAgent(
        {"code": 0.88, "research": 0.55, "vision": 0.22, "security": 0.46, "writing": 0.61},
        1.15,
        0.96,
        1.16,
    ),
    "researcher": HiddenAgent(
        {"code": 0.50, "research": 0.84, "vision": 0.65, "security": 0.31, "writing": 0.91},
        0.86,
        1.02,
        1.23,
    ),
    "vision": HiddenAgent(
        {"code": 0.29, "research": 0.46, "vision": 0.89, "security": 0.56, "writing": 0.35},
        1.04,
        1.12,
        0.91,
    ),
    "reviewer": HiddenAgent(
        {"code": 0.71, "research": 0.62, "vision": 0.54, "security": 0.82, "writing": 0.86},
        0.80,
        0.98,
        0.87,
    ),
}


# Team compatibility is applied as a geometric factor across all used pairs.
# It is not an additive bonus available to the router.
PAIR_COMPATIBILITY = {
    frozenset(("generalist", "coder")): 1.03,
    frozenset(("generalist", "researcher")): 0.94,
    frozenset(("generalist", "vision")): 1.00,
    frozenset(("generalist", "reviewer")): 1.06,
    frozenset(("coder", "reviewer")): 0.98,
    frozenset(("researcher", "reviewer")): 1.04,
    frozenset(("coder", "researcher")): 0.88,
    frozenset(("coder", "vision")): 1.05,
    frozenset(("researcher", "vision")): 0.96,
    frozenset(("vision", "reviewer")): 0.90,
}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


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


def evaluate_plan(
    task: Task,
    difficulty: float,
    mode: Mode,
    agents: tuple[str, ...],
    assignments: Mapping[str, str],
    advertised: Mapping[str, Agent],
) -> ExternalResult:
    """Evaluate a plan with structurally held-out quality and resource models."""

    requirement_scores: dict[str, float] = {}
    for requirement in task.requirements:
        agent_id = assignments[requirement.name]
        latent_skill = HIDDEN_AGENTS[agent_id].skills.get(requirement.name, 0.0)
        threshold = 0.36 + 0.38 * difficulty + 0.16 * requirement.minimum**2
        requirement_scores[requirement.name] = _sigmoid(
            8.5 * (latent_skill**1.35 - threshold)
        )

    used_agents = tuple(sorted(set(assignments.values())))
    geometric_quality = _weighted_geometric_mean(task.requirements, requirement_scores)
    bottleneck = min(requirement_scores.values())
    pairs = list(combinations(used_agents, 2))
    compatibility = (
        math.prod(
            PAIR_COMPATIBILITY.get(frozenset(pair), 0.93) for pair in pairs
        )
        ** (1.0 / len(pairs))
        if pairs
        else 1.0
    )
    span_penalty = 1.0 / (1.0 + 0.055 * max(0, len(used_agents) - 1) ** 1.7)
    transfer_factor = 1.0
    if mode is Mode.HANDOFF:
        transfer_factor -= (
            0.03
            + 0.14
            * math.sqrt(task.progress)
            * (1.0 - task.context_transferability) ** 2
        )
    quality = max(
        0.0,
        min(
            1.0,
            (0.72 * geometric_quality + 0.28 * bottleneck)
            * compatibility
            * span_penalty
            * transfer_factor,
        ),
    )

    total_weight = sum(item.weight for item in task.requirements)
    workload = {agent_id: 0.0 for agent_id in used_agents}
    for requirement in task.requirements:
        workload[assignments[requirement.name]] += requirement.weight / total_weight
    actual_cost = sum(
        advertised[agent_id].cost
        * (
            0.22 * HIDDEN_AGENTS[agent_id].startup_cost_multiplier
            + 0.78
            * HIDDEN_AGENTS[agent_id].work_cost_multiplier
            * workload[agent_id] ** 0.82
        )
        for agent_id in used_agents
    )

    finish: dict[str, float] = {}
    agent_ready = {agent_id: 0.0 for agent_id in used_agents}
    remaining = {item.name: item for item in task.requirements}
    while remaining:
        ready = sorted(
            (
                item
                for item in remaining.values()
                if set(item.depends_on).issubset(finish)
            ),
            key=lambda item: item.name,
        )
        for requirement in ready:
            agent_id = assignments[requirement.name]
            dependency_ready = max(
                (finish[name] for name in requirement.depends_on), default=0.0
            )
            start = max(dependency_ready, agent_ready[agent_id])
            work_fraction = requirement.weight / total_weight
            duration = (
                70.0
                + advertised[agent_id].latency_ms
                * HIDDEN_AGENTS[agent_id].latency_multiplier
                * work_fraction**0.72
            )
            finish[requirement.name] = start + duration
            agent_ready[agent_id] = finish[requirement.name]
            remaining.pop(requirement.name)
    actual_latency = max(finish.values()) * (
        1.0 + 0.035 * max(0, len(used_agents) - 1) ** 1.5
    )

    cost_ratio = actual_cost / task.budget
    latency_ratio = actual_latency / task.deadline_ms
    deadline_penalty = max(0.0, latency_ratio - 1.0)
    utility = (
        quality
        - 0.22 * cost_ratio
        - 0.10 * latency_ratio
        - 0.40 * deadline_penalty
    )

    per_agent: dict[str, list[float]] = {agent_id: [] for agent_id in agents}
    for requirement_name, agent_id in assignments.items():
        per_agent[agent_id].append(requirement_scores[requirement_name])
    agent_scores = {
        agent_id: sum(scores) / len(scores) if scores else 0.5
        for agent_id, scores in per_agent.items()
    }
    return ExternalResult(
        quality,
        utility,
        actual_cost,
        actual_latency,
        requirement_scores,
        agent_scores,
    )
