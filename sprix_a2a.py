"""Transport-neutral helpers for integrating SAGE with A2A systems.

Agent Cards advertise discoverable skills but do not provide trustworthy,
portable proficiency, price, or latency scores.  This module therefore keeps
card parsing separate from locally supplied marketplace evidence and converts a
route decision into an execution plan without transmitting any task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from sprix_sage import Agent, Mode, RouteDecision, Task


@dataclass(frozen=True)
class AgentCardProfile:
    """Normalized result of combining an Agent Card with local evidence."""

    agent: Agent
    card_name: str
    declared_skills: tuple[str, ...]
    card_url: str | None = None
    card_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent.agent_id,
            "card_name": self.card_name,
            "declared_skills": list(self.declared_skills),
            "card_url": self.card_url,
            "card_version": self.card_version,
            "skills": dict(self.agent.skills),
            "cost": self.agent.cost,
            "latency_ms": self.agent.latency_ms,
            "permissions": sorted(self.agent.permissions),
            "availability": self.agent.availability,
            "load": self.agent.load,
        }


@dataclass(frozen=True)
class ExecutionStep:
    requirement: str
    agent_id: str
    depends_on: tuple[str, ...]
    weight: float
    minimum: float

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement": self.requirement,
            "agent_id": self.agent_id,
            "depends_on": list(self.depends_on),
            "weight": self.weight,
            "minimum": self.minimum,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """A routing result shaped for an A2A client or marketplace executor."""

    task_id: str
    mode: Mode
    owner_agent_id: str
    executor_agent_ids: tuple[str, ...]
    steps: tuple[ExecutionStep, ...]
    communication_edges: tuple[tuple[str, str], ...]
    estimated_cost: float
    estimated_latency_ms: float
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "mode": self.mode.value,
            "owner_agent_id": self.owner_agent_id,
            "executor_agent_ids": list(self.executor_agent_ids),
            "steps": [step.to_dict() for step in self.steps],
            "communication_edges": [list(edge) for edge in self.communication_edges],
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "rationale": self.rationale,
        }


def profile_from_agent_card(
    card: Mapping[str, object],
    *,
    agent_id: str,
    skill_scores: Mapping[str, float],
    cost: float,
    latency_ms: float,
    permissions: frozenset[str] = frozenset(),
    availability: float = 1.0,
    load: float = 0.0,
) -> AgentCardProfile:
    """Combine declared card skills with locally measured routing evidence.

    ``skill_scores`` must use skill IDs declared by the card.  The caller owns
    score calibration; SAGE intentionally does not infer proficiency from skill
    descriptions or marketing text.
    """

    name = card.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Agent Card name must be a non-empty string")
    raw_skills = card.get("skills")
    if not isinstance(raw_skills, list) or not raw_skills:
        raise ValueError("Agent Card skills must be a non-empty list")

    declared: list[str] = []
    for index, raw_skill in enumerate(raw_skills):
        if not isinstance(raw_skill, Mapping):
            raise ValueError(f"Agent Card skill {index} must be an object")
        skill_id = raw_skill.get("id")
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError(f"Agent Card skill {index} must have a non-empty id")
        declared.append(skill_id.strip())
    if len(declared) != len(set(declared)):
        raise ValueError("Agent Card skill IDs must be unique")

    if not skill_scores:
        raise ValueError("skill_scores must contain at least one locally calibrated score")
    unknown = sorted(set(skill_scores) - set(declared))
    if unknown:
        raise ValueError(f"skill_scores reference undeclared card skills: {unknown}")

    url = card.get("url")
    version = card.get("version")
    if url is not None and not isinstance(url, str):
        raise ValueError("Agent Card url must be a string when provided")
    if version is not None and not isinstance(version, str):
        raise ValueError("Agent Card version must be a string when provided")

    agent = Agent(
        agent_id=agent_id,
        skills=dict(skill_scores),
        cost=cost,
        latency_ms=latency_ms,
        permissions=permissions,
        availability=availability,
        load=load,
    )
    return AgentCardProfile(
        agent=agent,
        card_name=name.strip(),
        declared_skills=tuple(declared),
        card_url=url,
        card_version=version,
    )


def execution_plan(task: Task, decision: RouteDecision) -> ExecutionPlan:
    """Convert a SAGE decision into a validated transport-neutral plan."""

    if not decision.agents:
        raise ValueError("route decision must select at least one agent")
    requirement_by_name = {requirement.name: requirement for requirement in task.requirements}
    unknown = set(decision.assignments) - set(requirement_by_name)
    if unknown:
        raise ValueError(f"decision assigns unknown requirements: {sorted(unknown)}")
    invalid_agents = set(decision.assignments.values()) - set(decision.agents)
    if invalid_agents:
        raise ValueError(f"decision assigns unselected agents: {sorted(invalid_agents)}")

    steps = tuple(
        ExecutionStep(
            requirement=requirement.name,
            agent_id=decision.assignments[requirement.name],
            depends_on=tuple(
                dependency
                for dependency in requirement.depends_on
                if dependency in decision.assignments
            ),
            weight=requirement.weight,
            minimum=requirement.minimum,
        )
        for requirement in task.requirements
        if requirement.name in decision.assignments
    )
    if len(steps) != len(decision.assignments):
        raise ValueError("decision assignments could not be converted into execution steps")

    owner = decision.agents[0]
    return ExecutionPlan(
        task_id=task.task_id,
        mode=decision.mode,
        owner_agent_id=owner,
        executor_agent_ids=decision.agents,
        steps=steps,
        communication_edges=decision.topology,
        estimated_cost=decision.cost,
        estimated_latency_ms=decision.latency_ms,
        rationale=decision.explanation,
    )
