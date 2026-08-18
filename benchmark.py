"""Deterministic external simulator for comparing SAGE with routing baselines.

Unlike the original smoke test, the evaluator does not use SAGE's noisy-OR
coverage or predicted success probability as ground truth.  Advertised skills
are imperfect, quality is nonlinear, pair effects are hidden, and realized cost
and latency differ from bids.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import math
import random
from statistics import mean, pstdev
from typing import Mapping

from sprix_sage import Agent, ExecutionOutcome, Mode, Requirement, RouteDecision, SAGERouter, Task


SKILLS = ("code", "research", "vision", "security", "writing")


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(frozen=True)
class HiddenAgent:
    skills: Mapping[str, float]
    cost_multiplier: float
    latency_multiplier: float


@dataclass(frozen=True)
class ExternalResult:
    quality: float
    utility: float
    cost: float
    latency_ms: float
    requirement_scores: Mapping[str, float]
    agent_scores: Mapping[str, float]


def make_agents() -> tuple[list[Agent], dict[str, HiddenAgent]]:
    agents = [
        Agent("generalist", {"code": 0.72, "research": 0.74, "vision": 0.58, "security": 0.66, "writing": 0.80}, 0.05, 900, frozenset({"public", "secure"})),
        Agent("coder", {"code": 0.97, "research": 0.43, "vision": 0.31, "security": 0.60, "writing": 0.45}, 0.09, 1150, frozenset({"public"})),
        Agent("researcher", {"code": 0.42, "research": 0.97, "vision": 0.51, "security": 0.38, "writing": 0.83}, 0.08, 1350, frozenset({"public"})),
        Agent("vision", {"code": 0.38, "research": 0.58, "vision": 0.97, "security": 0.35, "writing": 0.47}, 0.10, 1100, frozenset({"public"})),
        Agent("reviewer", {"code": 0.62, "research": 0.76, "vision": 0.40, "security": 0.94, "writing": 0.91}, 0.06, 800, frozenset({"public", "secure"})),
    ]
    hidden = {
        "generalist": HiddenAgent({"code": 0.68, "research": 0.70, "vision": 0.55, "security": 0.61, "writing": 0.82}, 1.02, 1.04),
        "coder": HiddenAgent({"code": 0.94, "research": 0.36, "vision": 0.28, "security": 0.51, "writing": 0.42}, 1.18, 1.12),
        "researcher": HiddenAgent({"code": 0.38, "research": 0.93, "vision": 0.48, "security": 0.34, "writing": 0.88}, 0.97, 1.20),
        "vision": HiddenAgent({"code": 0.33, "research": 0.51, "vision": 0.91, "security": 0.31, "writing": 0.44}, 1.08, 0.96),
        "reviewer": HiddenAgent({"code": 0.57, "research": 0.72, "vision": 0.36, "security": 0.90, "writing": 0.95}, 0.94, 0.90),
    }
    return agents, hidden


HIDDEN_SYNERGY = {
    frozenset(("generalist", "coder")): 0.045,
    frozenset(("generalist", "researcher")): 0.030,
    frozenset(("generalist", "vision")): 0.020,
    frozenset(("generalist", "reviewer")): 0.055,
    frozenset(("coder", "reviewer")): 0.035,
    frozenset(("researcher", "reviewer")): 0.050,
    frozenset(("coder", "researcher")): -0.025,
    frozenset(("coder", "vision")): 0.015,
    frozenset(("researcher", "vision")): 0.025,
    frozenset(("vision", "reviewer")): -0.015,
}


def generate_task(rng: random.Random, index: int) -> tuple[Task, float]:
    count = rng.choices((1, 2, 3, 4), weights=(0.18, 0.36, 0.32, 0.14))[0]
    chosen = rng.sample(SKILLS, count)
    raw_weights = [rng.uniform(0.6, 1.4) for _ in chosen]
    total = sum(raw_weights)
    requirements: list[Requirement] = []
    for position, (name, raw_weight) in enumerate(zip(chosen, raw_weights)):
        dependencies: tuple[str, ...] = ()
        if position and rng.random() < 0.48:
            dependencies = (chosen[rng.randrange(position)],)
        requirements.append(
            Requirement(name, raw_weight / total, rng.uniform(0.58, 0.78), dependencies)
        )
    secure = rng.random() < 0.12
    task = Task(
        f"task-{index}",
        tuple(requirements),
        value=1.0,
        budget=rng.choice((0.12, 0.20, 0.32, 0.45)),
        deadline_ms=rng.choice((900, 1400, 2200, 3200)),
        required_permissions=frozenset({"secure" if secure else "public"}),
        risk_tolerance=rng.uniform(0.2, 0.8),
        progress=rng.uniform(0.0, 0.85),
        handoff_friction=rng.uniform(0.15, 0.35),
        coordination_overhead=rng.uniform(0.02, 0.12),
        context_transferability=rng.uniform(0.35, 0.90),
    )
    difficulty = rng.uniform(0.15, 0.85)
    return task, difficulty


def assign_single(task: Task, agent_id: str) -> dict[str, str]:
    return {requirement.name: agent_id for requirement in task.requirements}


def external_evaluate(
    task: Task,
    difficulty: float,
    mode: Mode,
    agents: tuple[str, ...],
    assignments: Mapping[str, str],
    hidden: Mapping[str, HiddenAgent],
    advertised: Mapping[str, Agent],
) -> ExternalResult:
    requirement_scores: dict[str, float] = {}
    for requirement in task.requirements:
        agent_id = assignments[requirement.name]
        skill = hidden[agent_id].skills.get(requirement.name, 0.0)
        threshold = 0.42 + 0.28 * difficulty + 0.12 * requirement.minimum
        requirement_scores[requirement.name] = sigmoid(7.0 * (skill - threshold))

    weighted_quality = sum(
        requirement.weight * requirement_scores[requirement.name]
        for requirement in task.requirements
    )
    bottleneck = min(requirement_scores.values())
    used_agents = tuple(sorted(set(assignments.values())))
    pair_effect = sum(
        HIDDEN_SYNERGY.get(frozenset(pair), -0.01)
        for pair in combinations(used_agents, 2)
    )
    coordination_penalty = 0.025 * max(0, len(used_agents) - 1) ** 2
    handoff_penalty = 0.0
    if mode is Mode.HANDOFF:
        handoff_penalty = task.progress * (1.0 - task.context_transferability) * task.handoff_friction
    quality = max(
        0.0,
        min(1.0, 0.68 * weighted_quality + 0.32 * bottleneck + pair_effect - coordination_penalty - handoff_penalty),
    )

    actual_cost = sum(advertised[agent_id].cost * hidden[agent_id].cost_multiplier for agent_id in agents)
    total_weight = sum(requirement.weight for requirement in task.requirements)
    finish: dict[str, float] = {}
    agent_ready = {agent_id: 0.0 for agent_id in used_agents}
    remaining = {item.name: item for item in task.requirements}
    while remaining:
        ready = sorted(
            (item for item in remaining.values() if set(item.depends_on).issubset(finish)),
            key=lambda item: item.name,
        )
        for requirement in ready:
            agent_id = assignments[requirement.name]
            dependency_ready = max((finish[name] for name in requirement.depends_on), default=0.0)
            start = max(dependency_ready, agent_ready[agent_id])
            duration = (
                advertised[agent_id].latency_ms
                * hidden[agent_id].latency_multiplier
                * requirement.weight
                / total_weight
            )
            finish[requirement.name] = start + duration
            agent_ready[agent_id] = finish[requirement.name]
            remaining.pop(requirement.name)
    actual_latency = max(finish.values()) * (1.0 + task.coordination_overhead * max(0, len(used_agents) - 1))

    cost_ratio = actual_cost / task.budget
    latency_ratio = actual_latency / task.deadline_ms
    deadline_penalty = max(0.0, latency_ratio - 1.0)
    utility = quality - 0.22 * cost_ratio - 0.10 * latency_ratio - 0.40 * deadline_penalty

    per_agent: dict[str, list[float]] = {agent_id: [] for agent_id in agents}
    for requirement_name, agent_id in assignments.items():
        per_agent[agent_id].append(requirement_scores[requirement_name])
    agent_scores = {
        agent_id: sum(scores) / len(scores) if scores else 0.5
        for agent_id, scores in per_agent.items()
    }
    return ExternalResult(quality, utility, actual_cost, actual_latency, requirement_scores, agent_scores)


def eligible_solo(task: Task, agent: Agent) -> bool:
    return (
        task.required_permissions.issubset(agent.permissions)
        and agent.cost <= task.budget
        and agent.latency_ms <= task.deadline_ms
    )


def simulate(
    seed: int = 11,
    tasks: int = 500,
) -> tuple[dict[str, dict[str, float]], Counter[str], int]:
    rng = random.Random(seed)
    agents, hidden = make_agents()
    agent_map = {agent.agent_id: agent for agent in agents}
    learned = SAGERouter(agents, "generalist", max_collaborators=3, beam_width=10, exploration=True, seed=seed)
    static = SAGERouter(agents, "generalist", max_collaborators=3, beam_width=10, exploration=False, seed=seed)

    metrics = {
        name: {"quality": 0.0, "utility": 0.0, "cost": 0.0, "latency": 0.0, "misses": 0.0}
        for name in ("self", "skill_solo", "oracle_solo", "static_sage", "learned_sage")
    }
    modes: Counter[str] = Counter()

    def accumulate(name: str, result: ExternalResult, task: Task) -> None:
        metrics[name]["quality"] += result.quality
        metrics[name]["utility"] += result.utility
        metrics[name]["cost"] += result.cost / task.budget
        metrics[name]["latency"] += result.latency_ms / task.deadline_ms
        metrics[name]["misses"] += float(result.latency_ms > task.deadline_ms)

    for index in range(tasks):
        task, difficulty = generate_task(rng, index)
        solo_agents = [agent for agent in agents if eligible_solo(task, agent)]

        self_result = external_evaluate(
            task, difficulty, Mode.SELF, ("generalist",), assign_single(task, "generalist"), hidden, agent_map
        )
        accumulate("self", self_result, task)

        skill_agent = max(
            solo_agents,
            key=lambda agent: (
                sum(item.weight * agent.skills.get(item.name, 0.0) for item in task.requirements)
                - 0.12 * agent.cost / task.budget
                - 0.05 * agent.latency_ms / task.deadline_ms
            ),
        )
        skill_result = external_evaluate(
            task,
            difficulty,
            Mode.SELF if skill_agent.agent_id == "generalist" else Mode.HANDOFF,
            (skill_agent.agent_id,),
            assign_single(task, skill_agent.agent_id),
            hidden,
            agent_map,
        )
        accumulate("skill_solo", skill_result, task)

        oracle_candidates = []
        for agent in solo_agents:
            mode = Mode.SELF if agent.agent_id == "generalist" else Mode.HANDOFF
            result = external_evaluate(
                task, difficulty, mode, (agent.agent_id,), assign_single(task, agent.agent_id), hidden, agent_map
            )
            oracle_candidates.append(result)
        oracle_feasible = [
            result
            for result in oracle_candidates
            if result.cost <= task.budget and result.latency_ms <= task.deadline_ms
        ]
        accumulate("oracle_solo", max(oracle_feasible, key=lambda result: result.utility), task)

        static_decision = static.route(task)
        static_result = external_evaluate(
            task,
            difficulty,
            static_decision.mode,
            static_decision.agents,
            static_decision.assignments,
            hidden,
            agent_map,
        )
        accumulate("static_sage", static_result, task)

        learned_decision = learned.route(task)
        learned_result = external_evaluate(
            task,
            difficulty,
            learned_decision.mode,
            learned_decision.agents,
            learned_decision.assignments,
            hidden,
            agent_map,
        )
        accumulate("learned_sage", learned_result, task)
        modes[learned_decision.mode.value] += 1
        learned.record_outcome(
            learned_decision,
            ExecutionOutcome(
                learned_result.quality,
                agent_scores=learned_result.agent_scores,
                requirement_scores=learned_result.requirement_scores,
                actual_cost=learned_result.cost,
                actual_latency_ms=learned_result.latency_ms,
            ),
        )

    averages = {
        name: {
            "quality": row["quality"] / tasks,
            "utility": row["utility"] / tasks,
            "cost": row["cost"] / tasks,
            "latency": row["latency"] / tasks,
            "misses": 100.0 * row["misses"] / tasks,
        }
        for name, row in metrics.items()
    }
    return averages, modes, learned.success_model.updates


def run(seed: int = 11, tasks: int = 500) -> None:
    metrics, modes, updates = simulate(seed, tasks)
    print(f"tasks: {tasks} (external nonlinear simulator; seed={seed})")
    print("strategy       quality  utility  cost/budget  latency/deadline  deadline-miss")
    for name, row in metrics.items():
        print(
            f"{name:14s} {row['quality']:7.3f}  {row['utility']:7.3f}"
            f"      {row['cost']:7.3f}           {row['latency']:7.3f}"
            f"         {row['misses']:5.1f}%"
        )
    print(f"learned route mix: {dict(modes)}")
    print(f"online model updates: {updates}")


def run_suite(seeds: tuple[int, ...] = (3, 7, 11, 19, 23), tasks_per_seed: int = 500) -> None:
    runs = [simulate(seed, tasks_per_seed)[0] for seed in seeds]
    print(
        f"tasks: {len(seeds) * tasks_per_seed} "
        f"({len(seeds)} seeds x {tasks_per_seed}; external nonlinear simulator)"
    )
    print("strategy       quality       utility       cost/budget   latency/deadline  deadline-miss")
    for name in runs[0]:
        values = {
            metric: [run_metrics[name][metric] for run_metrics in runs]
            for metric in ("quality", "utility", "cost", "latency", "misses")
        }
        print(
            f"{name:14s} {mean(values['quality']):.3f}+/-{pstdev(values['quality']):.3f}"
            f"  {mean(values['utility']):.3f}+/-{pstdev(values['utility']):.3f}"
            f"    {mean(values['cost']):.3f}+/-{pstdev(values['cost']):.3f}"
            f"      {mean(values['latency']):.3f}+/-{pstdev(values['latency']):.3f}"
            f"       {mean(values['misses']):4.1f}%"
        )


if __name__ == "__main__":
    run_suite()
