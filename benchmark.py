"""Deterministic synthetic benchmark with structural and learning ablations.

The held-out evaluator lives in ``benchmark_evaluator.py`` and deliberately
uses different quality, compatibility, cost, latency, and handoff equations
from SAGE. Greedy and random team baselines prevent the comparison from
crediting SAGE merely for being the only policy allowed to collaborate.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Mapping

from benchmark_evaluator import ExternalResult, evaluate_plan
from sprix_learning import OnlineSuccessModel
from sprix_sage import Agent, ExecutionOutcome, Mode, Requirement, RouteDecision, SAGERouter, Task

SKILLS = ("code", "research", "vision", "security", "writing")
DEFAULT_SEEDS = (3, 7, 11, 19, 23)
STRATEGIES = (
    "incumbent",
    "skill_solo",
    "oracle_solo",
    "random_team",
    "greedy_team",
    "static_sage",
    "learned_no_explore",
    "learned_explore",
    "learned_random_init",
)
LEARNED_STRATEGIES = (
    "learned_no_explore",
    "learned_explore",
    "learned_random_init",
)
METRICS = ("quality", "utility", "cost", "latency", "misses")


@dataclass(frozen=True)
class Plan:
    mode: Mode
    agents: tuple[str, ...]
    assignments: Mapping[str, str]


def make_agents() -> list[Agent]:
    """Return public Agent Card-like values visible to every routing policy."""

    return [
        Agent(
            "generalist",
            {"code": 0.72, "research": 0.74, "vision": 0.58, "security": 0.66, "writing": 0.80},
            0.05,
            900,
            frozenset({"public", "secure"}),
        ),
        Agent(
            "coder",
            {"code": 0.97, "research": 0.43, "vision": 0.31, "security": 0.60, "writing": 0.45},
            0.09,
            1150,
            frozenset({"public"}),
        ),
        Agent(
            "researcher",
            {"code": 0.42, "research": 0.97, "vision": 0.51, "security": 0.38, "writing": 0.83},
            0.08,
            1350,
            frozenset({"public"}),
        ),
        Agent(
            "vision",
            {"code": 0.38, "research": 0.58, "vision": 0.97, "security": 0.35, "writing": 0.47},
            0.10,
            1100,
            frozenset({"public"}),
        ),
        Agent(
            "reviewer",
            {"code": 0.62, "research": 0.76, "vision": 0.40, "security": 0.94, "writing": 0.91},
            0.06,
            800,
            frozenset({"public", "secure"}),
        ),
    ]


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
    return task, rng.uniform(0.15, 0.85)


def assign_single(task: Task, agent_id: str) -> dict[str, str]:
    return {requirement.name: agent_id for requirement in task.requirements}


def eligible_agents(task: Task, agents: list[Agent]) -> list[Agent]:
    return [
        agent
        for agent in agents
        if agent.availability > 0
        and task.required_permissions.issubset(agent.permissions)
    ]


def plan_mode(agent_ids: tuple[str, ...]) -> Mode:
    if agent_ids == ("generalist",):
        return Mode.SELF
    if len(agent_ids) == 1:
        return Mode.HANDOFF
    return Mode.COLLABORATE


def visible_plan_resources(
    task: Task,
    team: tuple[str, ...],
    assignments: Mapping[str, str],
    agent_map: Mapping[str, Agent],
) -> tuple[float, float]:
    """Cheap quote-only model available to non-SAGE baselines."""

    total_weight = sum(item.weight for item in task.requirements)
    workload = {agent_id: 0.0 for agent_id in team}
    finish: dict[str, float] = {}
    agent_ready = {agent_id: 0.0 for agent_id in team}
    remaining = {item.name: item for item in task.requirements}
    for item in task.requirements:
        workload[assignments[item.name]] += item.weight / total_weight
    cost = sum(
        agent_map[agent_id].cost * (0.15 + 0.85 * workload[agent_id])
        for agent_id in team
    )
    while remaining:
        ready = sorted(
            (
                item
                for item in remaining.values()
                if set(item.depends_on).issubset(finish)
            ),
            key=lambda item: item.name,
        )
        for item in ready:
            agent_id = assignments[item.name]
            dependency_ready = max(
                (finish[name] for name in item.depends_on), default=0.0
            )
            start = max(dependency_ready, agent_ready[agent_id])
            finish[item.name] = (
                start + agent_map[agent_id].latency_ms * item.weight / total_weight
            )
            agent_ready[agent_id] = finish[item.name]
            remaining.pop(item.name)
    topology = {
        (assignments[dependency], assignments[item.name])
        for item in task.requirements
        for dependency in item.depends_on
        if assignments[dependency] != assignments[item.name]
    }
    latency = max(finish.values()) * (
        1.0 + task.coordination_overhead * len(topology)
    )
    return cost, latency


def visible_feasible(
    task: Task,
    team: tuple[str, ...],
    assignments: Mapping[str, str],
    agent_map: Mapping[str, Agent],
) -> bool:
    cost, latency = visible_plan_resources(task, team, assignments, agent_map)
    return cost <= task.budget and latency <= task.deadline_ms


def greedy_assign(
    task: Task,
    team: tuple[str, ...],
    agent_map: Mapping[str, Agent],
) -> dict[str, str]:
    return {
        item.name: max(
            team,
            key=lambda agent_id: agent_map[agent_id].skills.get(item.name, 0.0),
        )
        for item in task.requirements
    }


def visible_score(
    task: Task,
    team: tuple[str, ...],
    assignments: Mapping[str, str],
    agent_map: Mapping[str, Agent],
) -> float:
    total_weight = sum(item.weight for item in task.requirements)
    capability = sum(
        item.weight * agent_map[assignments[item.name]].skills.get(item.name, 0.0)
        for item in task.requirements
    ) / total_weight
    cost, latency = visible_plan_resources(task, team, assignments, agent_map)
    return capability - 0.22 * cost / task.budget - 0.10 * latency / task.deadline_ms


def greedy_team_plan(
    task: Task,
    agents: list[Agent],
    agent_map: Mapping[str, Agent],
    max_team_size: int = 3,
) -> Plan:
    allowed = {agent.agent_id for agent in eligible_agents(task, agents)}
    team: tuple[str, ...] = (
        ("generalist",) if "generalist" in allowed else (min(allowed),)
    )
    assignments = greedy_assign(task, team, agent_map)
    score = visible_score(task, team, assignments, agent_map)
    while len(team) < max_team_size:
        options: list[tuple[float, tuple[str, ...], dict[str, str]]] = []
        for agent_id in sorted(allowed - set(team)):
            candidate_team = team + (agent_id,)
            candidate_assignments = greedy_assign(task, candidate_team, agent_map)
            if visible_feasible(task, candidate_team, candidate_assignments, agent_map):
                options.append(
                    (
                        visible_score(task, candidate_team, candidate_assignments, agent_map),
                        candidate_team,
                        candidate_assignments,
                    )
                )
        if not options:
            break
        best_score, best_team, best_assignments = max(options, key=lambda item: item[0])
        if best_score <= score + 1e-12:
            break
        score, team, assignments = best_score, best_team, best_assignments
    used = tuple(sorted(set(assignments.values()), key=team.index))
    return Plan(plan_mode(used), used, assignments)


def random_team_plan(
    task: Task,
    agents: list[Agent],
    agent_map: Mapping[str, Agent],
    rng: random.Random,
    max_team_size: int = 3,
) -> Plan:
    allowed = sorted(agent.agent_id for agent in eligible_agents(task, agents))
    incumbent = "generalist" if "generalist" in allowed else allowed[0]
    peers = [agent_id for agent_id in allowed if agent_id != incumbent]
    attempts: list[Plan] = []
    for _ in range(24):
        team_size = rng.randint(1, min(max_team_size, len(allowed)))
        team = (incumbent, *rng.sample(peers, min(team_size - 1, len(peers))))
        assignments = {
            item.name: team[rng.randrange(len(team))] for item in task.requirements
        }
        used = tuple(agent_id for agent_id in team if agent_id in assignments.values())
        plan = Plan(plan_mode(used), used, assignments)
        if visible_feasible(task, used, assignments, agent_map):
            return plan
        attempts.append(plan)
    fallback = Plan(Mode.SELF, (incumbent,), assign_single(task, incumbent))
    feasible_attempts = [
        plan
        for plan in attempts
        if visible_feasible(task, plan.agents, plan.assignments, agent_map)
    ]
    return rng.choice(feasible_attempts) if feasible_attempts else fallback


def decision_plan(decision: RouteDecision) -> Plan:
    return Plan(decision.mode, decision.agents, decision.assignments)


def evaluate(
    task: Task,
    difficulty: float,
    plan: Plan,
    agent_map: Mapping[str, Agent],
) -> ExternalResult:
    return evaluate_plan(
        task,
        difficulty,
        plan.mode,
        plan.agents,
        plan.assignments,
        agent_map,
    )


def _empty_metrics() -> dict[str, float]:
    return {metric: 0.0 for metric in METRICS}


def _accumulate(row: dict[str, float], result: ExternalResult, task: Task) -> None:
    row["quality"] += result.quality
    row["utility"] += result.utility
    row["cost"] += result.cost / task.budget
    row["latency"] += result.latency_ms / task.deadline_ms
    row["misses"] += float(result.latency_ms > task.deadline_ms)


def _average(row: Mapping[str, float], count: int) -> dict[str, float]:
    result = {metric: row[metric] / count for metric in METRICS}
    result["misses"] *= 100.0
    return result


def simulate(
    seed: int = 11,
    tasks: int = 500,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, Counter[str]],
    dict[str, int],
    dict[str, dict[str, dict[str, float]]],
]:
    rng = random.Random(seed)
    baseline_rng = random.Random(seed + 100_003)
    agents = make_agents()
    agent_map = {agent.agent_id: agent for agent in agents}
    routers = {
        "static_sage": SAGERouter(
            agents,
            "generalist",
            max_collaborators=3,
            beam_width=10,
            exploration=False,
            seed=seed,
        ),
        "learned_no_explore": SAGERouter(
            agents,
            "generalist",
            max_collaborators=3,
            beam_width=10,
            exploration=False,
            seed=seed,
        ),
        "learned_explore": SAGERouter(
            agents,
            "generalist",
            max_collaborators=3,
            beam_width=10,
            exploration=True,
            seed=seed,
        ),
        "learned_random_init": SAGERouter(
            agents,
            "generalist",
            max_collaborators=3,
            beam_width=10,
            exploration=False,
            seed=seed,
            success_model=OnlineSuccessModel.randomized(seed),
        ),
    }
    metrics = {name: _empty_metrics() for name in STRATEGIES}
    route_mix: dict[str, Counter[str]] = {name: Counter() for name in routers}
    window = min(100, max(1, tasks // 2))
    curve_sums = {
        name: {"first": _empty_metrics(), "last": _empty_metrics()}
        for name in LEARNED_STRATEGIES
    }

    for index in range(tasks):
        task, difficulty = generate_task(rng, index)
        allowed = eligible_agents(task, agents)

        plans: dict[str, Plan] = {
            "incumbent": Plan(Mode.SELF, ("generalist",), assign_single(task, "generalist")),
            "random_team": random_team_plan(task, agents, agent_map, baseline_rng),
            "greedy_team": greedy_team_plan(task, agents, agent_map),
        }
        skill_agent = max(
            allowed,
            key=lambda agent: (
                sum(
                    item.weight * agent.skills.get(item.name, 0.0)
                    for item in task.requirements
                )
                - 0.12 * agent.cost / task.budget
                - 0.05 * agent.latency_ms / task.deadline_ms
            ),
        )
        plans["skill_solo"] = Plan(
            plan_mode((skill_agent.agent_id,)),
            (skill_agent.agent_id,),
            assign_single(task, skill_agent.agent_id),
        )

        oracle_candidates: list[tuple[ExternalResult, Plan]] = []
        for agent in allowed:
            plan = Plan(
                plan_mode((agent.agent_id,)),
                (agent.agent_id,),
                assign_single(task, agent.agent_id),
            )
            oracle_candidates.append((evaluate(task, difficulty, plan, agent_map), plan))
        oracle_feasible = [
            candidate
            for candidate in oracle_candidates
            if candidate[0].cost <= task.budget
            and candidate[0].latency_ms <= task.deadline_ms
        ]
        plans["oracle_solo"] = max(
            oracle_feasible or oracle_candidates,
            key=lambda candidate: candidate[0].utility,
        )[1]

        decisions = {name: router.route(task) for name, router in routers.items()}
        plans.update({name: decision_plan(decision) for name, decision in decisions.items()})

        results = {
            name: evaluate(task, difficulty, plan, agent_map)
            for name, plan in plans.items()
        }
        for name, result in results.items():
            _accumulate(metrics[name], result, task)
        for name, decision in decisions.items():
            route_mix[name][decision.mode.value] += 1

        for name in LEARNED_STRATEGIES:
            result = results[name]
            if index < window:
                _accumulate(curve_sums[name]["first"], result, task)
            if index >= tasks - window:
                _accumulate(curve_sums[name]["last"], result, task)
            routers[name].record_outcome(
                decisions[name],
                ExecutionOutcome(
                    result.quality,
                    agent_scores=result.agent_scores,
                    requirement_scores=result.requirement_scores,
                    actual_cost=result.cost,
                    actual_latency_ms=result.latency_ms,
                ),
            )

    averages = {name: _average(row, tasks) for name, row in metrics.items()}
    updates = {name: router.success_model.updates for name, router in routers.items()}
    curves = {
        name: {
            "first": _average(windows["first"], window),
            "last": _average(windows["last"], window),
        }
        for name, windows in curve_sums.items()
    }
    return averages, route_mix, updates, curves


def _summarize_values(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "population_stddev": pstdev(values)}


def summarize_suite(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    tasks_per_seed: int = 500,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one benchmark seed is required")
    if tasks_per_seed <= 0:
        raise ValueError("tasks_per_seed must be positive")
    simulations = [simulate(seed, tasks_per_seed) for seed in seeds]
    runs = [result[0] for result in simulations]

    strategies = {
        name: {
            metric: _summarize_values(
                [run_metrics[name][metric] for run_metrics in runs]
            )
            for metric in METRICS
        }
        for name in STRATEGIES
    }
    route_mix_by_strategy: dict[str, Counter[str]] = {
        name: Counter() for name in ("static_sage", *LEARNED_STRATEGIES)
    }
    for _, mixes, _, _ in simulations:
        for name, modes in mixes.items():
            route_mix_by_strategy[name].update(modes)

    learning_curve = {
        name: {
            period: {
                metric: _summarize_values(
                    [simulation[3][name][period][metric] for simulation in simulations]
                )
                for metric in METRICS
            }
            for period in ("first", "last")
        }
        for name in LEARNED_STRATEGIES
    }
    model_updates = {
        name: [simulation[2][name] for simulation in simulations]
        for name in ("static_sage", *LEARNED_STRATEGIES)
    }
    return {
        "schema_version": 2,
        "simulator": "held_out_geometric_v2",
        "seeds": list(seeds),
        "tasks_per_seed": tasks_per_seed,
        "total_tasks": len(seeds) * tasks_per_seed,
        "learning_curve_window": min(100, max(1, tasks_per_seed // 2)),
        "strategies": strategies,
        "route_mix_by_strategy": {
            name: dict(sorted(modes.items()))
            for name, modes in route_mix_by_strategy.items()
        },
        "model_updates": model_updates,
        "learning_curve": learning_curve,
        "ablation_design": {
            "static_sage": "informed prior, no updates, no exploration",
            "learned_no_explore": "informed prior, updates, no exploration",
            "learned_explore": "informed prior, updates, Thompson exploration",
            "learned_random_init": "weak random prior, updates, no exploration",
        },
    }


def run_suite(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    tasks_per_seed: int = 500,
) -> dict[str, object]:
    summary = summarize_suite(seeds, tasks_per_seed)
    print(
        f"tasks: {summary['total_tasks']} "
        f"({len(seeds)} seeds x {tasks_per_seed}; held-out geometric evaluator)"
    )
    print("strategy              quality       utility       cost/budget  deadline-miss")
    strategies = summary["strategies"]
    assert isinstance(strategies, dict)
    for name, values in strategies.items():
        print(
            f"{name:21s} {values['quality']['mean']:.3f}"
            f"+/-{values['quality']['population_stddev']:.3f}"
            f"  {values['utility']['mean']:.3f}"
            f"+/-{values['utility']['population_stddev']:.3f}"
            f"    {values['cost']['mean']:.3f}"
            f"+/-{values['cost']['population_stddev']:.3f}"
            f"      {values['misses']['mean']:4.1f}%"
        )
    window = summary["learning_curve_window"]
    curve = summary["learning_curve"]
    assert isinstance(curve, dict)
    print(f"learning curve: first vs last {window} tasks per seed")
    for name in LEARNED_STRATEGIES:
        print(
            f"{name:21s} quality {curve[name]['first']['quality']['mean']:.3f}"
            f" -> {curve[name]['last']['quality']['mean']:.3f};"
            f" utility {curve[name]['first']['utility']['mean']:.3f}"
            f" -> {curve[name]['last']['utility']['mean']:.3f}"
        )
    return summary


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated deterministic seeds (default: 3,7,11,19,23)",
    )
    parser.add_argument(
        "--tasks-per-seed",
        type=int,
        default=500,
        help="number of synthetic tasks for every seed (default: 500)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_path",
        help="write the complete summary to this JSON file",
    )
    args = parser.parse_args()
    if args.tasks_per_seed <= 0:
        parser.error("--tasks-per-seed must be positive")
    summary = run_suite(args.seeds, args.tasks_per_seed)
    if args.json_path is not None:
        args.json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"json summary: {args.json_path}")


if __name__ == "__main__":
    main()
