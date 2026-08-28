"""Deterministic benchmark for checkpoint-aware mid-execution rerouting.

The suite has two parts.  A controlled progress intervention varies only the
fraction of one in-flight requirement that is complete.  A trajectory replay
suite samples completed DAG nodes, concrete artifact portability, sunk
resources, and failures.  Every strategy is scored by the separate
``benchmark_dynamic_evaluator`` module.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations, product
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, cast

from benchmark import (
    DEFAULT_SEEDS,
    Plan,
    eligible_agents,
    generate_task,
    greedy_team_plan,
    make_agents,
    parse_seeds,
    plan_mode,
    visible_feasible,
    visible_score,
)
from benchmark_dynamic_evaluator import (
    Artifact,
    DynamicResult,
    ExecutionCheckpoint,
    checkpoint_progress,
    evaluate_recovery,
    latent_requirement_score,
)
from sprix_sage import Agent, ExecutionState, Mode, Requirement, SAGERouter, Task

DYNAMIC_STRATEGIES = (
    "progress_aware",
    "progress_masked",
    "always_continue",
    "always_handoff",
    "static_greedy",
    "static_coalition",
    "dynamic_oracle",
)
DYNAMIC_METRICS = (
    "quality",
    "utility",
    "added_cost",
    "recovery_latency",
    "wasted_work",
    "switch_rate",
    "deadline_miss",
)
PROGRESS_POINTS = tuple(round(index / 10, 1) for index in range(10))


@dataclass(frozen=True)
class DynamicCase:
    task: Task
    difficulty: float
    checkpoint: ExecutionCheckpoint


def _topological_requirements(task: Task) -> tuple[Requirement, ...]:
    pending = {item.name: item for item in task.requirements}
    emitted: set[str] = set()
    ordered: list[Requirement] = []
    while pending:
        ready = sorted(
            (
                item
                for item in pending.values()
                if set(item.depends_on).issubset(emitted)
            ),
            key=lambda item: item.name,
        )
        if not ready:
            raise RuntimeError("task requirements must form a DAG")
        for item in ready:
            ordered.append(item)
            emitted.add(item.name)
            pending.pop(item.name)
    return tuple(ordered)


def _remaining_task(case: DynamicCase) -> Task:
    completed = set(case.checkpoint.completed_artifacts)
    requirements = tuple(
        replace(
            item,
            depends_on=tuple(name for name in item.depends_on if name not in completed),
        )
        for item in case.task.requirements
        if item.name not in completed
    )
    remaining_budget = max(0.015, case.task.budget - case.checkpoint.elapsed_cost)
    remaining_deadline = max(
        120.0,
        case.task.deadline_ms - case.checkpoint.elapsed_latency_ms,
    )
    return replace(
        case.task,
        requirements=requirements,
        budget=remaining_budget,
        deadline_ms=remaining_deadline,
        progress=0.0,
    )


def _routing_task(case: DynamicCase) -> Task:
    remaining_budget = max(0.015, case.task.budget - case.checkpoint.elapsed_cost)
    remaining_deadline = max(
        120.0,
        case.task.deadline_ms - case.checkpoint.elapsed_latency_ms,
    )
    return replace(
        case.task,
        budget=remaining_budget,
        deadline_ms=remaining_deadline,
        progress=0.0,
    )


def _artifact_transferability(checkpoint: ExecutionCheckpoint) -> float:
    values = [
        artifact.portability
        for artifact in checkpoint.completed_artifacts.values()
    ]
    values.append(checkpoint.inflight_portability)
    return sum(values) / len(values)


def _execution_state(
    case: DynamicCase,
    *,
    mask_progress: bool,
) -> ExecutionState:
    checkpoint = case.checkpoint
    active_assignments = {
        name: artifact.producer
        for name, artifact in checkpoint.completed_artifacts.items()
    }
    active_assignments[checkpoint.inflight_requirement] = checkpoint.inflight_agent
    artifact_transferability = {
        name: artifact.portability
        for name, artifact in checkpoint.completed_artifacts.items()
    }
    artifact_transferability[checkpoint.inflight_requirement] = (
        checkpoint.inflight_portability
    )
    return ExecutionState(
        active_agents=checkpoint.active_agents,
        active_mode=checkpoint.active_mode,
        completed_requirements=frozenset(checkpoint.completed_artifacts),
        progress=0.0 if mask_progress else checkpoint_progress(case.task, checkpoint),
        transferable_context=_artifact_transferability(checkpoint),
        active_assignments=active_assignments,
        inflight_requirement=checkpoint.inflight_requirement,
        inflight_progress=0.0 if mask_progress else checkpoint.inflight_fraction,
        inflight_quality=checkpoint.inflight_quality,
        artifact_transferability=artifact_transferability,
        failed_agents=checkpoint.failed_agents,
        failure_count=checkpoint.failure_count,
    )


def _all_visible_plans(
    task: Task,
    agents: list[Agent],
    max_team_size: int = 3,
) -> Iterable[Plan]:
    allowed = sorted(agent.agent_id for agent in eligible_agents(task, agents))
    requirement_names = tuple(item.name for item in task.requirements)
    for size in range(1, min(max_team_size, len(allowed)) + 1):
        for team in combinations(allowed, size):
            # Match SAGE's public action space: multi-agent collaboration keeps
            # the incumbent; otherwise the action is a single-agent handoff.
            if size > 1 and "generalist" not in team:
                continue
            for owners in product(team, repeat=len(requirement_names)):
                used = tuple(agent_id for agent_id in team if agent_id in owners)
                if len(used) != size:
                    continue
                assignments = dict(zip(requirement_names, owners))
                yield Plan(plan_mode(used), used, assignments)


def _communication_edges(task: Task, assignments: Mapping[str, str]) -> int:
    return len(
        {
            (assignments[dependency], assignments[item.name])
            for item in task.requirements
            for dependency in item.depends_on
            if assignments[dependency] != assignments[item.name]
        }
    )


def static_coalition_plan(
    task: Task,
    agents: list[Agent],
    agent_map: Mapping[str, Agent],
) -> Plan:
    """Enumerate a bounded static coalition and its DAG-induced topology."""

    candidates: list[tuple[float, Plan]] = []
    fallback: list[tuple[float, Plan]] = []
    for plan in _all_visible_plans(task, agents):
        score = visible_score(task, plan.agents, plan.assignments, agent_map)
        score -= 0.025 * _communication_edges(task, plan.assignments)
        item = (score, plan)
        fallback.append(item)
        if visible_feasible(task, plan.agents, plan.assignments, agent_map):
            candidates.append(item)
    if not fallback:
        raise RuntimeError("no authorized static coalition is available")
    return max(candidates or fallback, key=lambda item: item[0])[1]


def _continue_plan(task: Task) -> Plan:
    assignments = {item.name: "generalist" for item in task.requirements}
    return Plan(Mode.SELF, ("generalist",), assignments)


def _handoff_plan(
    task: Task,
    agents: list[Agent],
    agent_map: Mapping[str, Agent],
) -> Plan:
    peers = [
        agent
        for agent in eligible_agents(task, agents)
        if agent.agent_id != "generalist"
    ]
    if not peers:
        return _continue_plan(task)
    candidates: list[tuple[float, Plan]] = []
    for agent in peers:
        assignments = {item.name: agent.agent_id for item in task.requirements}
        plan = Plan(Mode.HANDOFF, (agent.agent_id,), assignments)
        candidates.append(
            (visible_score(task, plan.agents, plan.assignments, agent_map), plan)
        )
    return max(candidates, key=lambda item: item[0])[1]


def _dynamic_oracle_plan(
    case: DynamicCase,
    remaining_task: Task,
    agents: list[Agent],
    agent_map: Mapping[str, Agent],
) -> Plan:
    candidates: list[tuple[DynamicResult, Plan]] = []
    for plan in _all_visible_plans(remaining_task, agents):
        if set(plan.agents) & set(case.checkpoint.failed_agents):
            continue
        result = evaluate_recovery(
            case.task,
            case.difficulty,
            plan.mode,
            plan.agents,
            plan.assignments,
            agent_map,
            case.checkpoint,
        )
        candidates.append((result, plan))
    if not candidates:
        raise RuntimeError("no hidden-state oracle plan is available")
    feasible = [
        item
        for item in candidates
        if item[0].total_cost <= case.task.budget and not item[0].deadline_miss
    ]
    return max(feasible or candidates, key=lambda item: item[0].utility)[1]


def plans_for_case(
    case: DynamicCase,
    agents: list[Agent],
) -> dict[str, Plan]:
    """Build every baseline with a common candidate registry and constraints."""

    agent_map = {agent.agent_id: agent for agent in agents}
    routing_task = _routing_task(case)
    remaining_task = _remaining_task(case)
    aware_router = SAGERouter(
        agents,
        "generalist",
        max_collaborators=3,
        beam_width=10,
        exploration=False,
    )
    masked_router = SAGERouter(
        agents,
        "generalist",
        max_collaborators=3,
        beam_width=10,
        exploration=False,
    )
    aware = aware_router.route(
        routing_task,
        state=_execution_state(case, mask_progress=False),
    )
    masked = masked_router.route(
        routing_task,
        state=_execution_state(case, mask_progress=True),
    )
    return {
        "progress_aware": Plan(aware.mode, aware.agents, aware.assignments),
        "progress_masked": Plan(masked.mode, masked.agents, masked.assignments),
        "always_continue": _continue_plan(remaining_task),
        "always_handoff": _handoff_plan(remaining_task, agents, agent_map),
        "static_greedy": greedy_team_plan(remaining_task, agents, agent_map),
        "static_coalition": static_coalition_plan(remaining_task, agents, agent_map),
        "dynamic_oracle": _dynamic_oracle_plan(
            case,
            remaining_task,
            agents,
            agent_map,
        ),
    }


def evaluate_case(
    case: DynamicCase,
    agents: list[Agent],
) -> tuple[dict[str, DynamicResult], dict[str, Plan]]:
    plans = plans_for_case(case, agents)
    agent_map = {agent.agent_id: agent for agent in agents}
    results = {
        name: evaluate_recovery(
            case.task,
            case.difficulty,
            plan.mode,
            plan.agents,
            plan.assignments,
            agent_map,
            case.checkpoint,
        )
        for name, plan in plans.items()
    }
    return results, plans


def _make_checkpoint(
    task: Task,
    difficulty: float,
    rng: random.Random,
) -> ExecutionCheckpoint:
    ordered = _topological_requirements(task)
    total_weight = sum(item.weight for item in ordered)
    target_work = rng.uniform(0.08, 0.88) * total_weight
    completed: dict[str, Artifact] = {}
    consumed = 0.0
    inflight = ordered[0]
    for index, item in enumerate(ordered):
        if index < len(ordered) - 1 and consumed + item.weight <= target_work:
            portability = rng.uniform(0.15, 0.95)
            completed[item.name] = Artifact(
                "generalist",
                latent_requirement_score("generalist", item, difficulty),
                portability,
            )
            consumed += item.weight
            continue
        inflight = item
        break
    inflight_fraction = max(
        0.02,
        min(0.95, (target_work - consumed) / inflight.weight),
    )
    progress = (consumed + inflight.weight * inflight_fraction) / total_weight
    generalist = next(agent for agent in make_agents() if agent.agent_id == "generalist")
    elapsed_cost = generalist.cost * (0.12 + 0.88 * progress**0.88)
    elapsed_latency = generalist.latency_ms * progress**0.76
    failed = rng.random() < 0.14
    return ExecutionCheckpoint(
        active_agents=("generalist",),
        active_mode=Mode.SELF,
        completed_artifacts=completed,
        inflight_requirement=inflight.name,
        inflight_agent="generalist",
        inflight_fraction=inflight_fraction,
        inflight_quality=latent_requirement_score("generalist", inflight, difficulty),
        inflight_portability=rng.uniform(0.10, 0.90),
        elapsed_cost=elapsed_cost,
        elapsed_latency_ms=elapsed_latency,
        failed_agents=frozenset({"generalist"}) if failed else frozenset(),
        failure_count=1 if failed else 0,
    )


def generate_dynamic_case(rng: random.Random, index: int) -> DynamicCase:
    task, difficulty = generate_task(rng, index)
    task = replace(task, progress=0.0)
    checkpoint = _make_checkpoint(task, difficulty, rng)
    return DynamicCase(task, difficulty, checkpoint)


def _progress_case(
    rng: random.Random,
    index: int,
    progress: float,
) -> DynamicCase:
    # The controlled intervention uses a near-substitutable incumbent and
    # specialist on one fixed requirement family. Large capability gaps belong
    # in the trajectory suite and can justify switching even very late.
    skill = "research"
    task = Task(
        f"progress-{index}",
        (Requirement(skill, minimum=rng.uniform(0.62, 0.76)),),
        value=1.0,
        budget=0.28,
        deadline_ms=2200,
        required_permissions=frozenset({"public"}),
        risk_tolerance=0.5,
        progress=0.0,
        handoff_friction=rng.uniform(0.45, 0.80),
        coordination_overhead=0.04,
        context_transferability=rng.uniform(0.15, 0.55),
        replan_friction=0.04,
    )
    requirement = task.requirements[0]
    difficulty = rng.uniform(0.35, 0.70)
    portability = task.context_transferability
    checkpoint = ExecutionCheckpoint(
        active_agents=("generalist",),
        active_mode=Mode.SELF,
        completed_artifacts={},
        inflight_requirement=skill,
        inflight_agent="generalist",
        inflight_fraction=progress,
        inflight_quality=latent_requirement_score("generalist", requirement, difficulty),
        inflight_portability=portability,
        elapsed_cost=0.0,
        elapsed_latency_ms=0.0,
    )
    return DynamicCase(task, difficulty, checkpoint)


def _empty_metrics() -> dict[str, float]:
    return {name: 0.0 for name in DYNAMIC_METRICS}


def _accumulate(row: dict[str, float], result: DynamicResult, task: Task) -> None:
    row["quality"] += result.quality
    row["utility"] += result.utility
    row["added_cost"] += result.added_cost / task.budget
    row["recovery_latency"] += result.recovery_latency_ms / task.deadline_ms
    row["wasted_work"] += result.wasted_work
    row["switch_rate"] += float(result.switched)
    row["deadline_miss"] += float(result.deadline_miss)


def _average_metrics(row: Mapping[str, float], count: int) -> dict[str, float]:
    result = {name: row[name] / count for name in DYNAMIC_METRICS}
    result["switch_rate"] *= 100.0
    result["deadline_miss"] *= 100.0
    return result


def simulate_trajectories(
    seed: int,
    trajectories: int,
) -> tuple[dict[str, dict[str, float]], dict[str, Counter[str]]]:
    if trajectories <= 0:
        raise ValueError("trajectories must be positive")
    rng = random.Random(seed)
    agents = make_agents()
    metrics = {name: _empty_metrics() for name in DYNAMIC_STRATEGIES}
    modes: dict[str, Counter[str]] = {
        name: Counter() for name in DYNAMIC_STRATEGIES
    }
    for index in range(trajectories):
        case = generate_dynamic_case(rng, index)
        results, plans = evaluate_case(case, agents)
        for name, result in results.items():
            _accumulate(metrics[name], result, case.task)
            modes[name][plans[name].mode.value] += 1
    return (
        {name: _average_metrics(row, trajectories) for name, row in metrics.items()},
        modes,
    )


def _summarize(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "population_stddev": pstdev(values)}


def summarize_trajectories(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    trajectories_per_seed: int = 200,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one trajectory seed is required")
    runs = [simulate_trajectories(seed, trajectories_per_seed) for seed in seeds]
    strategies = {
        name: {
            metric: _summarize([run[0][name][metric] for run in runs])
            for metric in DYNAMIC_METRICS
        }
        for name in DYNAMIC_STRATEGIES
    }
    mode_mix: dict[str, Counter[str]] = {
        name: Counter() for name in DYNAMIC_STRATEGIES
    }
    for _, run_modes in runs:
        for name, counts in run_modes.items():
            mode_mix[name].update(counts)
    return {
        "seeds": list(seeds),
        "trajectories_per_seed": trajectories_per_seed,
        "total_trajectories": len(seeds) * trajectories_per_seed,
        "strategies": strategies,
        "mode_mix": {
            name: dict(sorted(counts.items())) for name, counts in mode_mix.items()
        },
    }


def summarize_progress_sweep(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    cases_per_seed: int = 40,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one progress-sweep seed is required")
    if cases_per_seed <= 0:
        raise ValueError("cases_per_seed must be positive")
    agents = make_agents()
    points: list[dict[str, object]] = []
    for progress in PROGRESS_POINTS:
        per_seed: list[dict[str, dict[str, float]]] = []
        for seed in seeds:
            rng = random.Random(seed)
            totals = {name: _empty_metrics() for name in DYNAMIC_STRATEGIES}
            for index in range(cases_per_seed):
                case = _progress_case(rng, index, progress)
                results, _ = evaluate_case(case, agents)
                for name, result in results.items():
                    _accumulate(totals[name], result, case.task)
            per_seed.append(
                {
                    name: _average_metrics(row, cases_per_seed)
                    for name, row in totals.items()
                }
            )
        strategy_summary = {
            name: {
                metric: _summarize([run[name][metric] for run in per_seed])
                for metric in DYNAMIC_METRICS
            }
            for name in DYNAMIC_STRATEGIES
        }
        aware_utility = strategy_summary["progress_aware"]["utility"]["mean"]
        masked_utility = strategy_summary["progress_masked"]["utility"]["mean"]
        points.append(
            {
                "progress": progress,
                "strategies": strategy_summary,
                "aware_minus_masked_utility": aware_utility - masked_utility,
            }
        )
    return {
        "seeds": list(seeds),
        "cases_per_seed": cases_per_seed,
        "controlled_variable": "inflight_fraction",
        "held_constant": [
            "task",
            "agent registry",
            "difficulty",
            "artifact portability",
            "budget",
            "deadline",
        ],
        "points": points,
    }


def summarize_dynamic_suite(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    trajectories_per_seed: int = 200,
    sweep_cases_per_seed: int = 40,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluator": "checkpoint_artifact_v1",
        "trajectory_replay": summarize_trajectories(seeds, trajectories_per_seed),
        "progress_intervention": summarize_progress_sweep(seeds, sweep_cases_per_seed),
        "scope": (
            "synthetic falsification study; not evidence of production or "
            "real-endpoint superiority"
        ),
    }


def write_progress_svg(summary: Mapping[str, object], path: Path) -> None:
    """Render the controlled progress intervention without plot dependencies."""

    intervention = cast(Mapping[str, Any], summary["progress_intervention"])
    points = cast(list[Mapping[str, Any]], intervention["points"])
    width, height = 900, 520
    left, right, top, bottom = 88, 32, 70, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [
        float(point["strategies"][name]["utility"]["mean"])
        for point in points
        for name in ("progress_aware", "progress_masked")
    ]
    y_min = math.floor((min(values) - 0.01) * 20.0) / 20.0
    y_max = math.ceil((max(values) + 0.01) * 20.0) / 20.0

    def x_position(progress: float) -> float:
        return left + plot_width * progress / 0.9

    def y_position(value: float) -> float:
        return top + plot_height * (y_max - value) / (y_max - y_min)

    def polyline(name: str) -> str:
        return " ".join(
            f"{x_position(float(point['progress'])):.1f},"
            f"{y_position(float(point['strategies'][name]['utility']['mean'])):.1f}"
            for point in points
        )

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520" role="img" aria-labelledby="title desc">',
        '<title id="title">Utility across in-flight progress</title>',
        '<desc id="desc">Progress-aware SAGE utility rises above an otherwise identical progress-masked baseline as in-flight completion increases from zero to ninety percent in a controlled synthetic intervention.</desc>',
        '<rect width="900" height="520" fill="#ffffff"/>',
        '<text x="88" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#172033">Checkpoint-aware continuation value</text>',
        '<text x="88" y="56" font-family="Arial, sans-serif" font-size="13" fill="#586174">Controlled synthetic intervention; mean over five seeds</text>',
    ]
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4
        y = y_position(value)
        lines.extend(
            (
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dfe4ec" stroke-width="1"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#586174">{value:.2f}</text>',
            )
        )
    for point in points:
        progress = float(point["progress"])
        x = x_position(progress)
        lines.extend(
            (
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#f0f2f6" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#586174">{progress:.1f}</text>',
            )
        )
    lines.extend(
        (
            f'<polyline points="{polyline("progress_masked")}" fill="none" stroke="#7b879a" stroke-width="3" stroke-dasharray="7 5"/>',
            f'<polyline points="{polyline("progress_aware")}" fill="none" stroke="#1267a5" stroke-width="3.5"/>',
        )
    )
    for point in points:
        progress = float(point["progress"])
        for name, color in (
            ("progress_masked", "#7b879a"),
            ("progress_aware", "#1267a5"),
        ):
            value = float(point["strategies"][name]["utility"]["mean"])
            lines.append(
                f'<circle cx="{x_position(progress):.1f}" cy="{y_position(value):.1f}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
            )
    lines.extend(
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#172033" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#172033" stroke-width="1.5"/>',
            f'<text x="{left+plot_width/2:.1f}" y="{height-26}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#172033">In-flight completion fraction</text>',
            f'<text x="24" y="{top+plot_height/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {top+plot_height/2:.1f})" font-family="Arial, sans-serif" font-size="14" fill="#172033">Held-out utility</text>',
            '<line x1="590" y1="30" x2="622" y2="30" stroke="#1267a5" stroke-width="3.5"/>',
            '<text x="630" y="34" font-family="Arial, sans-serif" font-size="13" fill="#172033">Progress-aware SAGE</text>',
            '<line x1="590" y1="52" x2="622" y2="52" stroke="#7b879a" stroke-width="3" stroke-dasharray="7 5"/>',
            '<text x="630" y="56" font-family="Arial, sans-serif" font-size="13" fill="#172033">Progress-masked SAGE</text>',
            '</svg>',
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    trajectories_per_seed: int = 200,
    sweep_cases_per_seed: int = 40,
) -> dict[str, object]:
    summary = summarize_dynamic_suite(
        seeds,
        trajectories_per_seed,
        sweep_cases_per_seed,
    )
    trajectory = summary["trajectory_replay"]
    assert isinstance(trajectory, dict)
    strategies = trajectory["strategies"]
    assert isinstance(strategies, dict)
    print(
        f"trajectories: {trajectory['total_trajectories']} "
        f"({len(seeds)} seeds x {trajectories_per_seed})"
    )
    print("strategy              utility       wasted-work  switch-rate  deadline-miss")
    for name in DYNAMIC_STRATEGIES:
        values = strategies[name]
        print(
            f"{name:21s} {values['utility']['mean']:.3f}"
            f"+/-{values['utility']['population_stddev']:.3f}"
            f"    {values['wasted_work']['mean']:.3f}"
            f"       {values['switch_rate']['mean']:5.1f}%"
            f"       {values['deadline_miss']['mean']:5.1f}%"
        )
    intervention = summary["progress_intervention"]
    assert isinstance(intervention, dict)
    print("progress intervention: aware-minus-masked utility")
    for point in intervention["points"]:
        print(f"p={point['progress']:.1f}  {point['aware_minus_masked_utility']:+.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated deterministic seeds (default: 3,7,11,19,23)",
    )
    parser.add_argument(
        "--trajectories-per-seed",
        type=int,
        default=200,
        help="checkpoint trajectories per seed (default: 200)",
    )
    parser.add_argument(
        "--sweep-cases-per-seed",
        type=int,
        default=40,
        help="fixed cases per progress point and seed (default: 40)",
    )
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument(
        "--svg",
        type=Path,
        dest="svg_path",
        help="write the controlled progress curve as an accessible SVG",
    )
    args = parser.parse_args()
    if args.trajectories_per_seed <= 0 or args.sweep_cases_per_seed <= 0:
        parser.error("trajectory and sweep case counts must be positive")
    summary = run_suite(
        args.seeds,
        args.trajectories_per_seed,
        args.sweep_cases_per_seed,
    )
    if args.json_path is not None:
        args.json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"json summary: {args.json_path}")
    if args.svg_path is not None:
        write_progress_svg(summary, args.svg_path)
        print(f"progress figure: {args.svg_path}")


if __name__ == "__main__":
    main()
