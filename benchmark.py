"""Small deterministic simulation comparing SAGE with simple routing baselines."""

from collections import Counter
import random

from sprix_sage import Agent, Requirement, SAGERouter, Task


def true_success(agent: Agent, requirements: tuple[Requirement, ...]) -> float:
    weight = sum(item.weight for item in requirements)
    return sum(item.weight * agent.skills.get(item.name, 0.0) for item in requirements) / weight


def true_team_success(team: list[Agent], requirements: tuple[Requirement, ...]) -> float:
    weight = sum(item.weight for item in requirements)
    score = 0.0
    for item in requirements:
        miss = 1.0
        for agent in team:
            miss *= 1.0 - agent.skills.get(item.name, 0.0)
        score += item.weight * (1.0 - miss)
    return score / weight


def run(seed: int = 11, tasks: int = 250) -> None:
    rng = random.Random(seed)
    agents = [
        Agent("generalist", {"code": 0.70, "research": 0.72, "vision": 0.58}, 0.05, 700),
        Agent("coder", {"code": 0.96, "research": 0.45, "vision": 0.30}, 0.10, 950),
        Agent("researcher", {"code": 0.42, "research": 0.97, "vision": 0.52}, 0.09, 1100),
        Agent("vision", {"code": 0.35, "research": 0.55, "vision": 0.96}, 0.11, 1000),
    ]
    agent_map = {agent.agent_id: agent for agent in agents}
    router = SAGERouter(agents, "generalist", max_collaborators=2)
    modes: Counter[str] = Counter()
    sage_quality = self_quality = best_single_quality = sage_cost = 0.0

    for index in range(tasks):
        chosen = rng.sample(["code", "research", "vision"], k=rng.choice((1, 2, 2, 3)))
        requirements = tuple(Requirement(name, 1 / len(chosen), minimum=0.60) for name in chosen)
        task = Task(
            f"task-{index}",
            requirements,
            budget=rng.choice((0.08, 0.15, 0.35)),
            deadline_ms=3500,
            progress=rng.random() * 0.8,
            coordination_overhead=rng.choice((0.04, 0.08, 0.15)),
        )
        decision = router.route(task)
        modes[decision.mode.value] += 1
        sage_quality += true_team_success([agent_map[item] for item in decision.agents], requirements)
        sage_cost += decision.cost
        self_quality += true_success(agents[0], requirements)
        best_single_quality += max(true_success(agent, requirements) for agent in agents)
        router.record_outcome(decision, rng.random() < decision.success_probability)

    print(f"tasks: {tasks}")
    print(f"mean capability proxy - self only : {self_quality / tasks:.3f}")
    print(f"mean capability proxy - best solo : {best_single_quality / tasks:.3f}")
    print(f"mean capability proxy - SAGE      : {sage_quality / tasks:.3f}")
    print(f"mean SAGE cost                     : {sage_cost / tasks:.3f}")
    print(f"route distribution                 : {dict(modes)}")


if __name__ == "__main__":
    run()
