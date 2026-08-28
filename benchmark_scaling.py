"""Measure SAGE decision latency as the candidate registry grows."""

from __future__ import annotations

import argparse
import random
import time
from statistics import median

from sprix_sage import Agent, Requirement, SAGERouter, Task

REQUIREMENTS = ("plan", "code", "research", "security", "vision", "review")


def make_case(agent_count: int, seed: int = 17) -> tuple[list[Agent], Task]:
    if agent_count < 2:
        raise ValueError("agent_count must be at least two")
    rng = random.Random(seed)
    agents = [
        Agent(
            f"agent-{index:04d}",
            {name: rng.uniform(0.10, 0.99) for name in REQUIREMENTS},
            rng.uniform(0.01, 0.08),
            rng.uniform(500, 1400),
            frozenset({"public"}),
        )
        for index in range(agent_count)
    ]
    task = Task(
        "scaling",
        tuple(
            Requirement(
                name,
                1.0 / len(REQUIREMENTS),
                0.60,
                depends_on=(REQUIREMENTS[index - 1],) if index in (2, 4) else (),
            )
            for index, name in enumerate(REQUIREMENTS)
        ),
        budget=1.5,
        deadline_ms=6000,
        required_permissions=frozenset({"public"}),
    )
    return agents, task


def measure(
    agent_count: int,
    candidate_limit: int | None = 12,
    repeats: int = 3,
) -> float:
    samples: list[float] = []
    for repeat in range(repeats):
        agents, task = make_case(agent_count, seed=17 + repeat)
        router = SAGERouter(
            agents,
            agents[0].agent_id,
            beam_width=10,
            assignment_beam_width=4,
            max_collaborators=3,
            candidate_limit=candidate_limit,
            seed=repeat,
        )
        started = time.perf_counter()
        trace = router.route_with_trace(task)
        samples.append(1000.0 * (time.perf_counter() - started))
        expected = min(agent_count, candidate_limit or agent_count)
        if len(trace.eligible_agents) != expected:
            raise AssertionError("candidate prefilter returned an unexpected count")
    return median(samples)


def parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not counts or any(count < 2 for count in counts):
        raise argparse.ArgumentTypeError("counts must be comma-separated integers >= 2")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-counts", type=parse_counts, default=(5, 20, 40, 80))
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compare-unfiltered", action="store_true")
    args = parser.parse_args()
    print("agents  top-k  median-route-ms")
    for count in args.agent_counts:
        print(
            f"{count:6d} {min(count, args.candidate_limit):6d} "
            f"{measure(count, args.candidate_limit, args.repeats):15.2f}"
        )
        if args.compare_unfiltered and count > args.candidate_limit:
            print(
                f"{count:6d} {'all':>6s} "
                f"{measure(count, None, args.repeats):15.2f}"
            )


if __name__ == "__main__":
    main()
