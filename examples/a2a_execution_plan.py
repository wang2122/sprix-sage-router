"""Create an auditable route and a transport-neutral A2A execution plan."""

import json

from sprix_a2a import execution_plan, profile_from_agent_card
from sprix_sage import Requirement, SAGERouter, Task

cards = [
    {
        "name": "Planning agent",
        "url": "https://agents.example/planner",
        "version": "1.0.0",
        "skills": [{"id": "planning", "name": "Technical planning"}],
    },
    {
        "name": "Coding agent",
        "url": "https://agents.example/coder",
        "version": "1.0.0",
        "skills": [{"id": "coding", "name": "Python implementation"}],
    },
]

profiles = [
    profile_from_agent_card(
        cards[0],
        agent_id="planner",
        skill_scores={"planning": 0.94},
        cost=0.05,
        latency_ms=700,
        permissions=frozenset({"repo:read"}),
    ),
    profile_from_agent_card(
        cards[1],
        agent_id="coder",
        skill_scores={"coding": 0.97},
        cost=0.09,
        latency_ms=950,
        permissions=frozenset({"repo:read"}),
    ),
]

task = Task(
    "implement-endpoint",
    requirements=(
        Requirement("planning", weight=0.35, minimum=0.70),
        Requirement("coding", weight=0.65, minimum=0.80, depends_on=("planning",)),
    ),
    budget=0.25,
    deadline_ms=3000,
    required_permissions=frozenset({"repo:read"}),
)

router = SAGERouter([profile.agent for profile in profiles], incumbent_id="planner")
trace = router.route_with_trace(task)
plan = execution_plan(task, trace.selected)

print(json.dumps({"trace": trace.to_dict(), "plan": plan.to_dict()}, indent=2))
