"""Replan after failure and persist learned evidence across router restarts."""

import json

from sprix_sage import Agent, ExecutionOutcome, ExecutionState, Requirement, SAGERouter, Task

agents = [
    Agent("generalist", {"coding": 0.90}, 0.02, 400),
    Agent("specialist", {"coding": 0.94}, 0.12, 650),
]
task = Task(
    "recover-build",
    (Requirement("coding", minimum=0.75),),
    budget=0.20,
    deadline_ms=2000,
)

router = SAGERouter(agents, incumbent_id="generalist")
initial = router.route(task)
router.record_outcome(
    initial,
    ExecutionOutcome(
        success=0.1,
        requirement_scores={"coding": 0.1},
        actual_cost=0.05,
        actual_latency_ms=750,
    ),
)

state = ExecutionState(
    active_agents=initial.agents,
    active_mode=initial.mode,
    progress=0.4,
    failed_agents=frozenset({"generalist"}),
    failure_count=1,
)
replanned = router.route(task, state=state)

snapshot_json = json.dumps(router.export_state())
restored = SAGERouter(agents, incumbent_id="generalist")
restored.restore_state(json.loads(snapshot_json))

print(f"initial : {initial.mode.value} -> {initial.agents}")
print(f"replan  : {replanned.mode.value} -> {replanned.agents}")
print(f"updates : {restored.success_model.updates}")
