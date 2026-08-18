from sprix_sage import Agent, Bid, ExecutionOutcome, Requirement, SAGERouter, Task


agents = [
    Agent(
        "incumbent-planner",
        {"planning": 0.94, "coding": 0.58, "security": 0.34},
        cost=0.06,
        latency_ms=850,
        permissions=frozenset({"repo:read"}),
    ),
    Agent(
        "code-specialist",
        {"planning": 0.45, "coding": 0.97, "security": 0.60},
        cost=0.12,
        latency_ms=1200,
        permissions=frozenset({"repo:read"}),
    ),
    Agent(
        "security-reviewer",
        {"planning": 0.52, "coding": 0.63, "security": 0.98},
        cost=0.10,
        latency_ms=1000,
        permissions=frozenset({"repo:read"}),
    ),
]

task = Task(
    "secure-feature",
    requirements=(
        Requirement("planning", weight=0.25, minimum=0.60),
        Requirement("coding", weight=0.45, minimum=0.75, depends_on=("planning",)),
        Requirement("security", weight=0.30, minimum=0.75, depends_on=("coding",)),
    ),
    value=1.0,
    budget=0.30,
    deadline_ms=3500,
    required_permissions=frozenset({"repo:read"}),
    progress=0.45,
)

bids = [
    Bid("incumbent-planner", task.task_id, 0.06, 850, confidence=0.76),
    Bid("code-specialist", task.task_id, 0.12, 1200, confidence=0.91),
    Bid("security-reviewer", task.task_id, 0.10, 1000, confidence=0.88),
]

router = SAGERouter(agents, incumbent_id="incumbent-planner", max_collaborators=2)
decision = router.route(task, bids)

print(f"mode       : {decision.mode.value}")
print(f"agents     : {', '.join(decision.agents)}")
print(f"utility    : {decision.utility:.3f}")
print(f"p(success) : {decision.success_probability:.3f}")
print(f"cost       : {decision.cost:.3f}")
print(f"latency    : {decision.latency_ms:.0f} ms")
print(f"assignments: {dict(decision.assignments)}")
print(f"topology   : {decision.topology}")
print(f"reason     : {decision.explanation}")

# The router learns from granular evidence rather than assigning identical
# credit to every team member.
router.record_outcome(
    decision,
    ExecutionOutcome(
        success=0.88,
        requirement_scores={"planning": 0.94, "coding": 0.84, "security": 0.89},
        actual_cost=0.19,
        actual_latency_ms=1900,
    ),
)
