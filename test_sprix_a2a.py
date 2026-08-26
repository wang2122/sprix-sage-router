import json
import unittest

from sprix_a2a import execution_plan, profile_from_agent_card
from sprix_sage import Agent, Requirement, SAGERouter, Task


class A2AAdapterTests(unittest.TestCase):
    def test_agent_card_profile_requires_local_scores_for_declared_skills(self) -> None:
        card = {
            "name": "Secure coder",
            "url": "https://agents.example/coder",
            "version": "1.2.0",
            "skills": [
                {"id": "planning", "name": "Planning"},
                {"id": "coding", "name": "Coding"},
            ],
        }

        profile = profile_from_agent_card(
            card,
            agent_id="coder",
            skill_scores={"coding": 0.91},
            cost=0.08,
            latency_ms=850,
            permissions=frozenset({"repo:read"}),
        )

        self.assertEqual(profile.agent.skills, {"coding": 0.91})
        self.assertEqual(profile.declared_skills, ("planning", "coding"))
        json.dumps(profile.to_dict(), sort_keys=True)

    def test_agent_card_profile_rejects_undeclared_skill_scores(self) -> None:
        card = {"name": "Coder", "skills": [{"id": "coding"}]}

        with self.assertRaisesRegex(ValueError, "undeclared"):
            profile_from_agent_card(
                card,
                agent_id="coder",
                skill_scores={"security": 0.9},
                cost=0.08,
                latency_ms=850,
            )

    def test_route_converts_to_transport_neutral_execution_plan(self) -> None:
        agents = [
            Agent("planner", {"plan": 0.98, "build": 0.10}, 0.02, 600),
            Agent("builder", {"plan": 0.10, "build": 0.99}, 0.03, 700),
        ]
        task = Task(
            "ship-feature",
            (
                Requirement("plan", 0.35, 0.75),
                Requirement("build", 0.65, 0.75, depends_on=("plan",)),
            ),
            budget=0.20,
            deadline_ms=3000,
            coordination_overhead=0.02,
        )
        decision = SAGERouter(agents, "planner").route(task)

        plan = execution_plan(task, decision)

        self.assertEqual(plan.owner_agent_id, "planner")
        self.assertEqual(plan.steps[1].depends_on, ("plan",))
        self.assertEqual(plan.steps[1].agent_id, "builder")
        self.assertIn(("planner", "builder"), plan.communication_edges)
        json.dumps(plan.to_dict(), sort_keys=True)


if __name__ == "__main__":
    unittest.main()
