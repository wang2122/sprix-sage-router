import unittest

from sprix_sage import Agent, Mode, Requirement, SAGERouter, Task


class SAGERouterTests(unittest.TestCase):
    def test_self_for_easy_task_with_expensive_peer(self) -> None:
        agents = [
            Agent("current", {"writing": 0.94}, 0.02, 300),
            Agent("peer", {"writing": 0.97}, 0.80, 500),
        ]
        task = Task("easy", (Requirement("writing", minimum=0.60),), budget=0.50, deadline_ms=2000)
        decision = SAGERouter(agents, "current").route(task)
        self.assertEqual(decision.mode, Mode.SELF)

    def test_handoff_to_clear_specialist(self) -> None:
        agents = [
            Agent("current", {"security": 0.12}, 0.02, 300),
            Agent("specialist", {"security": 0.99}, 0.05, 450),
        ]
        task = Task(
            "audit",
            (Requirement("security", minimum=0.80),),
            budget=0.20,
            deadline_ms=2000,
            progress=0.05,
        )
        decision = SAGERouter(agents, "current").route(task)
        self.assertEqual(decision.mode, Mode.HANDOFF)
        self.assertEqual(decision.agents, ("specialist",))

    def test_collaboration_for_complementary_skills(self) -> None:
        agents = [
            Agent("current", {"planning": 0.96, "coding": 0.12}, 0.02, 300),
            Agent("coder", {"planning": 0.10, "coding": 0.98}, 0.03, 400),
        ]
        task = Task(
            "feature",
            (Requirement("planning", 0.5, 0.75), Requirement("coding", 0.5, 0.75)),
            budget=0.20,
            deadline_ms=2000,
            progress=0.55,
            coordination_overhead=0.02,
        )
        decision = SAGERouter(agents, "current").route(task)
        self.assertEqual(decision.mode, Mode.COLLABORATE)
        self.assertEqual(set(decision.agents), {"current", "coder"})

    def test_permissions_are_a_hard_filter(self) -> None:
        agents = [
            Agent("current", {"finance": 0.75}, 0.02, 300, frozenset({"ledger:write"})),
            Agent("untrusted", {"finance": 1.0}, 0.00, 100),
        ]
        task = Task(
            "settle",
            (Requirement("finance"),),
            required_permissions=frozenset({"ledger:write"}),
            budget=0.20,
            deadline_ms=2000,
        )
        decision = SAGERouter(agents, "current").route(task)
        self.assertNotIn("untrusted", decision.agents)

    def test_outcomes_update_reliability_and_pair_synergy(self) -> None:
        agents = [
            Agent("current", {"planning": 0.9, "coding": 0.2}, 0.02, 300),
            Agent("coder", {"planning": 0.2, "coding": 0.9}, 0.02, 350),
        ]
        task = Task(
            "learn",
            (Requirement("planning", 0.5), Requirement("coding", 0.5)),
            budget=0.20,
            deadline_ms=2000,
            coordination_overhead=0.01,
        )
        router = SAGERouter(agents, "current")
        decision = router.route(task)
        before = router.reliability["current"].mean
        router.record_outcome(decision, True)
        self.assertGreater(router.reliability["current"].mean, before)
        if len(decision.agents) > 1:
            pair = tuple(sorted(decision.agents))
            self.assertGreater(router.synergy[pair].mean, 0.5)


if __name__ == "__main__":
    unittest.main()
