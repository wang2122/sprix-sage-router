import unittest

from sprix_sage import (
    Agent,
    ExecutionOutcome,
    ExecutionState,
    Mode,
    Requirement,
    SAGERouter,
    Task,
)


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

    def test_requirement_dependencies_must_form_a_dag(self) -> None:
        with self.assertRaises(ValueError):
            Task(
                "cycle",
                (
                    Requirement("plan", depends_on=("build",)),
                    Requirement("build", depends_on=("plan",)),
                ),
            )

    def test_dag_route_assigns_roles_and_builds_topology(self) -> None:
        agents = [
            Agent("planner", {"plan": 0.98, "build": 0.10}, 0.02, 600),
            Agent("builder", {"plan": 0.10, "build": 0.99}, 0.03, 700),
        ]
        task = Task(
            "dag",
            (
                Requirement("plan", 0.35, 0.75),
                Requirement("build", 0.65, 0.75, depends_on=("plan",)),
            ),
            budget=0.20,
            deadline_ms=3000,
            coordination_overhead=0.02,
        )
        decision = SAGERouter(agents, "planner").route(task)
        self.assertEqual(decision.mode, Mode.COLLABORATE)
        self.assertEqual(decision.assignments, {"plan": "planner", "build": "builder"})
        self.assertIn(("planner", "builder"), decision.topology)

    def test_team_level_deadline_is_enforced_after_dag_scheduling(self) -> None:
        agents = [
            Agent("planner", {"plan": 0.98, "build": 0.05}, 0.02, 800),
            Agent("builder", {"plan": 0.05, "build": 0.99}, 0.02, 800),
        ]
        task = Task(
            "tight-dag",
            (
                Requirement("plan", 0.5, 0.70),
                Requirement("build", 0.5, 0.70, depends_on=("plan",)),
            ),
            budget=0.20,
            deadline_ms=1200,
            coordination_overhead=1.0,
        )
        decision = SAGERouter(agents, "planner").route(task)
        self.assertNotEqual(decision.mode, Mode.COLLABORATE)
        self.assertLessEqual(decision.latency_ms, task.deadline_ms)

    def test_contextual_reliability_does_not_bleed_across_skills(self) -> None:
        agents = [Agent("current", {"code": 0.9, "writing": 0.9}, 0.02, 300)]
        task = Task("code", (Requirement("code"),), budget=0.20, deadline_ms=2000)
        router = SAGERouter(agents, "current")
        decision = router.route(task)
        router.record_outcome(
            decision,
            ExecutionOutcome(False, requirement_scores={"code": 0.0}),
        )
        self.assertLess(router.skill_reliability[("current", "code")].mean, 0.5)
        self.assertEqual(router._skill_belief("current", "writing").mean, 0.5)

    def test_partial_credit_updates_agents_differently(self) -> None:
        agents = [
            Agent("current", {"plan": 0.98, "code": 0.05}, 0.02, 300),
            Agent("coder", {"plan": 0.05, "code": 0.99}, 0.02, 350),
        ]
        task = Task(
            "credit",
            (Requirement("plan", 0.5), Requirement("code", 0.5)),
            budget=0.20,
            deadline_ms=2000,
            coordination_overhead=0.01,
        )
        router = SAGERouter(agents, "current")
        decision = router.route(task)
        router.record_outcome(
            decision,
            ExecutionOutcome(0.5, agent_scores={"current": 1.0, "coder": 0.0}),
        )
        self.assertGreater(router.reliability["current"].mean, 0.5)
        self.assertLess(router.reliability["coder"].mean, 0.5)
        self.assertEqual(router.success_model.updates, 1)

    def test_outcome_rejects_unselected_agent_evidence(self) -> None:
        agents = [
            Agent("current", {"code": 0.9}, 0.02, 300),
            Agent("peer", {"code": 0.8}, 0.03, 350),
        ]
        task = Task("evidence", (Requirement("code"),), budget=0.20, deadline_ms=2000)
        router = SAGERouter(agents, "current")
        decision = router.route(task)
        with self.assertRaises(ValueError):
            router.record_outcome(
                decision,
                ExecutionOutcome(1.0, agent_scores={"not-selected": 1.0}),
            )

    def test_failed_incumbent_triggers_replan_to_peer(self) -> None:
        agents = [
            Agent("current", {"code": 0.80}, 0.02, 300),
            Agent("peer", {"code": 0.90}, 0.03, 350),
        ]
        task = Task("recover", (Requirement("code"),), budget=0.20, deadline_ms=2000)
        state = ExecutionState(
            active_agents=("current",),
            active_mode=Mode.SELF,
            progress=0.45,
            failed_agents=frozenset({"current"}),
            failure_count=1,
        )
        decision = SAGERouter(agents, "current").route(task, state=state)
        self.assertEqual(decision.mode, Mode.HANDOFF)
        self.assertEqual(decision.agents, ("peer",))
        self.assertTrue(decision.switch_recommended)


if __name__ == "__main__":
    unittest.main()
