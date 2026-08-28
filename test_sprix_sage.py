import json
import math
import unittest

from sprix_sage import (
    Agent,
    Bid,
    ExecutionOutcome,
    ExecutionState,
    Mode,
    Requirement,
    RoutingTrace,
    SAGERouter,
    Task,
)


class SAGERouterTests(unittest.TestCase):
    def test_route_trace_explains_excluded_agents_and_serializes(self) -> None:
        agents = [
            Agent("current", {"finance": 0.82}, 0.02, 300, frozenset({"ledger:write"})),
            Agent("untrusted", {"finance": 1.0}, 0.00, 100),
            Agent("offline", {"finance": 1.0}, 0.00, 100, availability=0.0),
        ]
        task = Task(
            "settle-with-trace",
            (Requirement("finance"),),
            required_permissions=frozenset({"ledger:write"}),
            budget=0.20,
            deadline_ms=2000,
        )

        trace = SAGERouter(agents, "current").route_with_trace(task)

        self.assertIsInstance(trace, RoutingTrace)
        self.assertEqual(trace.selected.mode, Mode.SELF)
        self.assertEqual(trace.eligible_agents, ("current",))
        self.assertIn("missing_permissions:ledger:write", trace.excluded_agents["untrusted"])
        self.assertIn("unavailable", trace.excluded_agents["offline"])
        self.assertEqual(trace.alternatives[0], trace.selected)
        encoded = json.dumps(trace.to_dict(), sort_keys=True)
        self.assertIn('"selected"', encoded)
        self.assertNotIn("model_features", encoded)

    def test_duplicate_agent_ids_are_rejected(self) -> None:
        agents = [
            Agent("duplicate", {"planning": 0.95}, 0.02, 300),
            Agent("duplicate", {"coding": 0.95}, 0.03, 350),
        ]
        with self.assertRaisesRegex(ValueError, "agent IDs must be unique: duplicate"):
            SAGERouter(agents, "duplicate")

    def test_assignment_beam_width_must_be_positive(self) -> None:
        agents = [Agent("current", {"planning": 0.95}, 0.02, 300)]

        with self.assertRaisesRegex(ValueError, "beam widths positive"):
            SAGERouter(agents, "current", assignment_beam_width=0)

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

    def test_bid_task_id_must_match_routed_task(self) -> None:
        agents = [Agent("current", {"code": 0.9}, 0.02, 300)]
        task = Task("expected", (Requirement("code"),), budget=0.20, deadline_ms=2000)
        bid = Bid("current", "other", 0.01, 250)

        with self.assertRaisesRegex(ValueError, "targets task 'other', expected 'expected'"):
            SAGERouter(agents, "current").route(task, bids=[bid])

    def test_bid_must_reference_a_registered_agent(self) -> None:
        agents = [Agent("current", {"code": 0.9}, 0.02, 300)]
        task = Task("code", (Requirement("code"),), budget=0.20, deadline_ms=2000)
        bid = Bid("unknown", "code", 0.01, 250)

        with self.assertRaisesRegex(ValueError, "unknown agent: 'unknown'"):
            SAGERouter(agents, "current").route(task, bids=[bid])

    def test_duplicate_bids_for_an_agent_are_rejected(self) -> None:
        agents = [Agent("current", {"code": 0.9}, 0.02, 300)]
        task = Task("code", (Requirement("code"),), budget=0.20, deadline_ms=2000)
        bids = [
            Bid("current", "code", 0.01, 250),
            Bid("current", "code", 0.02, 300),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate bid for agent: 'current'"):
            SAGERouter(agents, "current").route(task, bids=bids)

    def test_bid_rejects_nan_cost_and_latency(self) -> None:
        invalid_quotes = ((math.nan, 250), (0.01, math.nan))

        for quoted_cost, promised_latency_ms in invalid_quotes:
            with self.subTest(
                quoted_cost=quoted_cost,
                promised_latency_ms=promised_latency_ms,
            ):
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    Bid("peer", "code", quoted_cost, promised_latency_ms)

    def test_agent_rejects_non_finite_cost_and_latency(self) -> None:
        invalid = (
            (math.nan, 300),
            (math.inf, 300),
            (-math.inf, 300),
            (0.02, math.nan),
            (0.02, math.inf),
            (0.02, -math.inf),
        )

        for cost, latency_ms in invalid:
            with self.subTest(cost=cost, latency_ms=latency_ms):
                with self.assertRaisesRegex(ValueError, "finite"):
                    Agent("peer", {"code": 0.9}, cost, latency_ms)

    def test_requirement_rejects_non_finite_weight(self) -> None:
        for weight in (math.nan, math.inf, -math.inf):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, "finite"):
                    Requirement("code", weight=weight)

    def test_task_rejects_invalid_value_budget_and_deadline(self) -> None:
        for field_name, value, message in (
            ("value", math.nan, "finite"),
            ("value", math.inf, "finite"),
            ("value", -math.inf, "finite"),
            ("budget", math.nan, "infinity"),
            ("budget", -math.inf, "infinity"),
            ("deadline_ms", math.nan, "infinity"),
            ("deadline_ms", -math.inf, "infinity"),
        ):
            with self.subTest(field=field_name, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    Task("t", (Requirement("code"),), **{field_name: value})

    def test_task_keeps_infinite_budget_and_deadline_defaults(self) -> None:
        task = Task("t", (Requirement("code"),))

        self.assertTrue(math.isinf(task.budget))
        self.assertTrue(math.isinf(task.deadline_ms))

    def test_checkpoint_state_rejects_progress_without_inflight_requirement(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires inflight_requirement"):
            ExecutionState(inflight_progress=0.4)

    def test_checkpoint_reuse_can_change_a_handoff_into_continuation(self) -> None:
        agents = [
            Agent("current", {"code": 0.70}, 0.02, 300),
            Agent("specialist", {"code": 0.93}, 0.04, 420),
        ]
        task = Task(
            "checkpoint",
            (Requirement("code", minimum=0.65),),
            budget=0.20,
            deadline_ms=2000,
            handoff_friction=0.70,
            context_transferability=0.10,
        )
        router = SAGERouter(agents, "current")
        early = router.route(
            task,
            state=ExecutionState(
                active_agents=("current",),
                active_assignments={"code": "current"},
                inflight_requirement="code",
                inflight_progress=0.0,
                artifact_transferability={"code": 0.10},
            ),
        )
        late = router.route(
            task,
            state=ExecutionState(
                active_agents=("current",),
                active_assignments={"code": "current"},
                inflight_requirement="code",
                inflight_progress=0.90,
                inflight_quality=0.85,
                artifact_transferability={"code": 0.10},
            ),
        )

        self.assertEqual(early.mode, Mode.HANDOFF)
        self.assertEqual(late.mode, Mode.SELF)
        self.assertLess(late.cost, early.cost)

    def test_pair_synergy_requires_explicit_pair_evidence(self) -> None:
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
            self.assertEqual(router.synergy[pair].mean, 0.5)
            router.record_outcome(
                decision,
                ExecutionOutcome(True, pair_scores={pair: 0.8}),
            )
            self.assertGreater(router.synergy[pair].mean, 0.5)

            with self.assertRaisesRegex(ValueError, "duplicate pair evidence"):
                router.record_outcome(
                    decision,
                    ExecutionOutcome(
                        True,
                        pair_scores={pair: 0.8, (pair[1], pair[0]): 0.7},
                    ),
                )

    def test_workload_sensitive_cost_does_not_charge_full_price_per_agent(self) -> None:
        agents = [
            Agent("current", {"large": 0.99, "small": 0.05}, 1.0, 300),
            Agent("peer", {"large": 0.05, "small": 0.99}, 1.0, 300),
        ]
        task = Task(
            "workload-cost",
            (Requirement("large", 0.9), Requirement("small", 0.1)),
            value=10.0,
            budget=3.0,
            deadline_ms=2000,
            coordination_overhead=0.0,
        )

        decision = SAGERouter(
            agents,
            "current",
            max_collaborators=1,
            cost_activation_fraction=0.10,
        ).route(task)

        self.assertEqual(decision.mode, Mode.COLLABORATE)
        self.assertLess(decision.cost, 2.0)
        self.assertGreater(decision.cost, 1.0)

    def test_infeasible_route_degrades_with_explicit_violation(self) -> None:
        agents = [Agent("current", {"code": 0.9}, 0.5, 1500)]
        task = Task(
            "degraded",
            (Requirement("code"),),
            budget=0.1,
            deadline_ms=500,
        )

        decision = SAGERouter(agents, "current").route(task)

        self.assertFalse(decision.feasible)
        self.assertEqual(len(decision.constraint_violations), 2)
        self.assertTrue(any(item.startswith("budget:") for item in decision.constraint_violations))
        self.assertTrue(any(item.startswith("deadline_ms:") for item in decision.constraint_violations))

        with self.assertRaisesRegex(RuntimeError, "no feasible route"):
            SAGERouter(agents, "current", allow_degraded=False).route(task)

    def test_degraded_route_never_bypasses_permissions(self) -> None:
        agents = [Agent("current", {"finance": 0.9}, 0.01, 100)]
        task = Task(
            "permission",
            (Requirement("finance"),),
            required_permissions=frozenset({"ledger:write"}),
        )

        with self.assertRaisesRegex(RuntimeError, "authorized and available"):
            SAGERouter(agents, "current").route(task)

    def test_candidate_prefilter_is_bounded_and_keeps_incumbent(self) -> None:
        agents = [
            Agent(f"agent-{index}", {"code": index / 20.0}, 0.01, 100)
            for index in range(20)
        ]
        task = Task("prefilter", (Requirement("code"),), budget=1.0, deadline_ms=1000)

        trace = SAGERouter(
            agents,
            "agent-0",
            candidate_limit=6,
            max_collaborators=1,
        ).route_with_trace(task)

        self.assertEqual(len(trace.eligible_agents), 6)
        self.assertIn("agent-0", trace.eligible_agents)
        self.assertEqual(len(trace.prefiltered_agents), 14)

    def test_candidate_prefilter_prioritizes_individually_feasible_peer(self) -> None:
        agents = [
            Agent("current", {"code": 0.9}, 1.0, 100),
            Agent("expensive", {"code": 1.0}, 1.0, 100),
            Agent("affordable", {"code": 0.5}, 0.01, 100),
        ]
        task = Task("prefilter-feasible", (Requirement("code"),), budget=0.1, deadline_ms=1000)

        trace = SAGERouter(
            agents,
            "current",
            candidate_limit=2,
            max_collaborators=0,
        ).route_with_trace(task)

        self.assertIn("affordable", trace.eligible_agents)
        self.assertTrue(trace.selected.feasible)

    def test_learned_state_round_trips_through_json(self) -> None:
        agents = [
            Agent("current", {"planning": 0.9, "coding": 0.2}, 0.02, 300),
            Agent("coder", {"planning": 0.2, "coding": 0.9}, 0.02, 350),
        ]
        task = Task(
            "persist",
            (Requirement("planning", 0.5), Requirement("coding", 0.5)),
            budget=0.20,
            deadline_ms=2000,
            coordination_overhead=0.01,
        )
        router = SAGERouter(agents, "current")
        decision = router.route(task)
        router.record_outcome(
            decision,
            ExecutionOutcome(
                0.8,
                requirement_scores={"planning": 0.9, "coding": 0.7},
                actual_cost=0.05,
                actual_latency_ms=700,
            ),
        )
        serialized = json.dumps(router.export_state(), sort_keys=True)

        restored = SAGERouter(agents, "current")
        restored.restore_state(json.loads(serialized))

        self.assertEqual(restored.export_state(), router.export_state())
        self.assertEqual(restored.route(task).mode, router.route(task).mode)

    def test_state_restore_rejects_a_different_agent_roster(self) -> None:
        source = SAGERouter([Agent("current", {"code": 0.9}, 0.02, 300)], "current")
        target = SAGERouter([Agent("other", {"code": 0.9}, 0.02, 300)], "other")

        with self.assertRaisesRegex(ValueError, "agent_ids"):
            target.restore_state(source.export_state())

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

    def test_collaboration_topology_connects_incumbent_to_peer_component(self) -> None:
        agents = [
            Agent("owner", {"oversight": 1.0}, 0.0, 100),
            Agent("planner", {"plan": 1.0}, 0.0, 100),
            Agent("builder", {"build": 1.0}, 0.0, 100),
        ]
        task = Task(
            "connected-topology",
            (
                Requirement("oversight", 1.0, 0.95),
                Requirement("plan", 1.0, 0.95),
                Requirement("build", 1.0, 0.95, depends_on=("plan",)),
            ),
            value=10.0,
            budget=1.0,
            deadline_ms=5000,
            coordination_overhead=0.01,
        )

        decision = SAGERouter(agents, "owner", max_collaborators=2).route(task)

        self.assertEqual(decision.mode, Mode.COLLABORATE)
        self.assertEqual(set(decision.agents), {"owner", "planner", "builder"})
        self.assertIn(("planner", "builder"), decision.topology)
        self.assertIn(("owner", "planner"), decision.topology)

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

    def test_assignment_search_parallelizes_independent_requirements(self) -> None:
        agents = [
            Agent("current", {"research": 0.95, "coding": 0.95}, 0.0, 600),
            Agent("peer", {"research": 0.90, "coding": 0.90}, 0.0, 600),
        ]
        task = Task(
            "parallel-deadline",
            (
                Requirement("research", 0.5, 0.55),
                Requirement("coding", 0.5, 0.55),
            ),
            value=5.0,
            budget=1.0,
            deadline_ms=550,
            coordination_overhead=0.10,
        )

        decision = SAGERouter(
            agents,
            "current",
            max_collaborators=1,
            assignment_beam_width=4,
        ).route(task)

        self.assertEqual(decision.mode, Mode.COLLABORATE)
        self.assertEqual(set(decision.assignments.values()), {"current", "peer"})
        self.assertLessEqual(decision.latency_ms, task.deadline_ms)
        self.assertEqual(decision.diagnostics["assignment_search_candidates"], 4.0)

    def test_single_agent_assignment_remains_deterministic(self) -> None:
        agents = [Agent("current", {"plan": 0.9, "build": 0.8}, 0.02, 300)]
        task = Task(
            "single-agent",
            (
                Requirement("plan", 0.4),
                Requirement("build", 0.6, depends_on=("plan",)),
            ),
            budget=0.20,
            deadline_ms=2000,
        )

        decision = SAGERouter(
            agents, "current", assignment_beam_width=1
        ).route(task)

        self.assertEqual(
            decision.assignments, {"plan": "current", "build": "current"}
        )
        self.assertEqual(decision.diagnostics["assignment_search_candidates"], 1.0)

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
