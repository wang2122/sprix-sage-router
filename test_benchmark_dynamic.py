import json
import tempfile
import unittest
from pathlib import Path

from benchmark import make_agents
from benchmark_dynamic import PROGRESS_POINTS, summarize_dynamic_suite, write_progress_svg
from benchmark_dynamic_evaluator import (
    ExecutionCheckpoint,
    evaluate_recovery,
    latent_requirement_score,
)
from sprix_types import Mode, Requirement, Task


class DynamicBenchmarkTests(unittest.TestCase):
    def test_summary_is_reproducible_and_json_serializable(self) -> None:
        first = summarize_dynamic_suite((3,), 4, 2)
        second = summarize_dynamic_suite((3,), 4, 2)

        self.assertEqual(first, second)
        self.assertEqual(
            first["trajectory_replay"]["total_trajectories"],
            4,
        )
        points = first["progress_intervention"]["points"]
        self.assertEqual(tuple(point["progress"] for point in points), PROGRESS_POINTS)
        self.assertIn("static_coalition", first["trajectory_replay"]["strategies"])
        self.assertIn("dynamic_oracle", first["trajectory_replay"]["strategies"])
        json.dumps(first, sort_keys=True)
        with tempfile.TemporaryDirectory() as directory:
            figure = Path(directory) / "progress.svg"
            write_progress_svg(first, figure)
            svg = figure.read_text(encoding="utf-8")
            self.assertIn("Progress-aware SAGE", svg)
            self.assertIn('role="img"', svg)

    def test_switch_waste_comes_from_artifact_portability(self) -> None:
        agents = make_agents()
        agent_map = {agent.agent_id: agent for agent in agents}
        requirement = Requirement("code", minimum=0.65)
        task = Task("artifact", (requirement,), budget=0.30, deadline_ms=2500)
        checkpoint = ExecutionCheckpoint(
            active_agents=("generalist",),
            active_mode=Mode.SELF,
            completed_artifacts={},
            inflight_requirement="code",
            inflight_agent="generalist",
            inflight_fraction=0.80,
            inflight_quality=latent_requirement_score("generalist", requirement, 0.5),
            inflight_portability=0.25,
            elapsed_cost=0.02,
            elapsed_latency_ms=500,
        )
        retained = evaluate_recovery(
            task,
            0.5,
            Mode.SELF,
            ("generalist",),
            {"code": "generalist"},
            agent_map,
            checkpoint,
        )
        switched = evaluate_recovery(
            task,
            0.5,
            Mode.HANDOFF,
            ("coder",),
            {"code": "coder"},
            agent_map,
            checkpoint,
        )

        self.assertEqual(retained.wasted_work, 0.0)
        self.assertAlmostEqual(switched.wasted_work, 0.60)
        self.assertLess(retained.added_cost, switched.added_cost)


if __name__ == "__main__":
    unittest.main()
