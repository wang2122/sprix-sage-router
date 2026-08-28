import json
import tempfile
import unittest
from pathlib import Path

from benchmark import parse_seeds, summarize_suite
from benchmark_evaluator import HIDDEN_AGENTS, PAIR_COMPATIBILITY


class BenchmarkTests(unittest.TestCase):
    def test_suite_summary_is_reproducible_and_json_serializable(self) -> None:
        first = summarize_suite((3,), tasks_per_seed=8)
        second = summarize_suite((3,), tasks_per_seed=8)

        self.assertEqual(first, second)
        self.assertEqual(first["total_tasks"], 8)
        self.assertEqual(first["model_updates"]["learned_no_explore"], [8])
        self.assertEqual(
            sum(first["route_mix_by_strategy"]["learned_no_explore"].values()),
            8,
        )
        self.assertIn("random_team", first["strategies"])
        self.assertIn("greedy_team", first["strategies"])
        self.assertIn("learned_random_init", first["strategies"])
        self.assertIn("learning_curve", first)
        json.dumps(first, sort_keys=True)

    def test_evaluator_is_structurally_held_out(self) -> None:
        self.assertLess(HIDDEN_AGENTS["generalist"].skills["code"], 0.60)
        self.assertGreater(HIDDEN_AGENTS["researcher"].skills["writing"], 0.90)
        self.assertTrue(all(value > 0 for value in PAIR_COMPATIBILITY.values()))
        self.assertTrue(any(value < 1 for value in PAIR_COMPATIBILITY.values()))
        self.assertTrue(any(value > 1 for value in PAIR_COMPATIBILITY.values()))

    def test_seed_parser_accepts_a_comma_separated_list(self) -> None:
        self.assertEqual(parse_seeds("3, 7,11"), (3, 7, 11))

    def test_summary_can_be_written_without_custom_encoders(self) -> None:
        summary = summarize_suite((7,), tasks_per_seed=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), summary)


if __name__ == "__main__":
    unittest.main()
