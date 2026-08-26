import json
import tempfile
import unittest
from pathlib import Path

from benchmark import parse_seeds, summarize_suite


class BenchmarkTests(unittest.TestCase):
    def test_suite_summary_is_reproducible_and_json_serializable(self) -> None:
        first = summarize_suite((3,), tasks_per_seed=8)
        second = summarize_suite((3,), tasks_per_seed=8)

        self.assertEqual(first, second)
        self.assertEqual(first["total_tasks"], 8)
        self.assertEqual(first["online_model_updates"], [8])
        self.assertEqual(sum(first["learned_route_mix"].values()), 8)
        json.dumps(first, sort_keys=True)

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
