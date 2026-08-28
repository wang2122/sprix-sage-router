import json
import unittest

from benchmark_trust import summarize_trust_suite


class TrustBenchmarkTests(unittest.TestCase):
    def test_contextual_trust_converges_without_exploration_confound(self) -> None:
        summary = summarize_trust_suite((3, 7), observations=250)
        heterogeneous = summary["heterogeneous"]["models"]
        contextual = heterogeneous["per_requirement"]["last"]
        global_only = heterogeneous["single_reputation"]["last"]

        self.assertLess(
            contextual["brier"]["mean"],
            global_only["brier"]["mean"],
        )
        self.assertLess(
            contextual["selection_regret"]["mean"],
            global_only["selection_regret"]["mean"],
        )
        self.assertEqual(summary["collection_policy"], "shared exogenous round-robin; no exploration bonus")
        json.dumps(summary, sort_keys=True)

    def test_homogeneous_negative_control_has_no_routing_advantage(self) -> None:
        summary = summarize_trust_suite((3,), observations=150)
        models = summary["homogeneous_negative_control"]["models"]

        self.assertEqual(
            models["per_requirement"]["last"]["selection_regret"]["mean"],
            0.0,
        )
        self.assertEqual(
            models["single_reputation"]["last"]["selection_regret"]["mean"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
