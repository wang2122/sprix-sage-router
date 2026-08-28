"""Convergence study for per-requirement trust versus one reputation score.

Both models receive the same exogenous, round-robin observation stream.  This
removes exploration spend and action-selection feedback as confounders.  The
heterogeneous scenario contains specialists and ranking reversals; the
homogeneous negative control contains no useful skill specialization.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, cast

from benchmark import DEFAULT_SEEDS, SKILLS, parse_seeds
from sprix_learning import BetaBelief

AGENT_IDS = ("generalist", "coder", "researcher", "vision", "reviewer")
DECLARED_SKILLS: dict[str, dict[str, float]] = {
    "generalist": dict.fromkeys(SKILLS, 0.80),
    "coder": {"code": 0.86, "research": 0.76, "vision": 0.75, "security": 0.76, "writing": 0.75},
    "researcher": {"code": 0.75, "research": 0.86, "vision": 0.76, "security": 0.74, "writing": 0.84},
    "vision": {"code": 0.75, "research": 0.75, "vision": 0.86, "security": 0.76, "writing": 0.74},
    "reviewer": {"code": 0.77, "research": 0.78, "vision": 0.75, "security": 0.86, "writing": 0.86},
}
HETEROGENEOUS_SUCCESS: dict[str, dict[str, float]] = {
    "generalist": {"code": 0.58, "research": 0.82, "vision": 0.43, "security": 0.74, "writing": 0.69},
    "coder": {"code": 0.88, "research": 0.55, "vision": 0.22, "security": 0.46, "writing": 0.61},
    "researcher": {"code": 0.50, "research": 0.84, "vision": 0.65, "security": 0.31, "writing": 0.91},
    "vision": {"code": 0.29, "research": 0.46, "vision": 0.89, "security": 0.56, "writing": 0.35},
    "reviewer": {"code": 0.71, "research": 0.62, "vision": 0.54, "security": 0.82, "writing": 0.86},
}
HOMOGENEOUS_SUCCESS: dict[str, dict[str, float]] = {
    agent_id: dict.fromkeys(SKILLS, 0.68) for agent_id in AGENT_IDS
}
TRUST_MODELS = ("per_requirement", "single_reputation")
TRUST_METRICS = ("brier", "selection_regret", "oracle_match")


@dataclass
class TrustModel:
    contextual: bool

    def __post_init__(self) -> None:
        self.global_beliefs = {agent_id: BetaBelief() for agent_id in AGENT_IDS}
        self.skill_beliefs = {
            (agent_id, skill): BetaBelief()
            for agent_id in AGENT_IDS
            for skill in SKILLS
        }

    def predict(self, agent_id: str, skill: str) -> float:
        global_value = self.global_beliefs[agent_id].mean
        if not self.contextual:
            return global_value
        return 0.35 * global_value + 0.65 * self.skill_beliefs[(agent_id, skill)].mean

    def selection_score(self, agent_id: str, skill: str) -> float:
        trust = self.predict(agent_id, skill)
        return DECLARED_SKILLS[agent_id][skill] * (0.65 + 0.35 * trust)

    def update(self, agent_id: str, skill: str, outcome: float) -> None:
        self.global_beliefs[agent_id].update(outcome, 0.35 if self.contextual else 1.0)
        if self.contextual:
            self.skill_beliefs[(agent_id, skill)].update(outcome)


def _evaluate_model(
    model: TrustModel,
    latent: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    squared_errors = [
        (model.predict(agent_id, skill) - latent[agent_id][skill]) ** 2
        for agent_id in AGENT_IDS
        for skill in SKILLS
    ]
    regret = 0.0
    oracle_match = 0.0
    for skill in SKILLS:
        selected = max(
            AGENT_IDS,
            key=lambda agent_id: (model.selection_score(agent_id, skill), agent_id),
        )
        best_value = max(latent[agent_id][skill] for agent_id in AGENT_IDS)
        selected_value = latent[selected][skill]
        regret += best_value - selected_value
        oracle_match += float(abs(selected_value - best_value) < 1e-12)
    return {
        "brier": sum(squared_errors) / len(squared_errors),
        "selection_regret": regret / len(SKILLS),
        "oracle_match": oracle_match / len(SKILLS),
    }


def simulate_trust(
    seed: int,
    observations: int,
    latent: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    if observations <= 0:
        raise ValueError("observations must be positive")
    rng = random.Random(seed)
    models = {
        "per_requirement": TrustModel(contextual=True),
        "single_reputation": TrustModel(contextual=False),
    }
    histories: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in TRUST_METRICS}
        for name in TRUST_MODELS
    }
    identification: dict[str, int | None] = {name: None for name in TRUST_MODELS}
    consecutive = {name: 0 for name in TRUST_MODELS}
    curve: list[dict[str, object]] = []
    agent_offset = seed % len(AGENT_IDS)
    skill_offset = (seed // len(AGENT_IDS)) % len(SKILLS)

    for index in range(observations):
        agent_id = AGENT_IDS[(index + agent_offset) % len(AGENT_IDS)]
        skill = SKILLS[((index // len(AGENT_IDS)) + skill_offset) % len(SKILLS)]
        outcome = float(rng.random() < latent[agent_id][skill])
        for model in models.values():
            model.update(agent_id, skill, outcome)
        for name, model in models.items():
            metrics = _evaluate_model(model, latent)
            for metric, value in metrics.items():
                histories[name][metric].append(value)
            if metrics["oracle_match"] >= 0.80:
                consecutive[name] += 1
            else:
                consecutive[name] = 0
            if identification[name] is None and consecutive[name] >= 20:
                identification[name] = index + 1
        if (index + 1) % 25 == 0 or index + 1 == observations:
            curve.append(
                {
                    "observations": index + 1,
                    "models": {
                        name: {
                            metric: histories[name][metric][-1]
                            for metric in TRUST_METRICS
                        }
                        for name in TRUST_MODELS
                    },
                }
            )

    window = min(100, max(1, observations // 2))
    return {
        "observations": observations,
        "window": window,
        "models": {
            name: {
                "first": {
                    metric: mean(histories[name][metric][:window])
                    for metric in TRUST_METRICS
                },
                "last": {
                    metric: mean(histories[name][metric][-window:])
                    for metric in TRUST_METRICS
                },
                "identification_observation": identification[name],
            }
            for name in TRUST_MODELS
        },
        "curve": curve,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "population_stddev": pstdev(values)}


def _summarize_scenario(
    seeds: tuple[int, ...],
    observations: int,
    latent: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    runs: list[Any] = [simulate_trust(seed, observations, latent) for seed in seeds]
    models: dict[str, object] = {}
    for name in TRUST_MODELS:
        periods = {
            period: {
                metric: _summary(
                    [
                        float(run["models"][name][period][metric])
                        for run in runs
                    ]
                )
                for metric in TRUST_METRICS
            }
            for period in ("first", "last")
        }
        identified = [
            int(value)
            for run in runs
            if (value := run["models"][name]["identification_observation"])
            is not None
        ]
        models[name] = {
            **periods,
            "identification": {
                "fraction": len(identified) / len(runs),
                "mean_observation": mean(identified) if identified else None,
            },
        }

    checkpoints = list(range(25, observations + 1, 25))
    if not checkpoints or checkpoints[-1] != observations:
        checkpoints.append(observations)
    curve: list[dict[str, object]] = []
    for position, checkpoint in enumerate(checkpoints):
        curve.append(
            {
                "observations": checkpoint,
                "models": {
                    name: {
                        metric: _summary(
                            [
                                float(run["curve"][position]["models"][name][metric])
                                for run in runs
                            ]
                        )
                        for metric in TRUST_METRICS
                    }
                    for name in TRUST_MODELS
                },
            }
        )
    return {
        "models": models,
        "curve": curve,
        "window": min(100, max(1, observations // 2)),
    }


def summarize_trust_suite(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    observations: int = 500,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one trust seed is required")
    if observations <= 0:
        raise ValueError("observations must be positive")
    return {
        "schema_version": 1,
        "seeds": list(seeds),
        "observations_per_seed": observations,
        "collection_policy": "shared exogenous round-robin; no exploration bonus",
        "heterogeneous": _summarize_scenario(
            seeds,
            observations,
            HETEROGENEOUS_SUCCESS,
        ),
        "homogeneous_negative_control": _summarize_scenario(
            seeds,
            observations,
            HOMOGENEOUS_SUCCESS,
        ),
    }


def run_suite(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    observations: int = 500,
) -> dict[str, object]:
    summary = summarize_trust_suite(seeds, observations)
    print(f"trust observations: {len(seeds) * observations} ({len(seeds)} seeds x {observations})")
    for scenario_name in ("heterogeneous", "homogeneous_negative_control"):
        scenario = cast(Any, summary[scenario_name])
        print(scenario_name)
        for name in TRUST_MODELS:
            values = scenario["models"][name]
            print(
                f"  {name:19s} last brier={values['last']['brier']['mean']:.4f}; "
                f"regret={values['last']['selection_regret']['mean']:.4f}; "
                f"oracle-match={values['last']['oracle_match']['mean']:.3f}; "
                f"identified={values['identification']['fraction']:.0%}"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="comma-separated deterministic seeds (default: 3,7,11,19,23)",
    )
    parser.add_argument(
        "--observations",
        type=int,
        default=500,
        help="shared observations per seed and scenario (default: 500)",
    )
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()
    if args.observations <= 0:
        parser.error("--observations must be positive")
    summary = run_suite(args.seeds, args.observations)
    if args.json_path is not None:
        args.json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"json summary: {args.json_path}")


if __name__ == "__main__":
    main()
