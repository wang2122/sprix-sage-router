"""Online belief and outcome models used by the SAGE router."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Mapping


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass
class BetaBelief:
    alpha: float = 2.0
    beta: float = 2.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        total = self.alpha + self.beta
        return math.sqrt((self.alpha * self.beta) / (total * total * (total + 1)))

    def draw(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def update(self, score: float | bool, weight: float = 1.0) -> None:
        value = float(score)
        if not 0 <= value <= 1:
            raise ValueError("belief update score must be in [0, 1]")
        if weight <= 0:
            raise ValueError("belief update weight must be positive")
        self.alpha += weight * value
        self.beta += weight * (1.0 - value)


@dataclass
class OnlineSuccessModel:
    """Small online logistic model updated from execution outcomes.

    The default prior is intentionally configurable. Benchmarks can therefore
    separate informed-prior performance from learning and exploration effects.
    """

    learning_rate: float = 0.08
    l2: float = 0.001
    updates: int = 0
    bias: float = -1.15
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "coverage": 2.35,
            "bottleneck": 1.35,
            "trust": 0.80,
            "synergy": 0.35,
            "redundancy": -0.45,
            "coordination_loss": -0.55,
            "handoff_loss": -0.70,
            "switch_loss": -0.55,
            "load": -0.35,
        }
    )

    @classmethod
    def randomized(cls, seed: int, scale: float = 0.20) -> "OnlineSuccessModel":
        """Return a reproducible weak random prior for ablation studies."""

        rng = random.Random(seed)
        template = cls()
        return cls(
            bias=rng.uniform(-scale, scale),
            weights={name: rng.uniform(-scale, scale) for name in template.weights},
        )

    @classmethod
    def zeroed(cls) -> "OnlineSuccessModel":
        """Return a neutral prior with no hand-tuned feature coefficients."""

        template = cls()
        return cls(bias=0.0, weights={name: 0.0 for name in template.weights})

    def predict(self, features: Mapping[str, float]) -> float:
        logit = self.bias + sum(
            self.weights.get(name, 0.0) * value for name, value in features.items()
        )
        return _clip(_sigmoid(logit), 0.01, 0.99)

    def update(self, features: Mapping[str, float], outcome: float) -> None:
        prediction = self.predict(features)
        error = _clip(outcome) - prediction
        rate = self.learning_rate / math.sqrt(1.0 + self.updates / 50.0)
        self.bias += rate * error
        for name in self.weights:
            value = features.get(name, 0.0)
            self.weights[name] += rate * (error * value - self.l2 * self.weights[name])
        self.updates += 1
