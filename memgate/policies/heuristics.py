"""Policies that need no training. All share act(obs, info) -> action (plan §3.12)."""
from __future__ import annotations

import numpy as np


class Policy:
    name = "policy"

    def act(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError


class AlwaysPolicy(Policy):
    name = "always"

    def act(self, obs, info):
        return 7


class NeverPolicy(Policy):
    name = "never"

    def act(self, obs, info):
        return 0


class RandomPolicy(Policy):
    name = "uniform"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info):
        return int(self.rng.integers(8))


class ThresholdPolicy(Policy):
    """Inject every candidate whose fused retrieval score is >= tau (what most harnesses do)."""

    name = "threshold"

    def __init__(self, tau: float):
        self.tau = tau

    def act(self, obs, info):
        return sum(1 << i for i, c in enumerate(info["cands"]) if c is not None and c.fused >= self.tau)


class MixedBehaviorPolicy(Policy):
    """Log-collection behaviour policy: per session, uniform random (p) or threshold (1-p) (plan §3.10)."""

    def __init__(self, tau: float, p_uniform: float, seed: int):
        rng = np.random.default_rng(seed)
        self.inner = RandomPolicy(seed) if rng.random() < p_uniform else ThresholdPolicy(tau)
        self.name = self.inner.name

    def act(self, obs, info):
        return self.inner.act(obs, info)


class ForcedPolicy(Policy):
    """E0: inject a fixed record at fixed turns, nothing otherwise. Used through a chooser, not act()."""

    name = "forced"
