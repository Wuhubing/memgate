"""LinUCB over the 8 joint actions (Li et al. 2010), with an optional risk-sensitive reward (plan §3.12)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from memgate.policies.heuristics import Policy


class LinUCB(Policy):
    name = "linucb"

    def __init__(self, dim: int, n_actions: int = 8, alpha: float = 0.5, reg: float = 1.0):
        self.alpha = alpha
        self.n_actions = n_actions
        self.A_inv = np.stack([np.eye(dim + 1) / reg for _ in range(n_actions)])
        self.b = np.zeros((n_actions, dim + 1))
        self.frozen = False

    @staticmethod
    def _x(obs: np.ndarray) -> np.ndarray:
        return np.r_[1.0, obs.astype(np.float64)]

    def scores(self, obs: np.ndarray) -> np.ndarray:
        x = self._x(obs)
        theta = np.einsum("aij,aj->ai", self.A_inv, self.b)
        mean = theta @ x
        if self.frozen:
            return mean
        width = np.sqrt(np.einsum("i,aij,j->a", x, self.A_inv, x))
        return mean + self.alpha * width

    def act(self, obs, info=None) -> int:
        return int(np.argmax(self.scores(obs)))

    def update(self, obs: np.ndarray, a: int, r: float) -> None:
        x = self._x(obs)
        Ai = self.A_inv[a]
        Ax = Ai @ x
        self.A_inv[a] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)   # Sherman-Morrison
        self.b[a] += r * x

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> LinUCB:
        with open(path, "rb") as f:
            p = pickle.load(f)
        p.frozen = True
        return p


def risk_sensitive_reward(out, factor: float = 2.0) -> float:
    """Wrong injections cost double (stands in for 'Learning When to Remember')."""
    return out.r_quality + factor * (out.r_stale + out.r_intrusion) + out.r_tokens


def train_linucb(env, steps: int, alpha: float, risk_sensitive: bool = False, seed: int = 0) -> LinUCB:
    pol = LinUCB(env.observation_space.shape[0], alpha=alpha)
    obs, info = env.reset(seed=seed)
    for _ in range(steps):
        a = pol.act(obs, info)
        nobs, r, term, trunc, info = env.step(a)
        out = info["outcome"]
        pol.update(obs, a, risk_sensitive_reward(out) if risk_sensitive else r)
        obs = nobs
        if term or trunc:
            obs, info = env.reset()
    pol.frozen = True
    return pol
