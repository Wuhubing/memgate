"""Relations the exact solution must satisfy on the small instance (原文档 "必须成立的关系")."""
from __future__ import annotations

import numpy as np
import pytest

from memgate.config import load_reward
from memgate.planning.abstract import KINDS, AbstractEnv, candidates, distinct_actions, initial_states
from memgate.planning.dp import SmallInstance

P_KIND = {"query": 0.4, "chat": 0.45, "update": 0.15}
REWARDS = ("R_full", "R_notok", "R_behavior")


@pytest.fixture(scope="module", params=["persistent", "ephemeral"])
def inst(request, bm):
    return SmallInstance(bm, 3, request.param, P_KIND)


def policies(inst, cfg):
    return {"dp": inst.solve(cfg), "greedy": inst.greedy(cfg), "amortized_greedy": inst.greedy(cfg, amortize=True),
            "always": inst.always(), "never": inst.never()}


@pytest.mark.parametrize("rname", REWARDS)
def test_dp_dominates(inst, rname):
    cfg = load_reward(rname)
    for p_stale in (0.0, 0.5, 1.0):
        v = {k: inst.evaluate(pi, cfg, p_stale).sum() for k, pi in policies(inst, cfg).items()}
        assert all(v["dp"] >= x - 1e-9 for x in v.values()), v


def test_gamma_zero_is_greedy(inst):
    cfg = load_reward("R_full")
    a = inst.evaluate(inst.solve(cfg, gamma=0.0), cfg, 0.5).sum()
    b = inst.evaluate(inst.greedy(cfg), cfg, 0.5).sum()
    assert a == pytest.approx(b, abs=1e-9)


def test_amortized_equals_greedy_in_ephemeral(bm):
    inst = SmallInstance(bm, 4, "ephemeral", P_KIND)
    cfg = load_reward("R_full")
    a = inst.evaluate(inst.greedy(cfg, amortize=True), cfg, 0.5)
    b = inst.evaluate(inst.greedy(cfg), cfg, 0.5)
    assert np.allclose(a, b)


def test_outcome_tables_are_distributions(inst):
    for _, s in initial_states(0.5):
        for kind in KINDS:
            for a in distinct_actions(candidates(s, kind, inst.mode)):
                t = inst.table(s, kind, a)
                assert t.p.sum() == pytest.approx(1.0) and (t.p >= 0).all()
                assert all(n.t == s.t + 1 for n in t.nexts)


def test_exact_value_matches_monte_carlo(inst):
    cfg = load_reward("R_full")
    pi = inst.solve(cfg)
    exact = inst.evaluate(pi, cfg, 0.5).sum()
    env = AbstractEnv(inst, cfg, 0.5, seed=0)
    rets = []
    for ep in range(1500):
        _, info = env.reset(seed=ep)
        tot, done = 0.0, False
        while not done:
            _, r, done, _, info = env.step(pi(env.s, env.kind))
            tot += r
        rets.append(tot)
    se = np.std(rets) / np.sqrt(len(rets))
    assert abs(np.mean(rets) - exact) < 4 * se + 1e-6


def test_components_sum_to_total(inst):
    cfg = load_reward("R_full")
    v = inst.evaluate(inst.always(), cfg, 0.5)
    assert v.shape == (4,) and v[3] <= 0 and v[0] >= 0


@pytest.mark.slow
def test_dqn_on_abstract_env_tracks_dp_and_greedy(bm):
    """gamma=0 DQN ~ greedy value, gamma=1 DQN close to DP (plan §5)."""
    from stable_baselines3 import DQN
    inst = SmallInstance(bm, 6, "persistent", P_KIND)
    cfg = load_reward("R_behavior")
    v_dp = inst.evaluate(inst.solve(cfg), cfg, 0.5).sum()
    v_gr = inst.evaluate(inst.greedy(cfg), cfg, 0.5).sum()

    def value_of(model):
        env = AbstractEnv(inst, cfg, 0.5, seed=1)
        rets = []
        for ep in range(2000):
            obs, _ = env.reset(seed=10_000 + ep)
            tot, done = 0.0, False
            while not done:
                a = int(model.predict(obs, deterministic=True)[0])
                obs, r, done, _, _ = env.step(a)
                tot += r
            rets.append(tot)
        return float(np.mean(rets))

    vals = {}
    for g in (0.0, 1.0):
        m = DQN("MlpPolicy", AbstractEnv(inst, cfg, 0.5, seed=0), gamma=g, learning_starts=2000,
                exploration_fraction=0.3, seed=0, verbose=0)
        m.learn(60_000)
        vals[g] = value_of(m)
    tol = 0.1 * abs(v_dp) + 0.05
    assert abs(vals[0.0] - v_gr) < tol
    assert vals[1.0] > v_dp - tol
