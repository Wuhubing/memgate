from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from memgate.env.backends import SimBackend
from memgate.env.features import HISTORY_IDX, OBS_DIM, featurize
from memgate.env.gate_env import MemGateEnv, canonical_action, decode_action
from memgate.retrieval.retriever import Retriever
from memgate.world.generator import SessionFactory


def make_env(world, bm, reward, mode="persistent", mask=False, T=6):
    return MemGateEnv(world, SimBackend(world, bm, seed=1), mode=mode, reward=reward,
                      factory=SessionFactory(world, "train", 0), T_choices=[T], mask_history=mask, seed=0)


@pytest.mark.parametrize("mode", ["persistent", "ephemeral"])
def test_gym_api(world, bm, reward, mode):
    check_env(make_env(world, bm, reward, mode), skip_render_check=True)


@pytest.mark.parametrize("mode", ["persistent", "ephemeral"])
def test_episode(world, bm, reward, mode):
    env = make_env(world, bm, reward, mode, T=7)
    obs, info = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
    n, rng = 0, np.random.default_rng(0)
    while True:
        obs, r, term, trunc, info = env.step(int(rng.integers(8)))
        n += 1
        assert not trunc
        assert r == pytest.approx(info["outcome"].reward)
        if mode == "persistent":
            assert not set(c.record.id for c in env.cands if c) & set(env.st.in_context)
        if term:
            break
    assert n == 7


def test_mask_history(world, bm, reward):
    env = make_env(world, bm, reward, mask=True)
    env.reset(seed=1)
    for _ in range(5):
        obs, *_ = env.step(7)
        assert np.all(obs[list(HISTORY_IDX)] == 0)
    full = make_env(world, bm, reward, mask=False)
    full.reset(seed=1)
    seen = np.zeros(OBS_DIM)
    for _ in range(5):
        obs, *_ = full.step(7)
        seen = np.maximum(seen, np.abs(obs))
    assert seen[list(HISTORY_IDX)].sum() > 0     # the masked features carry information when unmasked


def test_no_label_leakage(world, bm, reward):
    """The observation must not change when generator-only labels change (E8)."""
    env = make_env(world, bm, reward)
    env.reset(seed=2)
    env.step(7)
    env.step(3)
    st = env.st.fork()
    obs = featurize(st, env.cands)
    st.store = [dataclasses.replace(r, superseded_by=None) for r in st.store]
    st.script = dataclasses.replace(st.script, turns=tuple(
        dataclasses.replace(t, answer_value=None, stale_values=(), relevant_ids=frozenset()) for t in st.script.turns))
    cands = [dataclasses.replace(c, record=dataclasses.replace(c.record, superseded_by=None)) if c else None
             for c in env.cands]
    assert np.array_equal(obs, featurize(st, cands))


def test_actions(world):
    ret = Retriever(world.emb)
    store = list(world.split_users("train")[0].initial_store)
    cands = ret.topk("q", store[:2])                   # third slot empty
    assert [r.id for r in decode_action(5, cands)] == [cands[0].record.id]
    assert canonical_action(5, cands) == 1 and canonical_action(7, cands) == 3


def test_same_seed_same_episode(world, bm, reward):
    def roll():
        env = make_env(world, bm, reward)
        env.reset(seed=11)
        return [env.step(a)[1] for a in (1, 7, 0, 2, 4, 6)]
    assert roll() == roll()
