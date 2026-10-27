"""Shared fixtures: a small offline world (hash embeddings) and a behaviour model fitted on FakeLLM logs."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from memgate.config import load_reward, load_world
from memgate.harness.llm_client import FakeLLM
from memgate.harness.runner import policy_chooser, read_logs, run_sessions
from memgate.policies.heuristics import MixedBehaviorPolicy
from memgate.retrieval.embed import HashEmbedder
from memgate.schema import Turn
from memgate.sim.behavior_model import BehaviorModel
from memgate.sim.dataset import build_datasets
from memgate.world.generator import current_record, generate_world, log_scripts, stale_records


def small_cfg(**over):
    base = {"n_users": 10, "split": {"train": 6, "val": 2, "test": 2}, "embedder": "hash", "seed": 3,
            "scripts_per_val_user": 5, "scripts_per_test_user": 5, "log_scripts": 120}
    return load_world(**{**base, **over})


@pytest.fixture(scope="session")
def world():
    return generate_world(small_cfg(), HashEmbedder())


@pytest.fixture(scope="session")
def reward():
    return load_reward("R_full")


@pytest.fixture(scope="session")
def fake_logs(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("logs")
    scripts = log_scripts(world, 120, [6, 8], seed=5)
    pols = {s.id: MixedBehaviorPolicy(0.03, 0.7, i) for i, s in enumerate(scripts)}
    asyncio.run(run_sessions(scripts, lambda s: policy_chooser(pols[s.id]), FakeLLM(world), world,
                             mode="persistent", reward=load_reward(), out_dir=out, progress=False))
    return read_logs(out)


@pytest.fixture(scope="session")
def bm(world, fake_logs):
    ds = build_datasets(fake_logs, world)
    return BehaviorModel.fit(ds.Xa, ds.ya, ds.Xm, ds.ym, ds.Xl, ds.yl, C=1.0)


def user_with_stale(world):
    for u in world.split_users("train"):
        for k in u.changing_keys:
            if stale_records(u.initial_store, k):
                return u, k
    raise RuntimeError("no user with an old version")


def query_turn(u, key, idx=0, text=None) -> Turn:
    cur = current_record(u.initial_store, key)
    olds = tuple(r.value for r in stale_records(u.initial_store, key))
    return Turn(idx, "query", text or f"What is {u.spec(key).np}?", key, cur.value, olds, frozenset({cur.id}))


def other_record(u, key):
    """A record of a different key whose value is not also some other key's value."""
    return next(r for r in u.initial_store if r.key != key)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
