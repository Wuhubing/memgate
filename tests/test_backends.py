"""Live and sim produce outcomes through the same reward function and state update (plan §2.4)."""
from __future__ import annotations

import asyncio
import json

import pytest

from memgate.env.backends import StepResult
from memgate.env.gate_env import MemGateEnv
from memgate.harness.llm_client import FakeLLM
from memgate.harness.runner import policy_chooser, run_session
from memgate.policies.heuristics import RandomPolicy
from memgate.retrieval.retriever import Retriever
from memgate.world.generator import SessionFactory


class ReplayBackend:
    """Feeds logged (answer, mentioned, intrusions) back into the gym environment."""

    def __init__(self, rows):
        self.rows = {r["t"]: r for r in rows}

    def step(self, st, turn, chosen):
        r = self.rows[turn.idx]
        assert json.loads(r["injected_ids"]) == [c.id for c in chosen]
        return StepResult(r["answer"], frozenset(json.loads(r["mentioned_ids"])),
                          frozenset(json.loads(r["intrusion_ids"])), bool(r["hedged"]))


@pytest.mark.parametrize("mode", ["persistent", "ephemeral"])
def test_live_rows_replay_identically_in_env(world, reward, mode):
    fac = SessionFactory(world, "train", seed=8)
    for i in range(10):
        s = fac.make(10)
        rows = asyncio.run(run_session(s, policy_chooser(RandomPolicy(i)), FakeLLM(world), world, mode=mode,
                                       reward=reward, retriever=Retriever(world.emb)))
        env = MemGateEnv(world, ReplayBackend(rows), mode=mode, reward=reward, scripts=[s])
        obs, info = env.reset(options={"script": s})
        for r in rows:
            assert json.loads(r["state_json"]) == env.st.snapshot()
            assert [round(float(v), 6) for v in obs] == json.loads(r["obs"])
            obs, rew, term, trunc, info = env.step(r["action"])
            assert rew == pytest.approx(r["reward"])
        assert term
