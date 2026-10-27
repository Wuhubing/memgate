"""Two interchangeable ways of producing one turn's outcome (plan §2.4).

Both return the same pieces; the environment then calls the one shared reward function and the one
shared state update, so the backends cannot drift apart (tested in test_backends.py).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from memgate.env.state import SessionState
from memgate.harness.prompt import SYSTEMS, build_messages
from memgate.harness.scorer import classify_answer, intrusion_ids, mentioned_ids
from memgate.schema import MemoryRecord, Turn
from memgate.text import normalize


@dataclass
class StepResult:
    answer: str
    mentioned: frozenset[str]
    intrusions: frozenset[str]
    hedged: bool = False
    response: str | None = None
    stored_user: str | None = None
    llm_meta: dict | None = None


class SimBackend:
    def __init__(self, world, model, seed: int = 0):
        self.world = world
        self.model = model
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    @lru_cache(maxsize=200_000)  # noqa: B019 - backend objects live for a whole run
    def _user_values(self, user_id: str, text: str) -> frozenset[str]:
        return frozenset(self.world.user(user_id).index.values_in(text))

    def sims(self, st: SessionState, turn: Turn) -> dict[str, float]:
        q = self.world.emb.get(turn.text)
        return {r.id: float(self.world.emb.get(r.text) @ q) for r in st.store}

    def context(self, st: SessionState, turn: Turn) -> tuple[dict[str, float], set[str]]:
        return self.sims(st, turn), set(self._user_values(st.script.user_id, turn.text))

    def step(self, st: SessionState, turn: Turn, chosen: list[MemoryRecord]) -> StepResult:
        sims, uv = self.context(st, turn)
        ids = [r.id for r in chosen]
        answer, ment = self.model.sample(st, turn, ids, sims, uv, self.rng)
        intr = intrusion_ids(st, turn, ment, self.world.user(st.script.user_id).index)
        return StepResult(answer, ment, intr)


class LiveBackend:
    """Calls the real LLM one turn at a time. For debugging inside the env; batch runs use the runner."""

    def __init__(self, world, llm, prompt_version: str = "v1", sample_idx: int = 0):
        self.world = world
        self.llm = llm
        self.system = SYSTEMS[prompt_version]
        self.sample_idx = sample_idx

    def step(self, st: SessionState, turn: Turn, chosen: list[MemoryRecord]) -> StepResult:
        msgs, stored = build_messages(self.system, st.history, turn.text, chosen, st.mode)
        resp = asyncio.run(self.llm.chat(msgs, session_id=st.script.id, turn_idx=turn.idx,
                                         sample_idx=self.sample_idx,
                                         meta={"turn": turn, "user_id": st.script.user_id}))
        return result_from_reply(self.world, st, turn, resp.text, stored, resp)


def result_from_reply(world, st: SessionState, turn: Turn, text: str, stored: str, resp=None) -> StepResult:
    index = world.user(st.script.user_id).index
    values = index.values_in(normalize(text), normalized=True)
    answer, hedged = classify_answer(turn, values)
    ment = mentioned_ids(st, turn, values)
    intr = intrusion_ids(st, turn, ment, index)
    meta = None
    if resp is not None:
        meta = {k: getattr(resp, k) for k in ("model", "system_fingerprint", "prompt_tokens", "completion_tokens",
                                              "cache_hit", "cache_miss", "finish_reason", "latency_s",
                                              "from_cache")}
    return StepResult(answer, ment, intr, hedged, text, stored, meta)
