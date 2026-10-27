"""The small instance on which DP is exact (原文档 模块 7, plan §3.11).

Five records: A_cur, A_stale, B_cur, B_stale (two changing keys, each with an old version that may
or may not exist at the start) and one trivia record R. Six request kinds. Each kind has a fixed
candidate list. Probabilities come from the same BehaviorModel as the simulator: an abstract state
is turned into a small concrete StateView and fed through the very same feature functions (G1).

Approximations (written down so the report can state them):
- the turn a memory entered the context is not tracked; every in-context memory is treated as
  injected one turn ago;
- an update of key A turns the current A record into the stale one (it keeps its context and
  mention status) and creates a fresh current record; an older stale A record is merged into it;
- mention counts are capped at 2.
"""
from __future__ import annotations

from typing import NamedTuple

import gymnasium as gym
import numpy as np

from memgate.schema import MemoryRecord, Turn
from memgate.sim.outcome_features import StateView

RECS = ("A_cur", "A_stale", "B_cur", "B_stale", "R")
A_CUR, A_STALE, B_CUR, B_STALE, R = range(5)
KINDS = ("qA", "qB", "chatA", "chatX", "uA", "uB")
TOKENS = 14
CAND = {
    "qA": (A_CUR, A_STALE, R), "qB": (B_CUR, B_STALE, R),
    "chatA": (A_CUR, A_STALE, R), "chatX": (R, A_CUR, B_CUR),
    "uA": (A_CUR, A_STALE, R), "uB": (B_CUR, B_STALE, R),
}
# cosine(query, record) by request kind; stands in for the log medians of each request type
SIM = {
    "qA": (0.7, 0.7, 0.2, 0.2, 0.1), "qB": (0.2, 0.2, 0.7, 0.7, 0.1),
    "chatA": (0.5, 0.5, 0.1, 0.1, 0.1), "chatX": (0.1, 0.1, 0.1, 0.1, 0.15),
    "uA": (0.7, 0.7, 0.2, 0.2, 0.1), "uB": (0.2, 0.2, 0.7, 0.7, 0.1),
}


def kind_probs(p_kind: dict[str, float]) -> dict[str, float]:
    q, c, u = p_kind["query"], p_kind["chat"], p_kind["update"]
    tot = q + c + u
    return {"qA": q / 2 / tot, "qB": q / 2 / tot, "chatA": c / 2 / tot, "chatX": c / 2 / tot,
            "uA": u / 2 / tot, "uB": u / 2 / tot}


class AbsState(NamedTuple):
    t: int
    ctx: tuple[bool, ...]      # persistent: in context; ephemeral: ever injected
    ment: tuple[int, ...]      # unrelated mentions so far, capped at 2
    upd: tuple[bool, bool]     # A / B updated this session (the user said the new value)
    ex: tuple[bool, bool]      # A_stale / B_stale exists


def initial_states(p_stale: float) -> list[tuple[float, AbsState]]:
    out = []
    for ea in (False, True):
        for eb in (False, True):
            p = (p_stale if ea else 1 - p_stale) * (p_stale if eb else 1 - p_stale)
            if p > 0:
                out.append((p, AbsState(0, (False,) * 5, (0,) * 5, (False, False), (ea, eb))))
    return out


def exists(s: AbsState, i: int) -> bool:
    return not ((i == A_STALE and not s.ex[0]) or (i == B_STALE and not s.ex[1]))


def candidates(s: AbsState, kind: str, mode: str) -> tuple[int | None, ...]:
    out = []
    for i in CAND[kind]:
        ok = exists(s, i) and not (mode == "persistent" and s.ctx[i])
        out.append(i if ok else None)
    return tuple(out)


def chosen_of(a: int, cands) -> tuple[int, ...]:
    return tuple(c for j, c in enumerate(cands) if (a >> j) & 1 and c is not None)


def distinct_actions(cands) -> list[int]:
    return sorted({sum(1 << j for j, c in enumerate(cands) if (a >> j) & 1 and c is not None) for a in range(8)})


_KEY = {A_CUR: "A", A_STALE: "A", B_CUR: "B", B_STALE: "B", R: "R"}
_RECORDS = {i: MemoryRecord(RECS[i], "abs", _KEY[i], f"v_{RECS[i]}", RECS[i],
                            "trivia" if i == R else "schedule", _KEY[i], 0, None, TOKENS) for i in range(5)}


def turn_of(s: AbsState, kind: str) -> Turn:
    if kind in ("qA", "qB"):
        key = kind[1]
        cur, old = (A_CUR, A_STALE) if key == "A" else (B_CUR, B_STALE)
        stale = (f"v_{RECS[old]}",) if exists(s, old) else ()
        return Turn(s.t, "query", kind, key, f"v_{RECS[cur]}", stale, frozenset({RECS[cur]}))
    if kind == "chatA":
        return Turn(s.t, "chat", kind, topic_key="A")
    if kind == "chatX":
        return Turn(s.t, "chat", kind)
    return Turn(s.t, "update", kind, kind[1])


def view_of(s: AbsState, T: int, mode: str) -> StateView:
    store = [_RECORDS[i] for i in range(5) if exists(s, i)]
    ctx = {RECS[i]: s.t - 1 for i in range(5) if s.ctx[i]}
    return StateView(
        t=s.t, T=T, mode=mode, store=store,
        in_context=dict(ctx) if mode == "persistent" else {},
        injected_at={k: [v] for k, v in ctx.items()},
        mentions={RECS[i]: s.ment[i] for i in range(5) if s.ment[i]},
        intrusions={RECS[i]: s.ment[i] for i in range(5) if s.ment[i]},
        updated_keys={k for k, u in zip("AB", s.upd, strict=True) if u})


def next_state(s: AbsState, kind: str, chosen: tuple[int, ...], intruded: tuple[int, ...], mode: str) -> AbsState:
    ctx = list(s.ctx)
    for i in chosen:
        ctx[i] = True
    ment = list(s.ment)
    for i in intruded:
        ment[i] = min(ment[i] + 1, 2)
    upd, ex = list(s.upd), list(s.ex)
    if kind in ("uA", "uB"):
        k = 0 if kind == "uA" else 1
        cur, old = (A_CUR, A_STALE) if k == 0 else (B_CUR, B_STALE)
        ctx[old] = ctx[cur] or (ctx[old] and ex[k])
        ment[old] = max(ment[cur], ment[old] if ex[k] else 0)
        ctx[cur], ment[cur] = False, 0
        ex[k], upd[k] = True, True
    return AbsState(s.t + 1, tuple(ctx), tuple(ment), (upd[0], upd[1]), (ex[0], ex[1]))


class AbstractEnv(gym.Env):
    """The small instance as a gym env, for the 'DQN gamma=0 ~ greedy, gamma=1 ~ DP' check."""

    OBS = 1 + 5 + 5 + 2 + 2 + 6 + 3

    def __init__(self, solver, cfg, p_stale: float, seed: int = 0):
        self.solver = solver
        self.cfg = cfg
        self.p_stale = p_stale
        self.rng = np.random.default_rng(seed)
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (self.OBS,), np.float32)

    def obs_of(self, s: AbsState, kind: str) -> np.ndarray:
        cands = candidates(s, kind, self.solver.mode)
        x = [s.t / self.solver.T, *map(float, s.ctx), *(m / 2 for m in s.ment), *map(float, s.upd),
             *map(float, s.ex), *(float(kind == k) for k in KINDS), *(float(c is not None) for c in cands)]
        return np.array(x, dtype=np.float32)

    def _draw_kind(self) -> str:
        p = self.solver.p_kind
        return KINDS[self.rng.choice(len(KINDS), p=[p[k] for k in KINDS])]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        init = initial_states(self.p_stale)
        j = self.rng.choice(len(init), p=[p for p, _ in init])
        self.s, self.kind = init[j][1], self._draw_kind()
        return self.obs_of(self.s, self.kind), {"s": self.s, "kind": self.kind}

    def step(self, a):
        outs = self.solver.outcomes(self.s, self.kind, int(a))
        j = self.rng.choice(len(outs), p=[o[0] for o in outs])
        _, answer, intruded, nxt = outs[j]
        r = float(sum(self.solver.components(self.s, self.kind, int(a), answer, intruded, self.cfg)))
        self.s = nxt
        done = nxt.t >= self.solver.T
        if not done:
            self.kind = self._draw_kind()
            obs = self.obs_of(self.s, self.kind)
        else:
            obs = np.zeros(self.OBS, dtype=np.float32)
        return obs, r, done, False, {"s": self.s, "kind": self.kind}
