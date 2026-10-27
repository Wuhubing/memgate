"""Exact finite-horizon DP and exact policy evaluation on the small instance (原文档 模块 7).

V_t(s) = sum_k p(k) max_a Q_t(s,k,a),
Q_t(s,k,a) = sum_o P(o|s,k,a) [ r(s,k,a,o) + gamma V_{t+1}(f(s,k,a,o)) ].

Outcome tables depend only on the behaviour model, so they are cached once and shared by every
reward variant, every policy and every stale ratio (the stale ratio only changes the start state).
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from memgate.config import RewardConfig
from memgate.planning.abstract import (
    KINDS,
    RECS,
    SIM,
    TOKENS,
    AbsState,
    candidates,
    chosen_of,
    distinct_actions,
    initial_states,
    kind_probs,
    next_state,
    turn_of,
    view_of,
)
from memgate.sim.outcome_features import (
    ANSWER_CLASSES,
    ANSWER_FEATURES,
    MENTION_FEATURES,
    answer_features,
    mention_candidates,
    mention_features,
    vec,
)

COMPONENTS = ("quality", "stale", "intrusion", "tokens")
AbsPolicy = Callable[[AbsState, str], int]


@dataclass
class Table:
    p: np.ndarray            # (n,) outcome probabilities
    correct: np.ndarray      # (n,) 0/1
    stale: np.ndarray        # (n,) 0/1
    intr: list[tuple[int, ...]]        # intruded record indices per outcome
    intr_ment: list[tuple[int, ...]]   # their mention counts before this turn (n_m)
    nexts: list[AbsState]
    n_ctx: int               # memories in context after this action (what lambda charges)
    n_new: int               # newly injected memories (persistent)


def _cfg_key(cfg: RewardConfig) -> tuple:
    return (cfg.delta, cfg.alpha, cfg.beta, cfg.lam)


class SmallInstance:
    def __init__(self, model, T: int, mode: str, p_kind: dict[str, float]):
        self.model = model
        self.T = T
        self.mode = mode
        self.p_kind = kind_probs(p_kind)
        self._tab: dict[tuple, Table] = {}
        self._rew: dict[tuple, np.ndarray] = {}
        self._acache: dict[tuple, dict[str, float]] = {}
        self._mcache: dict[tuple, list[tuple[float, tuple[int, ...]]]] = {}

    # ---- model -------------------------------------------------------------------------------
    def table(self, s: AbsState, kind: str, a: int) -> Table:
        key = (s, kind, a)
        hit = self._tab.get(key)
        if hit is not None:
            return hit
        cands = candidates(s, kind, self.mode)
        chosen = chosen_of(a, cands)
        p_ans = self._answer_probs(s, kind, chosen)
        subsets = self._mention_subsets(s, kind, chosen)
        P, C, S, I, IM, N = [], [], [], [], [], []
        nxt_cache: dict[tuple[int, ...], AbsState] = {}
        for ans, pa in p_ans.items():
            if pa <= 0:
                continue
            for pm, intr in subsets:
                p = pa * pm
                if p <= 0:
                    continue
                P.append(p)
                C.append(ans == "correct")
                S.append(ans == "stale")
                I.append(intr)
                IM.append(tuple(s.ment[i] for i in intr))
                nx = nxt_cache.get(intr)
                if nx is None:
                    nx = nxt_cache[intr] = next_state(s, kind, chosen, intr, self.mode)
                N.append(nx)
        if self.mode == "persistent":
            n_new = sum(1 for i in chosen if not s.ctx[i])
            n_ctx = sum(s.ctx) + n_new
        else:
            n_ctx = n_new = len(chosen)
        t = Table(np.array(P), np.array(C, float), np.array(S, float), I, IM, N, n_ctx, n_new)
        self._tab[key] = t
        return t

    # The answer model depends on t (t_norm); the mention model does not, so mention outcome
    # distributions are shared by every t with the same context/mention/update bits.
    def _answer_probs(self, s: AbsState, kind: str, chosen: tuple[int, ...]) -> dict[str, float]:
        key = (s, kind, chosen)
        hit = self._acache.get(key)
        if hit is None:
            turn = turn_of(s, kind)
            if turn.kind != "query":
                hit = {"na": 1.0}
            else:
                view = view_of(s, self.T, self.mode)
                pa = self.model.answer_probs_x(vec(answer_features(view, turn, [RECS[i] for i in chosen]),
                                                   ANSWER_FEATURES)[None])[0]
                if not turn.stale_values:
                    pa[ANSWER_CLASSES.index("stale")] = 0.0
                    pa = pa / pa.sum()
                hit = dict(zip(ANSWER_CLASSES, pa.tolist(), strict=True))
            self._acache[key] = hit
        return hit

    def _mention_subsets(self, s: AbsState, kind: str, chosen: tuple[int, ...]):
        key = (s.ctx, s.ment, s.upd, s.ex, kind, chosen)
        hit = self._mcache.get(key)
        if hit is None:
            view = view_of(s, self.T, self.mode)
            turn = turn_of(s, kind)
            ids = [RECS[i] for i in chosen]
            cands = mention_candidates(view, turn, ids, set())
            idx = {r: i for i, r in enumerate(RECS)}
            if cands:
                X = np.stack([vec(mention_features(view, turn, ids, r, SIM[kind][idx[r.id]]), MENTION_FEATURES)
                              for r in cands])
                pm = self.model.mention_probs_x(X)
            else:
                pm = np.zeros(0)
            elig = [(idx[r.id], float(p)) for r, p in zip(cands, pm, strict=True)]
            hit = []
            for bits in itertools.product((0, 1), repeat=len(elig)):
                p, intr = 1.0, []
                for (i, q), b in zip(elig, bits, strict=True):
                    p *= q if b else 1.0 - q
                    if b:
                        intr.append(i)
                if p > 0:
                    hit.append((p, tuple(intr)))
            self._mcache[key] = hit
        return hit

    def outcomes(self, s: AbsState, kind: str, a: int) -> list[tuple[float, str, tuple[int, ...], AbsState]]:
        """(p, answer, intruded record indices, next state) — used by AbstractEnv sampling."""
        t = self.table(s, kind, a)
        out = []
        for j in range(len(t.p)):
            ans = "correct" if t.correct[j] else "stale" if t.stale[j] else ("neither" if kind[0] == "q" else "na")
            out.append((float(t.p[j]), ans, t.intr[j], t.nexts[j]))
        return out

    def rewards(self, s: AbsState, kind: str, a: int, cfg: RewardConfig, amortize: bool = False) -> np.ndarray:
        """(n, 4) reward components per outcome."""
        key = (s, kind, a, _cfg_key(cfg), amortize)
        hit = self._rew.get(key)
        if hit is not None:
            return hit
        t = self.table(s, kind, a)
        intr = np.array([-cfg.alpha * sum(cfg.beta ** m for m in im) for im in t.intr_ment])
        n_tok = t.n_ctx + (t.n_new * (self.T - s.t - 1) if amortize and self.mode == "persistent" else 0)
        r = np.stack([t.correct, -cfg.delta * t.stale, intr, np.full(len(t.p), -cfg.lam * TOKENS * n_tok)], axis=1)
        self._rew[key] = r
        return r

    def components(self, s, kind, a, answer, intruded, cfg: RewardConfig, amortize: bool = False) -> np.ndarray:
        """Reward components of one specific outcome (AbstractEnv)."""
        q = 1.0 if answer == "correct" else 0.0
        st = -cfg.delta if answer == "stale" else 0.0
        intr = -cfg.alpha * sum(cfg.beta ** s.ment[i] for i in intruded)
        t = self.table(s, kind, a)
        n_tok = t.n_ctx + (t.n_new * (self.T - s.t - 1) if amortize and self.mode == "persistent" else 0)
        return np.array([q, st, intr, -cfg.lam * TOKENS * n_tok])

    def expected_immediate(self, s, kind, a, cfg, amortize=False) -> float:
        t = self.table(s, kind, a)
        return float(t.p @ self.rewards(s, kind, a, cfg, amortize).sum(axis=1))

    # ---- planning ----------------------------------------------------------------------------
    def solve(self, cfg: RewardConfig, gamma: float = 1.0) -> AbsPolicy:
        """Optimal policy by backward induction (memoised recursion over reachable states)."""
        V: dict[AbsState, float] = {}
        best: dict[tuple[AbsState, str], int] = {}

        def v(s: AbsState) -> float:
            if s.t >= self.T:
                return 0.0
            got = V.get(s)
            if got is not None:
                return got
            tot = 0.0
            for kind in KINDS:
                pk = self.p_kind[kind]
                if pk == 0:
                    continue
                q_best, a_best = -np.inf, 0
                for a in distinct_actions(candidates(s, kind, self.mode)):
                    t = self.table(s, kind, a)
                    r = self.rewards(s, kind, a, cfg).sum(axis=1)
                    if gamma:
                        r = r + gamma * np.array([v(n) for n in t.nexts])
                    q = float(t.p @ r)
                    if q > q_best + 1e-12:
                        q_best, a_best = q, a
                best[(s, kind)] = a_best
                tot += pk * q_best
            V[s] = tot
            return tot

        for _, s0 in initial_states(0.5):
            v(s0)

        def pi(s: AbsState, kind: str) -> int:
            if (s, kind) not in best:
                v(s)
            return best[(s, kind)]
        return pi

    def greedy(self, cfg: RewardConfig, amortize: bool = False) -> AbsPolicy:
        cache: dict = {}

        def pi(s: AbsState, kind: str) -> int:
            k = (s, kind)
            if k not in cache:
                acts = distinct_actions(candidates(s, kind, self.mode))
                vals = [self.expected_immediate(s, kind, a, cfg, amortize) for a in acts]
                cache[k] = acts[int(np.argmax(vals))]   # ties -> fewest injections (acts is sorted)
            return cache[k]
        return pi

    def always(self) -> AbsPolicy:
        return lambda s, kind: 7

    def never(self) -> AbsPolicy:
        return lambda s, kind: 0

    # ---- exact evaluation --------------------------------------------------------------------
    def evaluate_many(self, pi: AbsPolicy, cfg: RewardConfig, p_stales: list[float]) -> dict[float, np.ndarray]:
        """Expected undiscounted return of `pi`, split into the four components, for each stale ratio."""
        memo: dict[AbsState, np.ndarray] = {}

        def v(s: AbsState) -> np.ndarray:
            if s.t >= self.T:
                return np.zeros(4)
            got = memo.get(s)
            if got is not None:
                return got
            tot = np.zeros(4)
            for kind in KINDS:
                pk = self.p_kind[kind]
                if pk == 0:
                    continue
                a = pi(s, kind)
                t = self.table(s, kind, a)
                r = self.rewards(s, kind, a, cfg) + np.stack([v(n) for n in t.nexts])
                tot += pk * (t.p @ r)
            memo[s] = tot
            return tot

        return {p: sum(w * v(s0) for w, s0 in initial_states(p)) for p in p_stales}

    def evaluate(self, pi: AbsPolicy, cfg: RewardConfig, p_stale: float) -> np.ndarray:
        return self.evaluate_many(pi, cfg, [p_stale])[p_stale]

    @property
    def n_states(self) -> int:
        return len({k[0] for k in self._tab})
