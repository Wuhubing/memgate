"""One-step planners on the fitted behaviour model: greedy and amortized greedy (plan §3.11).

greedy           : argmax of this turn's expected reward.
amortized greedy : the same, but a newly injected memory's token cost is charged for every turn it
                   will stay in context, lam * tokens * (T - t). In ephemeral mode it equals greedy.
"""
from __future__ import annotations

import numpy as np

from memgate.config import RewardConfig
from memgate.env.gate_env import canonical_action, decode_action
from memgate.policies.heuristics import Policy


def expected_reward(model, st, turn, chosen_ids: list[str], sims, user_values, cfg: RewardConfig,
                    amortize: bool) -> float:
    p_ans, p_m = model.probs(st, turn, chosen_ids, sims, user_values)
    e = p_ans.get("correct", 0.0) - cfg.delta * p_ans.get("stale", 0.0)
    e -= cfg.alpha * sum(p * cfg.beta ** st.intrusions.get(i, 0) for i, p in p_m.items())
    recs = st.by_id()
    ctx = st.effective_context(chosen_ids)
    tok_now = sum(recs[i].tokens for i in ctx)
    if amortize and st.mode == "persistent":
        new = [i for i in chosen_ids if i not in st.in_context]
        tok_now += sum(recs[i].tokens for i in new) * (st.T - st.t - 1)
    return e - cfg.lam * tok_now


class ModelGreedy(Policy):
    def __init__(self, model, backend, cfg: RewardConfig, amortize: bool = False):
        self.model = model
        self.backend = backend      # SimBackend: provides sims and user values for a turn
        self.cfg = cfg
        self.amortize = amortize
        self.name = "amortized_greedy" if amortize else "model_greedy"

    def act(self, obs, info) -> int:
        st, cands = info["state"], info["cands"]
        turn = st.turn
        sims, uv = self.backend.context(st, turn)
        best, best_v = 0, -np.inf
        for a in sorted({canonical_action(a, cands) for a in range(8)}):
            ids = [r.id for r in decode_action(a, cands)]
            v = expected_reward(self.model, st, turn, ids, sims, uv, self.cfg, self.amortize)
            if v > best_v + 1e-12:
                best, best_v = a, v
        return best
