"""Turns a reply into the four reward terms by string matching only (plan §3.6, §1.6)."""
from __future__ import annotations

from memgate.config import RewardConfig
from memgate.env.state import SessionState
from memgate.schema import Answer, MemoryRecord, Mode, Outcome, Turn
from memgate.text import normalize
from memgate.world.aliases import AliasIndex


def classify_answer(turn: Turn, values_said: set[str]) -> tuple[Answer, bool]:
    """correct if the current value appears ("it was X, now Y" counts, C4); else stale if an old
    value appears; else neither. Also returns `hedged` (both appear, C5)."""
    if turn.kind != "query":
        return "na", False
    cur = turn.answer_value in values_said
    old = any(v in values_said for v in turn.stale_values)
    if cur:
        return "correct", old
    return ("stale" if old else "neither"), False


def mentioned_ids(st: SessionState, turn: Turn, values_said: set[str]) -> frozenset[str]:
    pool = list(st.store)
    if turn.new_record is not None:
        pool.append(turn.new_record)
    return frozenset(r.id for r in pool if r.value in values_said)


def intrusion_ids(st: SessionState, turn: Turn, mentioned: frozenset[str] | set[str], index: AliasIndex
                  ) -> frozenset[str]:
    """Unrelated mentions: not needed this turn, not the turn's own key, and not something the user
    just said (C6, C7)."""
    recs = st.by_id()
    if turn.new_record is not None:
        recs[turn.new_record.id] = turn.new_record
    user_vals = index.values_in(turn.text)
    out = set()
    for i in mentioned:
        r = recs.get(i)
        if r is None or i in turn.relevant_ids or r.key == turn.target_key or r.value in user_vals:
            continue
        out.add(i)
    return frozenset(out)


def compute_reward(st: SessionState, turn: Turn, chosen: list[MemoryRecord], answer: Answer,
                   mentioned: frozenset[str], intrusions: frozenset[str], cfg: RewardConfig,
                   hedged: bool = False) -> Outcome:
    """The one reward function used by both backends (plan §2.4)."""
    r_quality = 1.0 if answer == "correct" else 0.0
    r_stale = -cfg.delta if answer == "stale" else 0.0
    # n_m is the number of *earlier* unrelated mentions; it is incremented after scoring (§1.6)
    r_intr = -cfg.alpha * sum(cfg.beta ** st.intrusions.get(i, 0) for i in intrusions)
    recs = st.by_id()
    ctx = st.effective_context([r.id for r in chosen])
    r_tok = -cfg.lam * sum(recs[i].tokens if i in recs else 0 for i in ctx)
    return Outcome(answer, frozenset(mentioned), frozenset(intrusions), r_quality, r_stale, r_intr, r_tok, hedged)


def score_response(st: SessionState, turn: Turn, chosen: list[MemoryRecord], response: str,
                   cfg: RewardConfig, index: AliasIndex) -> Outcome:
    values = index.values_in(normalize(response), normalized=True)
    answer, hedged = classify_answer(turn, values)
    ment = mentioned_ids(st, turn, values)
    intr = intrusion_ids(st, turn, ment, index)
    return compute_reward(st, turn, chosen, answer, ment, intr, cfg, hedged)


def token_cost(st: SessionState, chosen_ids: list[str], lam: float, mode: Mode | None = None) -> float:
    recs = st.by_id()
    return lam * sum(recs[i].tokens for i in st.effective_context(chosen_ids) if i in recs)
